#!/usr/bin/env python3
"""
compare_cpa_models.py

Automatically discovers CPA model checkpoints in a directory by naming pattern,
computes metrics for each, and then averages per configuration (original vs. intense).

Usage:
    python compare_cpa_models.py \
      --models_dir /path/to/pretrained_models \
      --adata /path/to/data.h5ad \
      --output_dir ./results \
      --control_group ctrl \
      --split_key split \
      --train_split train \
      --valid_split test \
      --ood_split ood \
      --intense_reg_rate 0.01 \
      --intense_p 2
"""
import os
import time
import argparse
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    silhouette_score,
    adjusted_mutual_info_score,
    accuracy_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

import scanpy as sc
from cpa import CPA
from cpa._utils import CPA_REGISTRY_KEYS

# Global control group label
CONTROL_GROUP = None

def compute_prediction_accuracy(adata_true, adata_pred, top_degs=None):
    """
    Compute R² on mean and variance, and gene-wise Wasserstein-1.
    If top_degs is None, uses all genes; else uses the provided list of gene indices.
    """
    X_true = adata_true.X.toarray() if hasattr(adata_true.X, 'toarray') else adata_true.X
    X_pred = adata_pred
    # filter out control
    if CONTROL_GROUP is not None:
        pg = adata_true.obs[CPA_REGISTRY_KEYS.PERTURBATION_KEY].values
        mask = pg != CONTROL_GROUP
        X_true = X_true[mask]
        X_pred = X_pred[mask]

    # log-transform
    X_true = np.log1p(X_true)
    X_pred = np.log1p(X_pred)

    if top_degs is not None:
        X_true = X_true[:, top_degs]
        X_pred = X_pred[:, top_degs]

    # mean R2
    r2_mean = r2_score(X_true.mean(0), X_pred.mean(0))
    # var R2
    r2_var = r2_score(X_true.var(0), X_pred.var(0))
    # gene-wise Wasserstein
    wds = [wasserstein_distance(X_true[:, i], X_pred[:, i]) for i in range(X_true.shape[1])]
    wd_mean = np.mean(wds)
    return {f"r2_mean_{len(top_degs) if top_degs is not None else 'all'}": r2_mean,
            f"r2_var_{len(top_degs) if top_degs is not None else 'all'}": r2_var,
            f"wd_mean_{len(top_degs) if top_degs is not None else 'all'}": wd_mean}


def compute_reconstruction_loss(model, adata, split_key, train_split):
    # MSE between true and predicted on in-distribution (train) cells
    idx = np.where(adata.obs[split_key] == train_split)[0]
    if len(idx) == 0:
        return {f"recon_mse_{train_split}": np.nan}
    adata_sub = adata[idx].copy()
    model.predict(adata_sub)
    key = f"{model.__class__.__name__}_pred"
    X_true = adata_sub.X.toarray() if hasattr(adata_sub.X, 'toarray') else adata_sub.X
    X_pred = adata_sub.obsm[key]
    mse = mean_squared_error(X_true.flatten(), X_pred.flatten())
    return {f"recon_mse_{train_split}": float(mse)}


def compute_disentanglement(model, adata, split_key="split", train_split="train", valid_split="test"):
    """
    Trains a logistic classifier on basal latent to predict perturbation.
    Returns accuracy on training and validation splits.
    """
    latents = model.get_latent_representation(adata)['latent_basal'].X
    y = adata.obs[CPA_REGISTRY_KEYS.PERTURBATION_KEY].astype(str).values
    # splits
    train_idx = np.where((adata.obs[split_key] == train_split) )[0]
    valid_idx = np.where((adata.obs[split_key] == valid_split) )[0]
    if train_idx.size < 2 or valid_idx.size < 1:
        return {"disentangle_acc_train": np.nan, "disentangle_acc_valid": np.nan}
    clf = LogisticRegression(max_iter=500)
    clf.fit(latents[train_idx], y[train_idx])
    acc_train = accuracy_score(y[train_idx], clf.predict(latents[train_idx]))
    acc_valid = accuracy_score(y[valid_idx], clf.predict(latents[valid_idx]))
    return {"disentangle_acc_train": float(acc_train), "disentangle_acc_valid": float(acc_valid)}


def compute_ood_metrics(model, adata, split_key, ood_split):
    idx = np.where(adata.obs[split_key] == ood_split)[0]
    if idx.size == 0:
        return {}
    adata_ood = adata[idx].copy()
    model.predict(adata_ood)
    pred_key = f"{model.__class__.__name__}_pred"
    # gather OOD prediction metrics and prefix keys with 'ood_'
    raw = compute_prediction_accuracy(adata_ood, adata_ood.obsm[pred_key], top_degs=None)
    mets = {f"ood_{k}": v for k, v in raw.items()}
    # compute uncertainty and prefix
    z = model.get_latent_representation(adata)['latent_after'].X
    seen_idx = np.where(adata.obs[split_key] != ood_split)[0]
    nn = NearestNeighbors(metric="cosine").fit(z[seen_idx])
    dists, _ = nn.kneighbors(z[idx], n_neighbors=1)
    mets['ood_uncertainty_cosine'] = float(np.mean(dists))
    return mets


def compute_latent_structure(model, adata, label_key="perturbation", cmap_key="cell_type"):
    """
    Silhouette & AMI on corrected latent; MOA recovery if available.
    """
    z = model.get_latent_representation(adata)['latent_after'].X
    labels = adata.obs[CPA_REGISTRY_KEYS.PERTURBATION_KEY].astype(str).values
    sil = silhouette_score(z, labels)
    ami = adjusted_mutual_info_score(labels, labels)  # self-comparison yields 1.0
    # if MOA annotation provided:
    if 'moa' in adata.obs:
        conds = adata.obs[CPA_REGISTRY_KEYS.PERTURBATION_KEY].values
        moa = adata.obs['moa'].values
        # build one vector per condition
        uniq = np.unique(conds)
        emb = np.array([z[conds==u].mean(0) for u in uniq])
        moa_map = {u: moa[conds==u][0] for u in uniq}
        nn = NearestNeighbors(n_neighbors=5, metric='cosine').fit(emb)
        rec = []
        for i,u in enumerate(uniq):
            d, idxs = nn.kneighbors(emb[i:i+1])
            rec.append(np.mean([moa_map[uniq[j]]==moa_map[u] for j in idxs[0]]))
        moa_rec = float(np.mean(rec))
    else:
        moa_rec = np.nan
    return {"silhouette": sil, "ami": ami, "moa_recovery": moa_rec}


def compute_efficiency(model, adata):
    """
    Measures inference time and (if GPU) peak memory.
    """
    torch.cuda.reset_max_memory_allocated()
    t0 = time.time()
    model.predict(adata)
    t1 = time.time()
    inf_time = t1 - t0
    mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else np.nan
    return {"inference_time_s": inf_time, "peak_mem_bytes": mem}

def main():
    global CONTROL_GROUP
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="/scratch/nmbiedou/experiment_new/Kang_Order_2_tf4",
                        help="Directory containing pretrained model subfolders")
    parser.add_argument("--adata", default="/home/nmbiedou/Documents/cpa/datasets/kang_normalized_hvg.h5ad",
                        help="Path to input AnnData (.h5ad)")
    parser.add_argument("--output_dir", default="/home/nmbiedou/Documents/cpa/metrics")
    parser.add_argument("--control_group", default="ctrl",
                        help="Label of control cells to exclude")
    parser.add_argument("--split_key", default="split_B",
                        help="AnnData.obs column for splits")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--valid_split", default="valid")
    parser.add_argument("--ood_split", default="ood")
    parser.add_argument("--intense_reg_rate", type=float, default=0.01,
                        help="Regularization rate used in intense models")
    parser.add_argument("--intense_p", type=int, default=2,
                        help="Interaction order used in intense models")
    args = parser.parse_args()

    CONTROL_GROUP = args.control_group
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data and setup AnnData
    adata = sc.read(args.adata)
    adata.X = adata.layers['counts'].copy()
    adata.obs['dose'] = adata.obs['condition'].apply(lambda x: '+'.join(['1.0' for _ in x.split('+')]))
    CPA.setup_anndata(
        adata,
        perturbation_key='condition',
        control_group=CONTROL_GROUP,
        dosage_key='dose',
        categorical_covariate_keys=['cell_type'],
        is_count_data=True,
        deg_uns_key='rank_genes_groups_cov',
        deg_uns_cat_key='cov_cond',
        max_comb_len=1,
    )

    # Discover model checkpoint directories
    orig_pattern = os.path.join(
        args.models_dir,
        "Kang_Original_seed_*"
    )
    intense_pattern = os.path.join(
        args.models_dir,
        f"Kang_Intense_reg_{str(args.intense_reg_rate).replace('.', '_')}_p_{args.intense_p}_seed_*"
    )
    model_configs = {
        'original_cpa': sorted(glob.glob(orig_pattern)),
        'intense_cpa': sorted(glob.glob(intense_pattern)),
    }

    all_results = []
    for config_name, paths in model_configs.items():
        for model_path in paths:
            # load model
            torch.manual_seed(0)
            np.random.seed(0)
            model = CPA.load(model_path, adata, use_gpu=torch.cuda.is_available())
            metrics = {'model': config_name}

            # Prediction accuracy
            model.predict(adata)
            preds = adata.obsm[f"{model.__class__.__name__}_pred"]
            metrics.update(compute_prediction_accuracy(adata, preds, top_degs=None))

            # Reconstruction loss
            metrics.update(
                compute_reconstruction_loss(
                    model, adata,
                    split_key=args.split_key,
                    train_split=args.train_split
                )
            )

            # Disentanglement
            metrics.update(
                compute_disentanglement(
                    model, adata,
                    split_key=args.split_key,
                    train_split=args.train_split,
                    valid_split=args.valid_split
                )
            )

            # OOD metrics
            metrics.update(
                compute_ood_metrics(
                    model, adata,
                    split_key=args.split_key,
                    ood_split=args.ood_split
                )
            )

            # Latent structure
            metrics.update(compute_latent_structure(model, adata))

            # Efficiency
            metrics.update(compute_efficiency(model, adata))

            all_results.append(metrics)


        # Create DataFrame and average per config
        df = pd.DataFrame(all_results)
        # Compute mean and standard deviation per metric per model configuration
        stats = df.groupby('model').agg(['mean', 'std'])
        # Flatten multiindex columns: e.g. r2_mean_all_mean, r2_mean_all_std
        stats.columns = ['_'.join([metric, stat]) for metric, stat in stats.columns]
        df_stats = stats.reset_index()
        # Save statistics
        stats_csv = os.path.join(args.output_dir, "comparison_metrics_stats.csv")
        df_stats.to_csv(stats_csv, index=False)
        print(f"Saved mean and std of metrics per model config to {stats_csv}")


if __name__ == '__main__':
    main()

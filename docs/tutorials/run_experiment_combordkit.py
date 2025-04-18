#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Predicting combinatorial drug perturbations using RDKit embeddings with CPA.
In this experiment, RDKit embeddings (derived from the SMILES information) are always used.
Additionally, the user can choose between the original CPA setting and the INTENSE CPA variant.
For the INTENSE variant, the additional parameters --intense_reg_rate and --intense_p must be provided.
Usage examples:
    # Run with the INTENSE modifications:
    python script_rdkit_intense.py --seed 6478 --use_intense 1 --intense_reg_rate 0.1 --intense_p 1
    # Run the original CPA variant:
    python script_rdkit_intense.py --seed 6478 --use_intense 0
"""

import argparse
import hashlib
import os
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import scanpy as sc
import cpa
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tqdm import tqdm

from docs.tutorials.utils import append_to_csv


# =============================================================================
# Argument Parsing
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=("Train CPA on the Combo Sci-Plex dataset using RDKit embeddings "
                     "for drugs with an option to switch between original and INTENSE CPA.")
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility. If not provided a seed is generated.")
    parser.add_argument("--use_intense", type=int, choices=[0, 1], required=True,
                        help="0 for original CPA; 1 for INTENSE CPA. "
                             "INTENSE CPA requires additional parameters.")
    parser.add_argument("--intense_reg_rate", type=float, default=None,
                        help="Regularization rate for INTENSE CPA. Required if use_intense is 1.")
    parser.add_argument("--intense_p", type=int, default=None,
                        help="p value for INTENSE CPA. Required if use_intense is 1.")
    parser.add_argument("--current_dir", type=str, default="/scratch/nmbiedou",
                        help="Base directory for datasets and experiment logs.")
    return parser.parse_args()


# =============================================================================
# Setup paths based on experiment configuration.
# =============================================================================
def setup_paths(current_dir, use_intense, seed, intense_reg_rate, intense_p):
    # Assume the dataset is stored inside a "datasets" folder in current_dir.
    data_dir = "/home/nmbiedou/Documents/cpa"
    data_path = os.path.join(data_dir,"datasets","combo_sciplex_prep_hvg_filtered.h5ad")

    # Create separate experiment directories depending on whether INTENSE CPA is used.
    if use_intense:
        save_path = os.path.join(current_dir, "experiment/Combo_RDKit",
                                 f"combo_RDKit_Intense_reg_{str(intense_reg_rate).replace('.', '_')}_p_{intense_p}_seed_{seed}")
    else:
        save_path = os.path.join(current_dir, "experiment/Combo_RDKit",
                                 f"combo_RDKit_Original_seed_{seed}")
    os.makedirs(save_path, exist_ok=True)

    results_dir = os.path.join(current_dir, "experiment/Combo_RDKit", "experiment_results")
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, "experiment_log.csv")
    return data_path, save_path, log_file


# =============================================================================
# Hyperparameters and Random Seed Setup
# =============================================================================
def setup_hparams(args):
    # Generate a seed if not provided.
    if args.seed is None:
        task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
        seed_input = f"{int(time.time() * 1000)}_{task_id}"
        seed_hash = hashlib.md5(seed_input.encode()).hexdigest()
        args.seed = int(seed_hash[:4], 16) % 10000
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Base hyperparameters for CPA using RDKit embeddings (from the tutorial).
    ae_hparams = {
        "n_latent": 64,
        "recon_loss": "nb",
        "doser_type": "linear",
        "n_hidden_encoder": 256,
        "n_layers_encoder": 3,
        "n_hidden_decoder": 512,
        "n_layers_decoder": 2,
        "use_batch_norm_encoder": True,
        "use_layer_norm_encoder": False,
        "use_batch_norm_decoder": True,
        "use_layer_norm_decoder": False,
        "dropout_rate_encoder": 0.25,
        "dropout_rate_decoder": 0.25,
        "variational": False,
        "seed": args.seed,
    }
    trainer_params = {
        "n_epochs_kl_warmup": None,
        "n_epochs_pretrain_ae": 50,
        "n_epochs_adv_warmup": 100,
        "n_epochs_mixup_warmup": 10,
        "mixup_alpha": 0.1,
        "adv_steps": None,
        "n_hidden_adv": 128,
        "n_layers_adv": 3,
        "use_batch_norm_adv": False,
        "use_layer_norm_adv": False,
        "dropout_rate_adv": 0.2,
        "reg_adv": 10.0,
        "pen_adv": 0.1,
        "lr": 0.0003,
        "wd": 4e-07,
        "adv_lr": 0.0003,
        "adv_wd": 4e-07,
        "adv_loss": "cce",
        "doser_lr": 0.0003,
        "doser_wd": 4e-07,
        "do_clip_grad": False,
        "gradient_clip_value": 1.0,
        "step_size_lr": 10,
    }
    # If the INTENSE variant is enabled, update the hyperparameters accordingly.
    if args.use_intense:
        if args.intense_reg_rate is None or args.intense_p is None:
            raise ValueError("For --use_intense 1, --intense_reg_rate and --intense_p must be provided.")
        ae_hparams.update({
            "use_intense": True,
            "intense_reg_rate": args.intense_reg_rate,
            "interaction_order": 3,
            "intense_p": args.intense_p,
        })
        trainer_params = {
            "n_epochs_kl_warmup": None,
            "n_epochs_pretrain_ae": 5,
            "n_epochs_adv_warmup": 3,
            "n_epochs_mixup_warmup": 5,
            "mixup_alpha": 0.2,
            "adv_steps": 5,
            "n_hidden_adv": 64,
            "n_layers_adv": 4,
            "use_batch_norm_adv": True,
            "use_layer_norm_adv": False,
            "dropout_rate_adv": 0.3,
            "reg_adv": 46.66573744649211,
            "pen_adv": 0.012341837149804745,
            "lr": 0.004645058485717965,
            "wd": 0.00000132057350759259,
            "adv_lr": 0.00008042981766266418,
            "adv_wd": 0.00000001041516981997,
            "adv_loss": "cce",
            "doser_lr": 0.0002641376293921358,
            "doser_wd": 0.00000010439338556854,
            "do_clip_grad": False,
            "gradient_clip_value": 1,
            "step_size_lr": 25,
            "momentum": 0.8849933379739408
        }
    else:
        ae_hparams["use_intense"] = False
    return ae_hparams, trainer_params


# =============================================================================
# Main Function: Data Loading, Model Training, Visualization & Evaluation
# =============================================================================
def main():
    args = parse_args()

    # Set up hyperparameters and experiment paths.
    ae_hparams, trainer_params = setup_hparams(args)
    data_path, save_path, log_file = setup_paths(args.current_dir, args.use_intense, args.seed,
                                       args.intense_reg_rate, args.intense_p)

    # Configure scanpy figure settings.
    sc.settings.set_figure_params(dpi=100)
    sc.settings.figdir = save_path

    # Load dataset from the provided path.
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Please download and place it in the datasets folder.")

    adata = sc.read(data_path)
    print(
        f"Running experiment with seed: {args.seed}, use_intense: {args.use_intense}, intense_reg_rate: {args.intense_reg_rate}, intense_p: {args.intense_p}")

    # Log experiment details.
    task_id = os.getenv("SLURM_ARRAY_TASK_ID", "unknown")
    log_entry = {
        "task_id": task_id,
        "use_intense": args.use_intense,
        "intense_reg_rate": args.intense_reg_rate if args.use_intense else None,
        "intense_p": args.intense_p if args.use_intense else None,
        "seed": args.seed,
        "save_path": save_path,
    }
    append_to_csv(log_entry, log_file)

    # Use raw count data.
    adata.X = adata.layers["counts"].copy()

    # =============================================================================
    # Set up AnnData for CPA with RDKit embeddings
    # =============================================================================
    setup_kwargs = dict(
        perturbation_key="condition_ID",
        dosage_key="log_dose",
        control_group="CHEMBL504",
        batch_key=None,
        smiles_key="smiles_rdkit",
        is_count_data=True,
        categorical_covariate_keys=["cell_type"],
        deg_uns_key="rank_genes_groups_cov",
        deg_uns_cat_key="cov_drug_dose",
        max_comb_len=2,
    )
    cpa.CPA.setup_anndata(adata, **setup_kwargs)

    # =============================================================================
    # Create and Train the CPA Model
    # =============================================================================
    model = cpa.CPA(
        adata=adata,
        split_key="split_1ct_MEC",
        train_split="train",
        valid_split="valid",
        test_split="ood",
        use_rdkit_embeddings=True,
        **ae_hparams
    )

    model.train(
        max_epochs=2000,
        use_gpu=True,
        batch_size=512,
        plan_kwargs=trainer_params,
        early_stopping_patience=10,
        check_val_every_n_epoch=5,
        save_path=save_path,
    )

    # Plot and save training history.
    history_plot_path = os.path.join(save_path, "history.png")
    cpa.pl.plot_history(model, history_plot_path)

    results_dir = os.path.join(args.current_dir, "experiment/Combo_RDKit", "experiment_results")
    os.makedirs(results_dir, exist_ok=True)

    # =============================================================================
    # Latent Space Visualization
    # =============================================================================
    latent_outputs = model.get_latent_representation(adata, batch_size=1024)

    # UMAP for basal latent representations (pre-perturbation)
    latent_basal_adata = latent_outputs["latent_basal"]
    sc.pp.neighbors(latent_basal_adata)
    sc.tl.umap(latent_basal_adata)
    sc.pl.umap(latent_basal_adata, color=["condition_ID"],
               frameon=False, wspace=0.2, save="latent_basal.png")

    # UMAP for final latent representations (post-perturbation)
    latent_adata = latent_outputs["latent_after"]
    sc.pp.neighbors(latent_adata)
    sc.tl.umap(latent_adata)
    sc.pl.umap(latent_adata, color=["condition_ID"],
               frameon=False, wspace=0.2, save="latent_after.png")

    # =============================================================================
    # Evaluation: Prediction and Metrics Computation
    # =============================================================================
    print("Starting prediction and evaluation...")
    model.predict(adata, batch_size=1024)

    n_top_degs = [10, 20, 50, None]  # 'None' means all genes are used.
    results = defaultdict(list)
    ctrl_adata = adata[adata.obs["condition_ID"] == "CHEMBL504"].copy()

    for cat in tqdm(adata.obs["cov_drug_dose"].unique()):
        if "CHEMBL504" not in cat:
            cat_adata = adata[adata.obs["cov_drug_dose"] == cat].copy()
            deg_list = adata.uns["rank_genes_groups_cov"][f"{cat}"]

            # Get raw counts for evaluation.
            x_true = cat_adata.layers["counts"].toarray()
            x_pred = cat_adata.obsm["CPA_pred"]
            x_ctrl = ctrl_adata.layers["counts"].toarray()

            # Log-transform the data.
            x_true = np.log1p(x_true)
            x_pred = np.log1p(x_pred)
            x_ctrl = np.log1p(x_ctrl)

            for n_top_deg in n_top_degs:
                if n_top_deg is not None:
                    degs = np.where(np.isin(adata.var_names, deg_list[:n_top_deg]))[0]
                else:
                    degs = np.arange(adata.n_vars)
                    n_top_deg = "all"

                x_true_deg = x_true[:, degs]
                x_pred_deg = x_pred[:, degs]
                x_ctrl_deg = x_ctrl[:, degs]

                r2_mean_deg = r2_score(x_true_deg.mean(0), x_pred_deg.mean(0))
                r2_var_deg = r2_score(x_true_deg.var(0), x_pred_deg.var(0))
                r2_mean_lfc_deg = r2_score(x_true_deg.mean(0) - x_ctrl_deg.mean(0),
                                           x_pred_deg.mean(0) - x_ctrl_deg.mean(0))
                r2_var_lfc_deg = r2_score(x_true_deg.var(0) - x_ctrl_deg.var(0),
                                          x_pred_deg.var(0) - x_ctrl_deg.var(0))

                # Assuming condition format "cov_cond_dose"
                cov, cond, dose = cat.split("_")
                results["cell_type"].append(cov)
                results["condition"].append(cond)
                results["dose"].append(dose)
                results["n_top_deg"].append(n_top_deg)
                results["r2_mean_deg"].append(r2_mean_deg)
                results["r2_var_deg"].append(r2_var_deg)
                results["r2_mean_lfc_deg"].append(r2_mean_lfc_deg)
                results["r2_var_lfc_deg"].append(r2_var_lfc_deg)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_path, 'evaluation_results.csv'), index=False)
    result_file = (os.path.join(results_dir, f'result_seed_{args.seed}_intense_'
                                             f'{str(args.intense_reg_rate).replace(".", "_")}_{args.intense_p}.csv')
                   if args.use_intense else
                   os.path.join(results_dir, f'result_seed_{args.seed}_original.csv'))
    df.to_csv(result_file, index=False)
    print(f"Experiment finished. Results are saved in {save_path}")


if __name__ == "__main__":
    main()

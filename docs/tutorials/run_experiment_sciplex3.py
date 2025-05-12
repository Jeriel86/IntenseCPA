#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Predicting drug perturbations using the sciplex3 dataset with CPA.
This script is parameterized for reproducibility and flexibility.
Usage example:
    python script.py --seed 434 --use_intense 1 --intense_reg_rate 0.1 --intense_p 1
"""
import hashlib
import os
import argparse
import random
import time

import numpy as np
import pandas as pd
import scanpy as sc
import cpa
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.metrics import r2_score
from tqdm import tqdm

from docs.tutorials.utils import append_to_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CPA on the sciplex3 dataset with configurable settings."
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--use_intense", type=int, choices=[0, 1], required=True,
                        help="0 for original CPA, 1 for INTENSE CPA")
    parser.add_argument("--intense_reg_rate", type=float, default=None,
                        help="Regularization rate for INTENSE CPA. Required if use_intense is 1")
    parser.add_argument("--intense_p", type=int, default=None,
                        help="p value for INTENSE CPA. Required if use_intense is 1")
    parser.add_argument("--current_dir", type=str, default="/scratch/nmbiedou",
                        help="Base directory for dataset and logs")
    return parser.parse_args()


def setup_paths(current_dir, use_intense, seed, intense_reg_rate, intense_p):
    data_dir = "/home/nmbiedou/Documents/cpa"
    data_path = os.path.join(data_dir, "datasets", "sciplex3_new.h5ad")
    if use_intense:
        save_path = os.path.join(
            current_dir, "experiment/Sciplex_Order_2",
            f"combo_Intense_reg_{str(intense_reg_rate).replace('.', '_')}_p_{intense_p}_seed_{seed}"
        )
    else:
        save_path = os.path.join(
            current_dir, "experiment/Sciplex_Order_2",
            f"combo_Original_seed_{seed}"
        )
    os.makedirs(save_path, exist_ok=True)
    results_dir = os.path.join(current_dir, 'experiment/Sciplex_Order_2', 'experiment_results')
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, 'experiment_log.csv')

    return data_path, save_path, log_file


def setup_hparams(args):
    # Base hyperparameters common to both settings
    if args.seed is None:
        # Get SLURM_ARRAY_TASK_ID, default to 0 if not running in SLURM array
        task_id = int(os.getenv('SLURM_ARRAY_TASK_ID', '0'))
        # Create a unique string combining time and task_id
        seed_input = f"{int(time.time() * 1000)}_{task_id}"
        # Hash the string and take the first 4 digits of the hex digest as the seed
        seed_hash = hashlib.md5(seed_input.encode()).hexdigest()
        args.seed = int(seed_hash[:4], 16) % 10000  # Convert hex to int, keep in 0-9999 range
    np.random.seed(args.seed)
    random.seed(args.seed)

    base_hparams = {
        "n_latent": 128,
        "recon_loss": "nb",
        "doser_type": "logsigm",
        "n_hidden_encoder": 512,
        "n_layers_encoder": 3,
        "n_hidden_decoder": 512,
        "n_layers_decoder": 3,
        "use_batch_norm_encoder": True,
        "use_layer_norm_encoder": False,
        "use_batch_norm_decoder": True,
        "use_layer_norm_decoder": False,
        "dropout_rate_encoder": 0.1,
        "dropout_rate_decoder": 0.1,
        "variational": False,
        "seed": args.seed,
    }
    # Trainer parameters default (can be adjusted per setting)
    trainer_params = {
        "n_epochs_kl_warmup": None,
        "n_epochs_pretrain_ae": 30,
        "n_epochs_adv_warmup": 50,
        "n_epochs_mixup_warmup": 3,
        "mixup_alpha": 0.1,
        "adv_steps": 2,
        "n_hidden_adv": 64,
        "n_layers_adv": 2,
        "use_batch_norm_adv": True,
        "use_layer_norm_adv": False,
        "dropout_rate_adv": 0.3,
        "reg_adv": 20.0,
        "pen_adv": 20.0,
        "lr": 0.0003,
        "wd": 4e-07,
        "adv_lr": 0.0003,
        "adv_wd": 4e-07,
        "adv_loss": "cce",
        "doser_lr": 0.0003,
        "doser_wd": 4e-07,
        "do_clip_grad": False,
        "gradient_clip_value": 1.0,
        "step_size_lr": 45,
    }
    # Overwrite with INTENSE settings if required
    if args.use_intense:
        base_hparams.update({
            "use_intense": True,
            "intense_reg_rate": args.intense_reg_rate,
            "interaction_order": 3,
            "intense_p": args.intense_p
        })
        trainer_params = {
            "n_epochs_kl_warmup": None,
            "n_epochs_pretrain_ae": 5,
            "n_epochs_adv_warmup": 3,
            "n_epochs_mixup_warmup": 0,
            "mixup_alpha": 0,
            "adv_steps": 3,
            "n_hidden_adv": 64,
            "n_layers_adv": 3,
            "use_batch_norm_adv": True,
            "use_layer_norm_adv": False,
            "dropout_rate_adv": 0,
            "reg_adv": 0.8032532418606549,
            "pen_adv": 16.07622882323635,
            "lr": 0.0012865737443496822,
            "wd": 0.00000910749307309212,
            "adv_lr": 0.00038342243722983504,
            "adv_wd": 0.00000489043147013771,
            "adv_loss": "cce",
            "doser_lr": 0.0007735652253832272,
            "doser_wd": 0.00000002529886527883,
            "do_clip_grad": False,
            "gradient_clip_value": 1,
            "step_size_lr": 25,
            "momentum": 0.3264281156751794
        }
    else:
        base_hparams["use_intense"] = False

    return base_hparams, trainer_params


def main():
    args = parse_args()

    # Validate parameters for intense CPA
    if args.use_intense and (args.intense_reg_rate is None or args.intense_p is None):
        raise ValueError("For --use_intense 1, --intense_reg_rate and --intense_p must be provided.")

    # Get hyperparameters and trainer settings
    ae_hparams, trainer_params = setup_hparams(args)

    # Set up directories and file paths
    data_path, save_path, log_file = setup_paths(args.current_dir, args.use_intense, args.seed,
                                       args.intense_reg_rate, args.intense_p)

    # Set figure parameters for scanpy
    sc.settings.set_figure_params(dpi=100)
    sc.settings.figdir = save_path
    # Load dataset
    try:
        adata = sc.read(data_path)
    except FileNotFoundError:
        raise FileNotFoundError("Dataset not found locally. Please download the dataset manually.")

    print(
        f"Running experiment with seed: {args.seed}, use_intense: {args.use_intense}, intense_reg_rate: {args.intense_reg_rate}, intense_p: {args.intense_p}")

    # Get SLURM task ID (if available)
    task_id = os.getenv('SLURM_ARRAY_TASK_ID', 'unknown')

    # Prepare the log entry
    log_entry = {
        'task_id': task_id,
        'use_intense': args.use_intense,
        'intense_reg_rate': args.intense_reg_rate if args.use_intense else None,
        'intense_p': args.intense_p if args.use_intense else None,
        'seed': args.seed,
        'save_path': save_path
    }
    append_to_csv(log_entry, log_file)

    adata.obs['condition'] = adata.obs['condition'] \
        .str.replace(r'^\(\+\)-', '', regex=True)
    
    # Prepare data for CPA using raw counts
    adata.X = adata.layers["counts"].copy()
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Setup anndata for CPA
    cpa.CPA.setup_anndata(
        adata,
        perturbation_key="condition",
        dosage_key="dose_val",
        control_group="control",
        batch_key=None,
        is_count_data=True,
        categorical_covariate_keys=["cell_type"],
        deg_uns_key="rank_genes_groups_cov",
        deg_uns_cat_key="cov_drug_dose_name",
        max_comb_len=1,
    )

    # Create CPA model
    model = cpa.CPA(
        adata=adata,
        split_key="split",
        train_split="train",
        valid_split="test",
        test_split="ood",
        **ae_hparams
    )

    # Train model
    model.train(
        max_epochs=2000,
        use_gpu=True,
        batch_size=128,
        plan_kwargs=trainer_params,
        early_stopping_patience=10,
        check_val_every_n_epoch=5,
        save_path=save_path,
    )

    # Plot training history
    history_plot_path = os.path.join(save_path, "history.png")
    cpa.pl.plot_history(model, history_plot_path)
    results_dir = os.path.join(args.current_dir, 'experiment/Sciplex_Order_2', 'experiment_results')
    os.makedirs(results_dir, exist_ok=True)

    if args.use_intense:
        scores = model.module.intense_fusion.mkl_fusion.scores()
        scores_file = os.path.join(results_dir,
                                   f'scores_reg_{str(args.intense_reg_rate).replace(".", "_")}_p_{args.intense_p}.csv')
        append_to_csv(scores, scores_file)

    # Latent space visualization
    latent_outputs = model.get_latent_representation(adata, batch_size=128)
    sc.settings.verbosity = 3

    # UMAP for latent_basal
    latent_basal_adata = latent_outputs['latent_basal']
    sc.pp.neighbors(latent_basal_adata)
    sc.tl.umap(latent_basal_adata)
    sc.pl.umap(latent_basal_adata, color=['condition'],
               frameon=False, wspace=0.2, save='latent_basal.png')

    # UMAP for latent_after
    latent_adata = latent_outputs['latent_after']
    sc.pp.neighbors(latent_adata)
    sc.tl.umap(latent_adata)
    sc.pl.umap(latent_adata, color=['condition'],
               frameon=False, wspace=0.2, save='latent_after.png')

    # --- Evaluation ---
    print("Starting prediction and evaluation...")
    model.predict(adata, batch_size=128)

    n_top_degs = [10, 20, 50, None]  # None means all genes
    results = defaultdict(list)
    ctrl_adata = adata[adata.obs['condition'] == 'control'].copy()

    for cat in tqdm(adata.obs['cov_drug_dose_name'].unique()):
        if 'control' not in cat:
            cat_adata = adata[adata.obs['cov_drug_dose_name'] == cat].copy()
            deg_cat = f'{cat}'
            deg_list = adata.uns['rank_genes_groups_cov'][deg_cat]

            x_true = cat_adata.layers['counts'].toarray()
            x_pred = cat_adata.obsm['CPA_pred']
            x_ctrl = ctrl_adata.layers['counts'].toarray()

            x_true = np.log1p(x_true)
            x_pred = np.log1p(x_pred)
            x_ctrl = np.log1p(x_ctrl)

            for n_top_deg in n_top_degs:
                if n_top_deg is not None:
                    degs = np.where(np.isin(adata.var_names, deg_list[:n_top_deg]))[0]
                else:
                    degs = np.arange(adata.n_vars)
                    n_top_deg = 'all'

                x_true_deg = x_true[:, degs]
                x_pred_deg = x_pred[:, degs]
                x_ctrl_deg = x_ctrl[:, degs]

                r2_mean_deg = r2_score(x_true_deg.mean(0), x_pred_deg.mean(0))
                r2_var_deg = r2_score(x_true_deg.var(0), x_pred_deg.var(0))
                r2_mean_lfc_deg = r2_score(x_true_deg.mean(0) - x_ctrl_deg.mean(0),
                                           x_pred_deg.mean(0) - x_ctrl_deg.mean(0))
                r2_var_lfc_deg = r2_score(x_true_deg.var(0) - x_ctrl_deg.var(0),
                                          x_pred_deg.var(0) - x_ctrl_deg.var(0))

                cov, cond, dose = cat.split('_')
                results['cell_type'].append(cov)
                results['condition'].append(cond)
                results['dose'].append(dose)
                results['n_top_deg'].append(n_top_deg)
                results['r2_mean_deg'].append(r2_mean_deg)
                results['r2_var_deg'].append(r2_var_deg)
                results['r2_mean_lfc_deg'].append(r2_mean_lfc_deg)
                results['r2_var_lfc_deg'].append(r2_var_lfc_deg)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_path, 'evaluation_results.csv'), index=False)
    result_file = (os.path.join(results_dir, f'result_seed_{args.seed}_intense_'
                                             f'{str(args.intense_reg_rate).replace(".", "_")}_{args.intense_p}.csv')
                   if args.use_intense else
                   os.path.join(results_dir, f'result_seed_{args.seed}_original.csv'))
    df.to_csv(result_file, index=False)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Predicting single-cell response to unseen combinatorial CRISPR perturbations
using the Norman 2019 dataset with CPA (Compositional Perturbation Autoencoder).

Steps:
1. Setting up environment and parsing arguments
2. Loading the dataset
3. Preprocessing the dataset
4. Creating and training a CPA model
5. Visualizing latent space
6. Evaluating prediction performance across perturbations
"""

import os
import sys
import gdown
import numpy as np
import time
import random
import pandas as pd
import scanpy as sc
import cpa
import argparse
from collections import defaultdict
from sklearn.metrics import r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import append_to_csv
import hashlib

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Run a single CPA experiment on Norman 2019 dataset")
parser.add_argument("--seed", type=int, default=None, help="Random seed for the experiment (optional)")
parser.add_argument("--use_intense", type=int, choices=[0, 1], required=True,
                    help="0 for original CPA, 1 for intense CPA")
parser.add_argument("--intense_reg_rate", type=float, default=None, help="Regularization rate for intense CPA")
parser.add_argument("--intense_p", type=int, default=None, help="p value for intense CPA")
args = parser.parse_args()

# Validate intense CPA parameters
if args.use_intense and (args.intense_reg_rate is None or args.intense_p is None):
    raise ValueError("For --use_intense 1, --intense_reg_rate and --intense_p must be provided.")

# Generate random seed if not provided
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

# --- Setting up Environment ---
current_dir = "/scratch/nmbiedou"
data_dir = "/home/nmbiedou/Documents/cpa"
sc.settings.set_figure_params(dpi=100)
data_path = os.path.join(data_dir, "datasets", "Norman2019_normalized_hvg.h5ad")

# Define save path based on parameters
if args.use_intense:
    save_path = os.path.join(
        current_dir, 'experiment/Norman_3',
        f'Norman_Intense_reg_{str(args.intense_reg_rate).replace(".", "_")}_p_{args.intense_p}_seed_{args.seed}'
    )
else:
    save_path = os.path.join(current_dir, 'experiment/Norman_3', f'Norman_Original_seed_{args.seed}')
os.makedirs(save_path, exist_ok=True)
sc.settings.figdir = save_path
# --- Loading Dataset ---
try:
    adata = sc.read(data_path)
except FileNotFoundError:
    print("Dataset not found locally. Downloading...")
    gdown.download(
        'https://drive.google.com/uc?export=download&id=109G9MmL-8-uh7OSjnENeZ5vFbo62kI7j',
        data_path,
        quiet=False
    )
    adata = sc.read(data_path)

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

# Define the log file location
results_dir = os.path.join(current_dir, 'experiment/Norman_3', 'experiment_results')
os.makedirs(results_dir, exist_ok=True)
log_file = os.path.join(results_dir, 'experiment_log.csv')
append_to_csv(log_entry, log_file)

print(f"Running experiment with seed: {args.seed}, use_intense: {args.use_intense}, "
      f"intense_reg_rate: {args.intense_reg_rate}, intense_p: {args.intense_p}")
print(adata)

# --- Preprocessing Dataset ---
adata.X = adata.layers['counts'].copy()

# Setup AnnData for CPA
cpa.CPA.setup_anndata(
    adata,
    perturbation_key='cond_harm',
    control_group='ctrl',
    dosage_key='dose_value',
    categorical_covariate_keys=['cell_type'],
    is_count_data=True,
    deg_uns_key='rank_genes_groups_cov',
    deg_uns_cat_key='cov_cond',
    max_comb_len=2
)

# --- CPA Model Parameters ---
if args.use_intense:
    model_params = {
        "n_latent": 32,
        "recon_loss": "nb",
        "doser_type": "linear",
        "n_hidden_encoder": 256,
        "n_layers_encoder": 4,
        "n_hidden_decoder": 256,
        "n_layers_decoder": 2,
        "use_batch_norm_encoder": True,
        "use_layer_norm_encoder": False,
        "use_batch_norm_decoder": False,
        "use_layer_norm_decoder": False,
        "dropout_rate_encoder": 0.2,
        "dropout_rate_decoder": 0.0,
        "variational": False,
        "seed": args.seed,
        "use_intense": True,
        "intense_reg_rate": args.intense_reg_rate,
        "intense_p": args.intense_p
    }
    trainer_params = {
        "n_epochs_kl_warmup": None,
        "n_epochs_adv_warmup": 0,
        "n_epochs_mixup_warmup": 1,
        "n_epochs_pretrain_ae": 1,
        "mixup_alpha": 0.1,
        "lr": 0.0001329496663990796,
        "wd": 0.00000005274980238001,
        "adv_steps": 30,
        "reg_adv": 21.307550534802644,
        "pen_adv": 0.026896850256354875,
        "adv_lr": 0.0016663826870585436,
        "adv_wd": 0.00000380102931347824,
        "doser_lr": 0.0002044287340750476,
        "doser_wd": 0.00000108683186233352,
        "n_layers_adv": 2,
        "n_hidden_adv": 32,
        "use_batch_norm_adv": True,
        "use_layer_norm_adv": False,
        "dropout_rate_adv": 0.25,
        "step_size_lr": 45,
        "do_clip_grad": False,
        "adv_loss": "cce",
        "gradient_clip_value": 5.0,
        "momentum": 0.21584635014258433
    }
else:
    model_params = {
        "n_latent": 32,
        "recon_loss": "nb",
        "doser_type": "linear",
        "n_hidden_encoder": 256,
        "n_layers_encoder": 4,
        "n_hidden_decoder": 256,
        "n_layers_decoder": 2,
        "use_batch_norm_encoder": True,
        "use_layer_norm_encoder": False,
        "use_batch_norm_decoder": False,
        "use_layer_norm_decoder": False,
        "dropout_rate_encoder": 0.2,
        "dropout_rate_decoder": 0.0,
        "variational": False,
        "seed": args.seed,
        "use_intense": False
    }
    trainer_params = {
        "n_epochs_kl_warmup": None,
        "n_epochs_adv_warmup": 50,
        "n_epochs_mixup_warmup": 10,
        "n_epochs_pretrain_ae": 10,
        "mixup_alpha": 0.1,
        "lr": 0.0001,
        "wd": 3.2170178270865573e-06,
        "adv_steps": 3,
        "reg_adv": 10.0,
        "pen_adv": 20.0,
        "adv_lr": 0.0001,
        "adv_wd": 7.051355554517135e-06,
        "n_layers_adv": 2,
        "n_hidden_adv": 128,
        "use_batch_norm_adv": True,
        "use_layer_norm_adv": False,
        "dropout_rate_adv": 0.3,
        "step_size_lr": 25,
        "do_clip_grad": False,
        "adv_loss": "cce",
        "gradient_clip_value": 5.0
    }

# Split dataset: Hold out specific combinations as OOD
adata.obs['split'] = np.random.choice(['train', 'valid'], size=adata.n_obs, p=[0.85, 0.15])
adata.obs.loc[adata.obs['cond_harm'].isin(['DUSP9+ETS2', 'CBL+CNN1']), 'split'] = 'ood'

# --- Creating and Training CPA Model ---
model = cpa.CPA(
    adata=adata,
    split_key='split',
    train_split='train',
    valid_split='valid',
    test_split='ood',
    **model_params
)

model.train(
    max_epochs=2000,
    use_gpu=True,
    batch_size=2048,
    plan_kwargs=trainer_params,
    early_stopping_patience=10 if args.use_intense else 5,
    check_val_every_n_epoch=5,
    save_path=save_path
)

# Plot training history
plot_path = os.path.join(save_path, f"history_seed_{args.seed}.png")
cpa.pl.plot_history(model, plot_path)
plt.close()

if args.use_intense:
    #plot_path_scores = os.path.join(save_path, "scores.png")
    #cpa.pl.plot_relevance_scores(model.module.intense_fusion.mkl_fusion, plot_path_scores)
    scores = model.module.intense_fusion.mkl_fusion.scores()
    scores_file = os.path.join(results_dir, f'scores_reg_{str(args.intense_reg_rate).replace(".", "_")}_p_{args.intense_p}.csv')
    append_to_csv(scores, scores_file)

# --- Latent Space Visualization ---
latent_outputs = model.get_latent_representation(adata, batch_size=2048)

# Basal latent space
sc.pp.neighbors(latent_outputs['latent_basal'])
sc.tl.umap(latent_outputs['latent_basal'])

groups = list(np.unique(adata[adata.obs['split'] == 'ood'].obs['cond_harm'].values))

sc.pl.umap(
    latent_outputs['latent_basal'],
    color='cond_harm',
    groups=groups,
    palette=sc.pl.palettes.godsnot_102,
    frameon=False,
    save=f'latent_basal_condharm_seed_{args.seed}.png'
)

sc.pl.umap(
    latent_outputs['latent_basal'],
    color='pathway',
    palette=sc.pl.palettes.godsnot_102,
    frameon=False,
    save=f'latent_basal_pathway_seed_{args.seed}.png'
)

# Final latent space
sc.pp.neighbors(latent_outputs['latent_after'])
sc.tl.umap(latent_outputs['latent_after'])
sc.pl.umap(
    latent_outputs['latent_after'],
    color='cond_harm',
    groups=groups,
    palette=sc.pl.palettes.godsnot_102,
    frameon=False,
    save=f'latent_after_cond_harm_seed_{args.seed}.png'
)

sc.pl.umap(
    latent_outputs['latent_after'],
    color='pathway',
    palette=sc.pl.palettes.godsnot_102,
    frameon=False,
    save=f'latent_after_pathway_seed_{args.seed}.png'
)


# --- Evaluation ---
# Store true expression and sample control cells
adata.layers['X_true'] = adata.X.copy()
ctrl_adata = adata[adata.obs['cond_harm'] == 'ctrl'].copy()
adata.X = ctrl_adata.X[np.random.choice(ctrl_adata.n_obs, size=adata.n_obs, replace=True), :]

# Predict perturbation effects
model.predict(adata, batch_size=2048)
adata.layers['CPA_pred'] = adata.obsm['CPA_pred'].copy()

# Log-transform for evaluation
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.normalize_total(adata, target_sum=1e4, layer='CPA_pred')
sc.pp.log1p(adata, layer='CPA_pred')

# Compute R2 scores
n_top_degs = [10, 20, 50, None]
results = defaultdict(list)
ctrl_adata = adata[adata.obs['cond_harm'] == 'ctrl'].copy()

for condition in tqdm(adata.obs['cond_harm'].unique()):
    if condition != 'ctrl':
        cond_adata = adata[adata.obs['cond_harm'] == condition].copy()

        deg_cat = f'K562_{condition}'
        deg_list = adata.uns['rank_genes_groups_cov'][deg_cat]

        x_true = cond_adata.layers['counts'].toarray()
        x_pred = cond_adata.obsm['CPA_pred']
        x_ctrl = ctrl_adata.layers['counts'].toarray()

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

            r2_mean_lfc_deg = r2_score(x_true_deg.mean(0) - x_ctrl_deg.mean(0), x_pred_deg.mean(0) - x_ctrl_deg.mean(0))

            results['condition'].append(condition)
            results['n_top_deg'].append(n_top_deg)
            results['r2_mean_deg'].append(r2_mean_deg)
            results['r2_mean_lfc_deg'].append(r2_mean_lfc_deg)
# Save results
df = pd.DataFrame(results)
df.to_csv(os.path.join(save_path, f'evaluation_results_seed_{args.seed}.csv'), index=False)

# Save to shared location for aggregation
results_dir = os.path.join(current_dir, 'experiment/Norman_3', 'experiment_results')
os.makedirs(results_dir, exist_ok=True)
result_file = (os.path.join(results_dir, f'result_seed_{args.seed}_intense_'
                                         f'{str(args.intense_reg_rate).replace(".", "_")}_{args.intense_p}.csv')
               if args.use_intense else
               os.path.join(results_dir, f'result_seed_{args.seed}_original.csv'))
df.to_csv(result_file, index=False)

# Print OOD conditions
#print("Evaluation results for OOD conditions:")
#print(df[df['condition'].isin(['DUSP9+ETS2', 'CBL+CNN1'])])
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Predicting combinatorial drug perturbations using the combo-sciplex dataset with CPA.

Steps:
1. Setting up environment
2. Loading the dataset
3. Preprocessing the dataset
4. Creating a CPA model
5. Training the model
6. Latent space visualisation
7. Prediction evaluation across different perturbations
8. Visualizing similarity between drug embeddings
"""

import os
import sys
import gdown
import numpy as np
import pandas as pd
import scanpy as sc
import cpa
from collections import defaultdict
import matplotlib.pyplot as plt


# Set figure parameters
sc.settings.set_figure_params(dpi=100)

# Define data path
current_dir = os.getcwd()
data_path = os.path.join(current_dir, "datasets", "combo_sciplex_prep_hvg_filtered.h5ad")
save_path = os.path.join(current_dir, 'lightning_logs', 'combo')
# Load dataset
try:
    adata = sc.read(data_path)
except FileNotFoundError:
    print("Dataset not found locally. Downloading...")
    gdown.download('https://drive.google.com/uc?export=download&id=1RRV0_qYKGTvD3oCklKfoZQFYqKJy4l6t', data_path, quiet=False)
    adata = sc.read(data_path)

# Prepare data for CPA by replacing adata.X with raw counts
adata.X = adata.layers['counts'].copy()

# Setup anndata for CPA
cpa.CPA.setup_anndata(adata,
                      perturbation_key='condition_ID',
                      dosage_key='log_dose',
                      control_group='CHEMBL504',
                      batch_key=None,
                      is_count_data=True,
                      categorical_covariate_keys=['cell_type'],
                      deg_uns_key='rank_genes_groups_cov',
                      deg_uns_cat_key='cov_drug_dose',
                      max_comb_len=2)

# Define model parameters
ae_hparams = {
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
    "seed": 434,
    "use_intense": False,
}

# Define trainer parameters
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

# Create CPA model
model = cpa.CPA(adata=adata,
                split_key='split_1ct_MEC',
                train_split='train',
                valid_split='valid',
                test_split='ood',
                **ae_hparams)

# Train model
model.train(max_epochs=2000,
            use_gpu=False,
            batch_size=128,
            plan_kwargs=trainer_params,
            early_stopping_patience=10,
            check_val_every_n_epoch=5,
            save_path=save_path)

# Plot training history and save to file
cpa.pl.plot_history(model)
plt.savefig(os.path.join(current_dir, 'training_history_combo.png'))
plt.close()

# Compute latent representations
latent_outputs = model.get_latent_representation(adata, batch_size=1024)

# Visualize latent space (basal and after)
sc.pp.neighbors(latent_outputs['latent_basal'])
sc.tl.umap(latent_outputs['latent_basal'])
sc.pl.umap(latent_outputs['latent_basal'], color='condition_ID', save='latent_basal_combo.png')

sc.pp.neighbors(latent_outputs['latent_after'])
sc.tl.umap(latent_outputs['latent_after'])
sc.pl.umap(latent_outputs['latent_after'], color='condition_ID', save='latent_after_combo.png')

# Evaluate prediction performance
model.predict(adata, batch_size=1024)

# Compute R2 scores for evaluation
results = defaultdict(list)
ctrl_adata = adata[adata.obs['condition_ID'] == 'CHEMBL504'].copy()
for cat in adata.obs['cov_drug_dose'].unique():
    if 'CHEMBL504' not in cat:
        cat_adata = adata[adata.obs['cov_drug_dose'] == cat].copy()
        deg_cat = f'{cat}'
        deg_list = adata.uns['rank_genes_groups_cov'][deg_cat]
        x_true = cat_adata.layers['counts'].toarray()
        x_pred = cat_adata.obsm['CPA_pred']
        x_ctrl = ctrl_adata.layers['counts'].toarray()
        x_true = np.log1p(x_true)
        x_pred = np.log1p(x_pred)
        x_ctrl = np.log1p(x_ctrl)
        for n_top_deg in [10, 20, 50, None]:
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

# Convert results to DataFrame and print for n_top_deg = 20
df = pd.DataFrame(results)
print(df[df['n_top_deg'] == 20])

# Visualize similarity between drug embeddings
cpa_api = cpa.ComPertAPI(adata, model,
                         de_genes_uns_key='rank_genes_groups_cov',
                         pert_category_key='cov_drug_dose',
                         control_group='CHEMBL504')
cpa_plots = cpa.pl.CompertVisuals(cpa_api, fileprefix=None)
drug_adata = cpa_api.get_pert_embeddings()
cpa_plots.plot_latent_embeddings(drug_adata.X, kind='perturbations', titlename='Drugs')
plt.savefig(os.path.join(current_dir, 'drug_embeddings.png'))
plt.close()
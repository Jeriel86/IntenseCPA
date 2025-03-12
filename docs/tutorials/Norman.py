#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Predicting single-cell response to unseen combinatorial CRISPR perturbations
using the Norman 2019 dataset with CPA (Compositional Perturbation Autoencoder).

Steps:
1. Setting up environment
2. Loading the dataset
3. Preprocessing the dataset
4. Creating a CPA model
5. Training the model
6. Latent space visualisation
7. Prediction evaluation across different perturbations
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
# Print and change current directory
print("Current directory:", os.getcwd())
os.chdir("../..")  # Adjust this based on your starting directory
print("New directory:", os.getcwd())
current_dir = os.getcwd()
data_path = os.path.join(current_dir, "datasets", "Norman2019_normalized_hvg.h5ad")

# Define save path for the results(model, images, csv)
save_path = os.path.join(current_dir, 'lightning_logs', 'Norman')
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# Load dataset
try:
    adata = sc.read(data_path)
except FileNotFoundError:
    print("Dataset not found locally. Downloading...")
    gdown.download('https://drive.google.com/uc?export=download&id=109G9MmL-8-uh7OSjnENeZ5vFbo62kI7j', data_path, quiet=False)
    adata = sc.read(data_path)

# Prepare data for CPA by replacing adata.X with raw counts
adata.X = adata.layers['counts'].copy()

# Setup anndata for CPA
cpa.CPA.setup_anndata(adata,
                      perturbation_key='cond_harm',
                      control_group='ctrl',
                      dosage_key='dose_value',
                      categorical_covariate_keys=['cell_type'],
                      is_count_data=True,
                      deg_uns_key='rank_genes_groups_cov',
                      deg_uns_cat_key='cov_cond',
                      max_comb_len=2)

# Define model parameters
model_params = {
    "n_latent": 128,                    # Increased latent space size
    "recon_loss": "nb",                 # Negative binomial reconstruction loss
    "doser_type": "linear",             # Linear doser type
    "n_hidden_encoder": 256,            # Encoder hidden units
    "n_layers_encoder": 1,              # Reduced encoder layers
    "n_hidden_decoder": 512,            # Increased decoder hidden units
    "n_layers_decoder": 5,              # Increased decoder layers
    "use_batch_norm_encoder": True,     # Batch norm in encoder
    "use_layer_norm_encoder": False,    # No layer norm in encoder
    "use_batch_norm_decoder": False,    # No batch norm in decoder
    "use_layer_norm_decoder": False,    # No layer norm in decoder
    "dropout_rate_encoder": 0.25,       # Slightly increased encoder dropout
    "dropout_rate_decoder": 0.2,        # Added decoder dropout
    "variational": False,               # Non-variational model
    "seed": 9010,                       # Updated seed
    "use_intense": True,                # Enable intense regularization
    "intense_reg_rate": 0.05,           # Intensity regularization rate
}
# Define trainer parameters (updated from "train_args")
trainer_params = {
    "n_epochs_kl_warmup": None,         # No KL warmup
    "n_epochs_adv_warmup": 1,           # Reduced adversarial warmup
    "n_epochs_mixup_warmup": 1,         # Reduced mixup warmup
    "n_epochs_pretrain_ae": 3,          # Reduced pretraining epochs
    "mixup_alpha": 0.5,                 # Increased mixup strength
    "lr": 0.00019662186085984122,      # Learning rate
    "wd": 1.6225757449999367e-08,      # Weight decay
    "adv_steps": 3,                     # Adversarial steps
    "reg_adv": 0.3892674000347504,     # Adversarial regularization
    "pen_adv": 0.36097151881121287,    # Adversarial penalty
    "adv_lr": 0.002455537115440242,    # Adversarial learning rate
    "adv_wd": 9.607119555361394e-08,   # Adversarial weight decay
    "n_layers_adv": 4,                  # Increased adversarial layers
    "n_hidden_adv": 128,                # Adversarial hidden units
    "use_batch_norm_adv": True,         # Batch norm in adversary
    "use_layer_norm_adv": False,        # No layer norm in adversary
    "dropout_rate_adv": 0.1,            # Reduced adversarial dropout
    "step_size_lr": 45,                 # Learning rate scheduler step size
    "do_clip_grad": False,              # No gradient clipping
    "adv_loss": "cce",                  # Categorical cross-entropy adversarial loss
    "gradient_clip_value": 5.0,         # Gradient clip value (unused since do_clip_grad=False)
}

# Split dataset: Leave DUSP9+ETS2 and CBL+CNN1 out of training dataset
adata.obs['split'] = np.random.choice(['train', 'valid'], size=adata.n_obs, p=[0.85, 0.15])
adata.obs.loc[adata.obs['cond_harm'].isin(['DUSP9+ETS2', 'CBL+CNN1']), 'split'] = 'ood'

# Create CPA model
model = cpa.CPA(adata=adata,
                split_key='split',
                train_split='train',
                valid_split='valid',
                test_split='ood',
                **model_params)

# Train model
model.train(max_epochs=2000,
            use_gpu=True,
            batch_size=2048,
            plan_kwargs=trainer_params,
            early_stopping_patience=5,
            check_val_every_n_epoch=5,
            save_path=os.path.join(current_dir, 'lightning_logs', 'Norman2019_Optimized'))

# Plot training history and save to file
plot_path = os.path.join(save_path, "figures", 'norman_training_history.png')
cpa.pl.plot_history(model,plot_path)
plt.close()

# Compute latent representations
latent_outputs = model.get_latent_representation(adata, batch_size=2048)

# Visualize latent space (basal and after)
sc.pp.neighbors(latent_outputs['latent_basal'])
sc.tl.umap(latent_outputs['latent_basal'])
sc.pl.umap(latent_outputs['latent_basal'], color='cond_harm', save='norman_latent_basal.png')

sc.pp.neighbors(latent_outputs['latent_after'])
sc.tl.umap(latent_outputs['latent_after'])
sc.pl.umap(latent_outputs['latent_after'], color='cond_harm', save='norman_latent_after.png')

# Evaluate prediction performance
# Store true expression and sample control cells
adata.layers['X_true'] = adata.X.copy()
ctrl_adata = adata[adata.obs['cond_harm'] == 'ctrl'].copy()
adata.X = ctrl_adata.X[np.random.choice(ctrl_adata.n_obs, size=adata.n_obs, replace=True), :]

# Predict perturbation effects
model.predict(adata, batch_size=2048)
adata.layers['CPA_pred'] = adata.obsm['CPA_pred'].copy()

# Normalize and log-transform for evaluation
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.normalize_total(adata, target_sum=1e4, layer='CPA_pred')
sc.pp.log1p(adata, layer='CPA_pred')

# Compute R2 scores for evaluation
from sklearn.metrics import r2_score
results = defaultdict(list)
for condition in adata.obs['cond_harm'].unique():
    if condition != 'ctrl':
        cond_adata = adata[adata.obs['cond_harm'] == condition].copy()
        deg_cat = f'K562_{condition}'
        deg_list = adata.uns['rank_genes_groups_cov'][deg_cat]
        x_true = cond_adata.layers['counts'].toarray()
        x_pred = cond_adata.obsm['CPA_pred']
        x_ctrl = ctrl_adata.layers['counts'].toarray()
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
            r2_mean_lfc_deg = r2_score(x_true_deg.mean(0) - x_ctrl_deg.mean(0),
                                     x_pred_deg.mean(0) - x_ctrl_deg.mean(0))
            results['condition'].append(condition)
            results['n_top_deg'].append(n_top_deg)
            results['r2_mean_deg'].append(r2_mean_deg)
            results['r2_mean_lfc_deg'].append(r2_mean_lfc_deg)

# Convert results to DataFrame and print OOD conditions
df = pd.DataFrame(results)
df.to_csv(os.path.join(save_path,"figures" ,'norman_evaluation_results.csv'), index=False)
print(df[df['condition'].isin(['DUSP9+ETS2', 'CBL+CNN1'])])
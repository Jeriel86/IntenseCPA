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
6. Latent space visualization
7. Prediction evaluation across different perturbations
8. Visualizing similarity between drug embeddings
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import cpa
import matplotlib.pyplot as plt

# Set figure parameters
sc.settings.set_figure_params(dpi=100)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
# Define data path
print("Current directory:", os.getcwd())
os.chdir("../..")  # Adjust this based on your starting directory
print("New directory:", os.getcwd())
current_dir = os.getcwd()

data_path = os.path.join(current_dir, "datasets", "combo_sciplex_prep_hvg_filtered.h5ad")
save_path = os.path.join(current_dir, "lightning_logs", "combo_original")

# Load dataset
try:
    adata = sc.read(data_path)
except FileNotFoundError:
    print("Dataset not found locally. Downloading...")
    # Add download logic here if needed
    raise

# Prepare data for CPA by replacing adata.X with raw counts
adata.X = adata.layers["counts"].copy()

# Setup anndata for CPA
cpa.CPA.setup_anndata(
    adata,
    perturbation_key="condition_ID",
    dosage_key="log_dose",
    control_group="CHEMBL504",
    batch_key=None,
    is_count_data=True,
    categorical_covariate_keys=["cell_type"],
    deg_uns_key="rank_genes_groups_cov",
    deg_uns_cat_key="cov_drug_dose",
    max_comb_len=2,
)

### Normal CPA setting
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
    "use_intense": False
}

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

### Optimized setting for INTENSE
# Define model hyperparameters
"""ae_hparams = {
    "n_latent": 32,
    "recon_loss": "nb",
    "doser_type": "logsigm",
    "n_hidden_encoder": 128,
    "n_layers_encoder": 1,
    "n_hidden_decoder": 256,
    "n_layers_decoder": 5,
    "use_batch_norm_encoder": True,
    "use_layer_norm_encoder": False,
    "use_batch_norm_decoder": True,
    "use_layer_norm_decoder": False,
    "dropout_rate_encoder": 0.1,
    "dropout_rate_decoder": 0.25,
    "variational": False,
    "seed": 1302,
    "use_intense": True,
    "intense_reg_rate": 0.05,
}

# Define trainer hyperparameters
trainer_params = {
    "n_epochs_kl_warmup": None,
    "n_epochs_pretrain_ae": 5,
    "n_epochs_adv_warmup": 0,
    "n_epochs_mixup_warmup": 10,
    "mixup_alpha": 0.2,
    "adv_steps": 25,
    "n_hidden_adv": 256,
    "n_layers_adv": 5,
    "use_batch_norm_adv": False,
    "use_layer_norm_adv": False,
    "dropout_rate_adv": 0.25,
    "reg_adv": 0.7476092303124932,
    "pen_adv": 12.046854697093057,
    "lr": 0.005720126880802159,
    "wd": 5.918645343488014e-08,
    "adv_lr": 1.1045215398085119e-05,
    "adv_wd": 2.1193608953288304e-07,
    "adv_loss": "cce",
    "doser_lr": 1.96717142007588e-05,
    "doser_wd": 1.880987671857387e-06,
    "do_clip_grad": False,
    "gradient_clip_value": 1.0,
    "step_size_lr": 25,
}
"""
# Create CPA model
model = cpa.CPA(
    adata=adata,
    split_key="split_1ct_MEC",
    train_split="train",
    valid_split="valid",
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


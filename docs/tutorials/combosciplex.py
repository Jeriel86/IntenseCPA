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

from sklearn.metrics import r2_score
from collections import defaultdict
from tqdm import tqdm



# Print and change current directory
#print("Current directory:", os.getcwd())
#os.chdir("../..")  # Adjust this based on your starting directory
#print("New directory:", os.getcwd())
current_dir = "/home/nmbiedou/Documents/cpa"

data_path = os.path.join(current_dir, "datasets", "combo_sciplex_prep_hvg_filtered.h5ad")
save_path = os.path.join(current_dir, "lightning_logs", "combo_main_2")

# Set figure parameters
sc.settings.set_figure_params(dpi=100)
# Load dataset
try:
    adata = sc.read(data_path)
except FileNotFoundError:
    print("Dataset not found locally. Downloading...")
    # Add download logic here if needed
    raise

# Prepare data for CPA by replacing adata.X with raw counts
adata.X = adata.layers["counts"].copy()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
### Optimized setting for INTENSE
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
    "use_intense": True,
    "interaction_order": 2,
    "intense_reg_rate":0.05073135389055399,
    "intense_p": 2
}

trainer_params = {
    "n_epochs_kl_warmup": None,
    "n_epochs_pretrain_ae": 3,
    "n_epochs_adv_warmup": 3,
    "n_epochs_mixup_warmup": 10,
    "mixup_alpha": 0.1,
    "adv_steps": 2,
    "n_hidden_adv": 256,
    "n_layers_adv": 2,
    "use_batch_norm_adv": False,
    "use_layer_norm_adv": True,
    "dropout_rate_adv": 0,
    "reg_adv": 1.419091687459432,
    "pen_adv": 12.775412073171998,
    "lr": 0.003273373979034034,
    "wd": 4e-07,
    "adv_lr": 0.00015304936848310163,
    "adv_wd": 0.00000011309928874122,
    "adv_loss": "cce",
    "doser_lr": 0.0007629540879596654,
    "doser_wd": 0.00000043589345787571,
    "do_clip_grad": False,
    "gradient_clip_value": 1.0,
    "step_size_lr": 25,
    "momentum": 0.5126039493891473
}

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
    early_stopping_patience=5,
    check_val_every_n_epoch=5,
    save_path=save_path,
)
plot_path = os.path.join(save_path, "history.png")
# Plot training history
cpa.pl.plot_history(model,plot_path)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
#### Load pretrained model ####
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#model = cpa.CPA.load(dir_path=save_path,
 #                    adata=adata, use_gpu=True)
###### Latent space UMAP visualization
latent_outputs = model.get_latent_representation(adata, batch_size=128)
sc.settings.verbosity = 3

latent_basal_adata = latent_outputs['latent_basal']
latent_adata = latent_outputs['latent_after']
sc.pp.neighbors(latent_basal_adata)
sc.tl.umap(latent_basal_adata)

sc.pl.umap(latent_basal_adata,
           color=['condition_ID'],
           frameon=False, wspace=0.2,
           save='latent_basal.png')

os.rename(
    os.path.join(sc.settings.figdir, f'umaplatent_basal.png'),
    os.path.join(save_path, f'latent_basal.png')
)


sc.pp.neighbors(latent_adata)
sc.tl.umap(latent_adata)

sc.pl.umap(latent_adata,
           color=['condition_ID'],
           frameon=False,
           wspace=0.2,
           save='latent_after.png')

os.rename(
    os.path.join(sc.settings.figdir, f'umaplatent_after.png'),
    os.path.join(save_path, f'latent_after.png')
)


##### Evaluation ####
print("prediction...")
model.predict(adata, batch_size=128)
print("evaluation...")


n_top_degs = [10, 20, 50, None]  # None means all genes

results = defaultdict(list)
ctrl_adata = adata[adata.obs['condition_ID'] == 'CHEMBL504'].copy()
for cat in tqdm(adata.obs['cov_drug_dose'].unique()):
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

            r2_mean_lfc_deg = r2_score(x_true_deg.mean(0) - x_ctrl_deg.mean(0), x_pred_deg.mean(0) - x_ctrl_deg.mean(0))
            r2_var_lfc_deg = r2_score(x_true_deg.var(0) - x_ctrl_deg.var(0), x_pred_deg.var(0) - x_ctrl_deg.var(0))

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
print(df)

# Optional: Save results to CSV
df.to_csv(os.path.join(save_path, 'evaluation_results.csv'), index=False)


### Visualization per condition ###

for cat in adata.obs["cov_drug_dose"].unique():
    if "CHEMBL504" not in cat:
        cat_adata = adata[adata.obs["cov_drug_dose"] == cat].copy()

        cat_adata.X = np.log1p(cat_adata.layers["counts"].A)
        cat_adata.obsm["CPA_pred"] = np.log1p(cat_adata.obsm["CPA_pred"])

        deg_list = adata.uns["rank_genes_groups_cov"][f'{cat}'][:20]

        print(cat, f"{cat_adata.shape}")
        cpa.pl.mean_plot(
            cat_adata,
            pred_obsm_key="CPA_pred",
            path_to_save=os.path.join(save_path, f"mean_plot_{cat}.png"),
            deg_list=deg_list,
            # gene_list=deg_list[:5],
            show=True,
            verbose=True,
        )

if __name__ == "__main__":
    pass  # Ensures script runs only if executed directly










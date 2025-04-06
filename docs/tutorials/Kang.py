# Predicting perturbation responses for unseen cell-types (context transfer)
#
# This script trains and evaluates a CPA model on the preprocessed Kang PBMC dataset.
# Steps:
# 1. Setting up environment
# 2. Loading the dataset
# 3. Preprocessing the dataset
# 4. Creating a CPA model
# 5. Training the model
# 6. Latent space visualization
# 7. Prediction evaluation across different perturbations

import sys
import os
import gdown
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from collections import defaultdict
from tqdm import tqdm

# Check if running in Google Colab
IN_COLAB = "google.colab" in sys.modules
branch = "latest"

# Install dependencies if in Colab
if IN_COLAB and branch == "stable":
    os.system("pip install cpa-tools")
    os.system("pip install scanpy")
elif IN_COLAB and branch != "stable":
    os.system("pip install --quiet --upgrade jsonschema")
    os.system("pip install git+https://github.com/theislab/cpa")
    os.system("pip install scanpy")

# Import CPA and Scanpy
import cpa
import scanpy as sc

# --- Setting up environment ---

# Print and change current directory
#print("Current directory:", os.getcwd())
#os.chdir("../..")  # Adjust this based on your starting directory
#print("New directory:", os.getcwd())
current_dir = "/home/nmbiedou/Documents/cpa"

# Uncomment to set GPU visibility
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Set Scanpy figure parameters
sc.settings.set_figure_params(dpi=100)

# Define data path dynamically
data_path = os.path.join(current_dir, "datasets", "kang_normalized_hvg.h5ad")

# Define save path for the results(model, images, csv)
save_path = os.path.join(current_dir, 'lightning_logs', 'Kang_Intense_Rite_2_config2')
sc.settings.figdir = save_path
# --- Loading dataset ---

# Load the preprocessed Kang PBMC dataset
try:
    adata = sc.read(data_path)
except:
    # Download from Google Drive if not found locally
    gdown.download('https://drive.google.com/uc?export=download&id=1z8gGKQ6Doi2blCU2IVihKA38h5fORRp')
    adata = sc.read(data_path)

print(adata)

# Replace adata.X with raw counts for CPA training
adata.X = adata.layers['counts'].copy()

# --- Dataset setup ---

# Create a dummy dosage variable for each condition
adata.obs['dose'] = adata.obs['condition'].apply(lambda x: '+'.join(['1.0' for _ in x.split('+')]))

# Print value counts for inspection
print(adata.obs['cell_type'].value_counts())
print(adata.obs['condition'].value_counts())

# Set up AnnData for CPA
print("Setup Anndata")
cpa.CPA.setup_anndata(
    adata,
    perturbation_key='condition',
    control_group='ctrl',
    dosage_key='dose',
    categorical_covariate_keys=['cell_type'],
    is_count_data=True,
    deg_uns_key='rank_genes_groups_cov',
    deg_uns_cat_key='cov_cond',
    max_comb_len=1,
)
print( "model params")
# --- CPA Model Parameters ---
model_params = {
    "n_latent": 64,
    "recon_loss": "nb",
    "doser_type": "linear",
    "n_hidden_encoder": 128,
    "n_layers_encoder": 2,
    "n_hidden_decoder": 512,
    "n_layers_decoder": 2,
    "use_batch_norm_encoder": True,
    "use_layer_norm_encoder": False,
    "use_batch_norm_decoder": False,
    "use_layer_norm_decoder": True,
    "dropout_rate_encoder": 0.0,
    "dropout_rate_decoder": 0.1,
    "variational": False,
    "seed": 6977,
    "use_intense": True,
    "use_rite": True,
    "intense_reg_rate": 0.1,
    "intense_p": 1,
    "interaction_order": 2,
    "rite_factor": 1,
}

trainer_params = {
    "n_epochs_adv_warmup": 5,
    "n_epochs_kl_warmup": None,
    "n_epochs_pretrain_ae": 10,
    "adv_steps": 5,
    "mixup_alpha": 0.1,
    "n_epochs_mixup_warmup": 3,
    "n_layers_adv": 4,
    "n_hidden_adv": 64,
    "use_batch_norm_adv": True,
    "use_layer_norm_adv": False,
    "dropout_rate_adv": 0.3,
    "pen_adv": 1.0754212899512865,
    "reg_adv": 1.7536582421567608,
    "lr": 0.00610331507808799,
    "wd": 0.00000001500803976284,
    "doser_lr": 0.004392578706283446,
    "doser_wd": 0.00000263511691407522,
    "adv_lr": 0.0000532609227647633,
    "adv_wd": 0.00000010496500499614,
    "adv_loss": "cce",
    "do_clip_grad": True,
    "gradient_clip_value": 1.0,
    "step_size_lr": 25,
    "momentum": 0.8381474745171963
}

"""trainer_params = {
    "n_epochs_kl_warmup": None,
    "n_epochs_pretrain_ae": 0,
    "n_epochs_adv_warmup": 5,
    "n_epochs_mixup_warmup": 10,
    "mixup_alpha": 0.2,
    "adv_steps": 2,
    "n_hidden_adv": 256,
    "n_layers_adv": 3,
    "use_batch_norm_adv": False,
    "use_layer_norm_adv": False,
    "dropout_rate_adv": 0.25,
    "reg_adv": 2.43828696766268,
    "pen_adv": 4.1724722219803425,
    "lr": 0.0001533493418490112,
    "wd": 0,
    "adv_lr": 0.0005984734868477526,
    "adv_wd": 0.00000001201376573356,
    "adv_loss": "cce",
    "doser_lr": 0.0003331089782353292,
    "doser_wd": 0.00000082009742906479,
    "do_clip_grad": False,
    "gradient_clip_value": 1.0,
    "step_size_lr": 25,
    "momentum": 0.0,
}"""
# --- Creating CPA Model ---
print("model build....")
# Exclude B cells treated with IFN-beta from training (OOD set)
model = cpa.CPA(
    adata=adata,
    split_key='split_B',
    train_split='train',
    valid_split='valid',
    test_split='ood',
    **model_params,
)

# --- Training CPA ---
print("Start training")
model.train(
    max_epochs=2000,
    use_gpu=True,  # Set to True if GPU is available
    batch_size=1024,
    plan_kwargs=trainer_params,
    early_stopping_patience=10,
    check_val_every_n_epoch=5,
    save_path=save_path
)

plot_path = os.path.join(save_path, "history.png")
# Plot training history
cpa.pl.plot_history(model,plot_path)

# --- Restore Best Model (Optional) ---
"""model = cpa.CPA.load(
     dir_path=os.path.join(current_dir, 'lightning_logs', 'Kang'),
     adata=adata,
     use_gpu=False
 )"""

# --- Latent Space Visualization ---

# Get latent representations
latent_outputs = model.get_latent_representation(adata, batch_size=2048)
#print(latent_outputs.keys())

# Basal latent space
sc.pp.neighbors(latent_outputs['latent_basal'])
sc.tl.umap(latent_outputs['latent_basal'])
sc.pl.umap(
    latent_outputs['latent_basal'],
    color=['condition', 'cell_type'],
    frameon=False,
    wspace=0.3,
    save='latent_basal.png'  # Saves the plot as a file
)

# Final latent space (after condition and cell_type embeddings)
sc.pp.neighbors(latent_outputs['latent_after'])
sc.tl.umap(latent_outputs['latent_after'])
sc.pl.umap(
    latent_outputs['latent_after'],
    color=['condition', 'cell_type'],
    frameon=False,
    wspace=0.3,
    save='latent_after.png'  # Saves the plot as a file
)

# --- Evaluation ---

# Predict perturbation responses
model.predict(adata, batch_size=2048)

# Evaluate prediction performance
n_top_degs = [10, 20, 50, None]  # None means all genes
results = defaultdict(list)

for cat in tqdm(adata.obs['cov_cond'].unique()):
    if 'ctrl' not in cat:
        cov, condition = cat.split('_')
        cat_adata = adata[adata.obs['cov_cond'] == cat].copy()
        ctrl_adata = adata[adata.obs['cov_cond'] == f'{cov}_ctrl'].copy()

        deg_cat = f'{cat}'
        deg_list = adata.uns['rank_genes_groups_cov'][deg_cat]

        x_true = cat_adata.layers['counts']
        x_pred = cat_adata.obsm['CPA_pred']
        x_ctrl = ctrl_adata.layers['counts']

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

            results['condition'].append(condition)
            results['cell_type'].append(cov)
            results['n_top_deg'].append(n_top_deg)
            results['r2_mean_deg'].append(r2_mean_deg)
            results['r2_var_deg'].append(r2_var_deg)
            results['r2_mean_lfc_deg'].append(r2_mean_lfc_deg)
            results['r2_var_lfc_deg'].append(r2_var_lfc_deg)

# Convert results to DataFrame and display
df = pd.DataFrame(results)
print(df)

# Optional: Save results to CSV
df.to_csv(os.path.join(save_path, 'evaluation_results.csv'), index=False)

if __name__ == "__main__":
    pass  # Ensures script runs only if executed directly
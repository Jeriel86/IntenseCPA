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
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'

# Set Scanpy figure parameters
sc.settings.set_figure_params(dpi=100)

# Define data path dynamically
data_path = os.path.join(current_dir, "datasets", "kang_normalized_hvg.h5ad")

# Define save path for the results(model, images, csv)
save_path = os.path.join(current_dir, 'lightning_logs', 'Kang_Intense_SGD_Optimized_2203')

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

# --- CPA Model Parameters ---
model_params = {
    "n_latent": 64,                    # Updated from model_args
    "recon_loss": "nb",                # Updated from model_args
    "doser_type": "linear",            # Updated from model_args
    "n_hidden_encoder": 128,           # Updated from model_args
    "n_layers_encoder": 2,             # Updated from model_args
    "n_hidden_decoder": 512,           # Updated from model_args
    "n_layers_decoder": 2,             # Updated from model_args
    "use_batch_norm_encoder": True,    # Updated from model_args
    "use_layer_norm_encoder": False,   # Updated from model_args
    "use_batch_norm_decoder": False,   # Updated from model_args
    "use_layer_norm_decoder": True,    # Updated from model_args
    "dropout_rate_encoder": 0.0,       # Updated from model_args
    "dropout_rate_decoder": 0.1,       # Updated from model_args
    "variational": False,              # Updated from model_args
    "seed": 6977,                      # Updated from model_args (6,977 interpreted as 6977)
    "use_intense": True,               # Updated from model_args
    "intense_reg_rate": 0.5,           # Updated from model_args
    "intense_p": 2                     # Updated from model_args
}

trainer_params = {
    "n_epochs_adv_warmup": 1,          # Updated from model_args
    "n_epochs_kl_warmup": None,        # Updated from model_args (null)
    "n_epochs_pretrain_ae": 3,         # Updated from model_args
    "adv_steps": 20,                   # Updated from model_args
    "mixup_alpha": 0.5,                # Updated from model_args
    "n_epochs_mixup_warmup": 1,        # Updated from model_args
    "n_layers_adv": 2,                 # Updated from model_args
    "n_hidden_adv": 128,               # Updated from model_args
    "use_batch_norm_adv": True,        # Updated from model_args
    "use_layer_norm_adv": False,       # Updated from model_args
    "dropout_rate_adv": 0.3,           # Updated from model_args
    "pen_adv": 0.06586477769085837,    # Updated from model_args
    "reg_adv": 25.915240512217768,     # Updated from model_args
    "lr": 0.00031846009054514735,      # Updated from model_args
    "wd": 0.00000001297631322054,      # Updated from model_args
    "doser_lr": 0.0011680586429996507, # Updated from model_args
    "doser_wd": 0.00000250280215373454,# Updated from model_args
    "adv_lr": 0.00001758762700595009,  # Updated from model_args
    "adv_wd": 0.00000007470316045061,  # Updated from model_args
    "adv_loss": "cce",                 # Updated from model_args
    "do_clip_grad": False,             # Updated from model_args
    "gradient_clip_value": 1,          # Updated from model_args
    "step_size_lr": 45,                # Updated from model_args
}
# --- Creating CPA Model ---

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

model.train(
    max_epochs=2000,
    use_gpu=True,  # Set to True if GPU is available
    batch_size=512,
    plan_kwargs=trainer_params,
    early_stopping_patience=10,
    check_val_every_n_epoch=5,
    save_path=save_path,
    num_gpus=8
)

plot_path = os.path.join(save_path, "history.png")
# Plot training history
cpa.pl.plot_history(model,plot_path)

# --- Restore Best Model (Optional) ---
# model = cpa.CPA.load(
#     dir_path=os.path.join(current_dir, 'lightning_logs', 'Kang'),
#     adata=adata,
#     use_gpu=False
# )

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
    save='latent_basal_2203_2.png'  # Saves the plot as a file
)

os.rename(
    os.path.join(sc.settings.figdir, f'umaplatent_basal_2203_2.png'),
    os.path.join(save_path, f'latent_basal_2203_2.png')
)


# Final latent space (after condition and cell_type embeddings)
sc.pp.neighbors(latent_outputs['latent_after'])
sc.tl.umap(latent_outputs['latent_after'])
sc.pl.umap(
    latent_outputs['latent_after'],
    color=['condition', 'cell_type'],
    frameon=False,
    wspace=0.3,
    save='latent_after_2203_22.png'  # Saves the plot as a file
)
os.rename(
    os.path.join(sc.settings.figdir, f'umaplatent_after_2203_2.png'),
    os.path.join(save_path, f'latent_after_2.png')
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
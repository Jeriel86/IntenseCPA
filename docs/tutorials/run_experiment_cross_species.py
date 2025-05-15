import hashlib
import sys
import os
import gdown
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from collections import defaultdict
from tqdm import tqdm
import argparse
import random
import time
import cpa
import scanpy as sc

from docs.tutorials.utils import append_to_csv, measure_training_pl, measure_inference_pl

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Run a single CPA experiment")
parser.add_argument("--seed", type=int, default=None, help="Random seed for the experiment (optional)")
parser.add_argument("--use_intense", type=int, choices=[0, 1], required=True, help="0 for original CPA, 1 for intense CPA")
parser.add_argument("--intense_reg_rate", type=float, default=None, help="Regularization rate for intense CPA")
parser.add_argument("--intense_p", type=int, default=None, help="p value for intense CPA")
args = parser.parse_args()

# --- Setting up environment ---
current_dir = "/scratch/nmbiedou"
data_dir = "/home/nmbiedou/Documents/cpa"
sc.settings.set_figure_params(dpi=100)
data_path = os.path.join(data_dir, "datasets", "cross_species_new.h5ad")

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


# Save path based on parameters
if args.use_intense:
    save_path = os.path.join(current_dir, 'experiment/Cross_Order_2', f'Cross_Intense_reg_{str(args.intense_reg_rate).replace(".", "_")}_p_{args.intense_p}_seed_{args.seed}')
else:
    save_path = os.path.join(current_dir, 'experiment/Cross_Order_2', f'Cross_Original_seed_{args.seed}')
os.makedirs(save_path, exist_ok=True)
sc.settings.figdir = save_path
# --- Loading dataset ---
try:
    adata = sc.read(data_path)
except:
    gdown.download('https://drive.google.com/uc?export=download&id=1z8gGKQ6Doi2blCU2IVihKA38h5fORRp')
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
results_dir = os.path.join(current_dir, 'experiment/Cross_Order_2', 'experiment_results')
os.makedirs(results_dir, exist_ok=True)
log_file = os.path.join(results_dir, 'experiment_log.csv')
append_to_csv(log_entry, log_file)

print(f"Running experiment with seed: {args.seed}, use_intense: {args.use_intense}, intense_reg_rate: {args.intense_reg_rate}, intense_p: {args.intense_p}")
print(adata)

adata.X = adata.layers['counts'].copy()

cpa.CPA.setup_anndata(
    adata,
    perturbation_key='condition',
    control_group='control',
    dosage_key='dose_val',
    categorical_covariate_keys=['species'],
    is_count_data=True,
    deg_uns_key='rank_genes_groups_cov',
    deg_uns_cat_key='cov_drug_dose_name',
    max_comb_len=1,
)

# --- CPA Model Parameters ---
if args.use_intense:
    model_params = {
        "n_latent": 256,
        "recon_loss": "nb",
        "doser_type": "logsigm",
        "n_hidden_encoder": 256,
        "n_layers_encoder": 3,
        "n_hidden_decoder": 128,
        "n_layers_decoder": 2,
        "use_batch_norm_encoder": False,
        "use_layer_norm_encoder": False,
        "use_batch_norm_decoder": False,
        "use_layer_norm_decoder": True,
        "dropout_rate_encoder": 0.25,
        "dropout_rate_decoder": 0.0,
        "variational": False,
        "seed": args.seed,
        "use_intense": True,
        "interaction_order": 2,
        "intense_reg_rate": args.intense_reg_rate,
        "intense_p": args.intense_p
    }
    trainer_params = {
        "n_epochs_adv_warmup": 10,
        "n_epochs_kl_warmup": None,
        "n_epochs_pretrain_ae": 5,
        "adv_steps": 3,
        "mixup_alpha": 0,
        "n_epochs_mixup_warmup": 0,
        "n_layers_adv": 2,
        "n_hidden_adv": 64,
        "use_batch_norm_adv": True,
        "use_layer_norm_adv": False,
        "dropout_rate_adv": 0.1,
        "pen_adv": 1.2275890306102464,
        "reg_adv": 7.635346541535559,
        "lr": 0.0003916534682803787,
        "wd": 5.985138372921983e-7,
        "doser_lr": 0.0004576703731531967,
        "doser_wd": 0.000007034097484816466,
        "adv_lr": 0.00033486219722040055,
        "adv_wd": 1.8820406393812192e-8,
        "adv_loss": "cce",
        "do_clip_grad": False,
        "gradient_clip_value": 1,
        "step_size_lr": 15,
        "momentum": 0.4633125728242274
    }
else:
    model_params = {
        "n_latent": 256,
        "recon_loss": "nb",
        "doser_type": "logsigm",
        "n_hidden_encoder": 256,
        "n_layers_encoder": 3,
        "n_hidden_decoder": 128,
        "n_layers_decoder": 2,
        "use_batch_norm_encoder": False,
        "use_layer_norm_encoder": False,
        "use_batch_norm_decoder": False,
        "use_layer_norm_decoder": True,
        "dropout_rate_encoder": 0.25,
        "dropout_rate_decoder": 0.0,
        "variational": False,
        "seed": args.seed,
        "use_intense": False
    }

    trainer_params = {
        "n_epochs_kl_warmup": None,
        "n_epochs_pretrain_ae": 3,
        "n_epochs_adv_warmup": 1,
        "n_epochs_mixup_warmup": 0,
        "mixup_alpha": 0.0,
        "adv_steps": 5,
        "n_hidden_adv": 128,
        "n_layers_adv": 1,
        "use_batch_norm_adv": False,
        "use_layer_norm_adv": False,
        "dropout_rate_adv": 0.25,
        "reg_adv": 7.724182665704546,
        "pen_adv": 3.942845558437145,
        "lr": 0.00014711192869112068,
        "wd": 1.9732737637298378e-7,
        "adv_lr": 0.00025915661856804447,
        "adv_wd": 7.711353579347824e-7,
        "adv_loss": "cce",
        "doser_lr": 0.0035605106986818422,
        "doser_wd": 1.4167222326960833e-7,
        "do_clip_grad": True,
        "gradient_clip_value": 1.0,
        "step_size_lr": 25
    }

# --- Creating CPA Model ---
model = cpa.CPA(
    adata=adata,
    split_key='split',
    train_split='train',
    valid_split='test',
    test_split='ood',
    **model_params,
)

# -----------------------
# Profiling: Training
# -----------------------
print("Profiling training...")

train_time, train_mem = measure_training_pl(
    model.train,
    max_epochs=2000,
    use_gpu=True,
    batch_size=512,
    plan_kwargs=trainer_params,
    early_stopping_patience=10 if args.use_intense else 5,
    check_val_every_n_epoch=5,
    save_path=save_path,
)
print(f"Training completed in {train_time:.2f}s, peak GPU memory: {train_mem / 1e9:.2f} GB")

# -----------------------
# Profiling: Inference
# -----------------------
print("Profiling inference...")
inf_time, inf_tp, inf_mem, latent_outputs = measure_inference_pl(
    model,
    adata,
    batch_size=2048
)
print(f"Inference completed in {inf_time:.2f}s, throughput: {inf_tp:.2f} samples/s, peak GPU memory: {inf_mem / 1e9:.2f} GB")

# -----------------------
# Save timing & memory metrics
# -----------------------
metrics_file = os.path.join(os.path.dirname(log_file), 'time_memory_metrics.csv')
metrics_entry = {
    'task_id': task_id,
    'use_intense': args.use_intense,
    'intense_reg_rate': args.intense_reg_rate if args.use_intense else None,
    'intense_p': args.intense_p if args.use_intense else None,
    'seed': args.seed,
    'train_time_s': train_time,
    'inf_time_s': inf_time,
    'inf_tp_s': inf_tp,
    'train_mem_GB': train_mem / 1e9,
    'inf_mem_GB': inf_mem / 1e9,
}
append_to_csv(metrics_entry, metrics_file)
print(f"Metrics saved to {metrics_file}")

# Plot training history
plot_path = os.path.join(save_path, "history.png")
cpa.pl.plot_history(model, plot_path)

if args.use_intense:
    scores = model.module.intense_fusion.mkl_fusion.scores()
    scores_file = os.path.join(results_dir, f'scores_reg_{str(args.intense_reg_rate).replace(".", "_")}_p_{args.intense_p}.csv')
    append_to_csv(scores,scores_file)
# --- Latent Space Visualization ---

# Basal latent space
sc.pp.neighbors(latent_outputs['latent_basal'])
sc.tl.umap(latent_outputs['latent_basal'])
sc.pl.umap(
    latent_outputs['latent_basal'],
    color=['condition', 'species'],
    frameon=False,
    wspace=0.3,
    save=f'latent_basal_seed_{args.seed}.png',
    show=False
)

# Final latent space
sc.pp.neighbors(latent_outputs['latent_after'])
sc.tl.umap(latent_outputs['latent_after'])
sc.pl.umap(
    latent_outputs['latent_after'],
    color=['condition', 'species'],
    frameon=False,
    wspace=0.3,
    save=f'latent_after_seed_{args.seed}.png',
    show=False
)

# --- Evaluation ---
model.predict(adata, batch_size=2048)
n_top_degs = [10, 20, 50, None]  # None means all genes

results = defaultdict(list)
ctrl_adata = adata[(adata.obs['condition'] == 'unst') & (adata.obs['dose_val'] == 1.0)].copy()
for cat in tqdm(adata.obs['cov_drug_dose_name'].unique()):
    if 'unst_1.0' not in cat:
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
            r2_mean_lfc_deg = r2_score(x_true_deg.mean(0) - x_ctrl_deg.mean(0), x_pred_deg.mean(0) - x_ctrl_deg.mean(0))
            r2_var_lfc_deg = r2_score(x_true_deg.var(0) - x_ctrl_deg.var(0), x_pred_deg.var(0) - x_ctrl_deg.var(0))

            cov, cond, dose = cat.split('_')

            results['species'].append(cov)
            results['condition'].append(cond)
            results['dose'].append(dose)
            results['n_top_deg'].append(n_top_deg)
            results['r2_mean_deg'].append(r2_mean_deg)
            results['r2_var_deg'].append(r2_var_deg)
            results['r2_mean_lfc_deg'].append(r2_mean_lfc_deg)
            results['r2_var_lfc_deg'].append(r2_var_lfc_deg)

df = pd.DataFrame(results)
df.to_csv(os.path.join(save_path, f'evaluation_results_seed_{args.seed}.csv'), index=False)

# Save results to a shared location for aggregation
results_dir = os.path.join(current_dir, 'experiment/Cross_Order_2', 'experiment_results')
os.makedirs(results_dir, exist_ok=True)
if args.use_intense:
    result_file = os.path.join(results_dir, f'result_seed_{args.seed}_intense_{str(args.intense_reg_rate).replace(".", "_")}_{args.intense_p}.csv')
else:
    result_file = os.path.join(results_dir, f'result_seed_{args.seed}_original.csv')
df.to_csv(result_file, index=False)
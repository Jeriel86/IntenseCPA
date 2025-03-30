from cpa import run_autotune
import cpa
import scanpy as sc
from ray import tune
import numpy as np
import pickle
import os
import gdown

# Define paths based on combo-sciplex structure
PROJECT_ROOT = "/home/nmbiedou/Documents/cpa"
ORIGINAL_DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "combo_sciplex_prep_hvg_filtered.h5ad")
PREPROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "combo_sciplex_prep_hvg_filtered_preprocessed.h5ad")
LOGGING_DIR = os.getenv("LOGGING_DIR", "/scratch/nmbiedou/autotune")


os.makedirs(LOGGING_DIR, exist_ok=True)

# Check if preprocessed data exists; if not, create and save it
if not os.path.exists(PREPROCESSED_DATA_PATH):
    # Load original data
    adata = sc.read_h5ad(ORIGINAL_DATA_PATH)
    adata.X = adata.layers['counts'].copy()  # Set raw counts

    # Add control column as in combo-sciplex example
    adata.obs['control'] = (adata.obs['condition_ID'] == 'CHEMBL504').astype(int)

    # Save the preprocessed data
    adata.write_h5ad(PREPROCESSED_DATA_PATH)
    print(f"Preprocessed data saved to {PREPROCESSED_DATA_PATH}")
else:
    print(f"Loading preprocessed data from {PREPROCESSED_DATA_PATH}")

# Load the preprocessed data
adata = sc.read_h5ad(PREPROCESSED_DATA_PATH)

# Subsample the data
#sc.pp.subsample(adata, fraction=0.1)

# Model hyperparameters for tuning (adapted from combo-sciplex example)
model_args = {
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
    'split_key': 'split_1ct_MEC',
    'train_split': 'train',
    'valid_split': 'valid',
    'test_split': 'ood',
    'use_intense': True,
    'intense_reg_rate': tune.choice([0.001, 0.005, 0.01, 0.05, 0.1]),
    'intense_p': tune.choice([1, 2])
}

# Training hyperparameters for tuning (adapted from combo-sciplex example)
train_args = {
    'n_epochs_adv_warmup': tune.choice([0, 1, 3, 5, 10, 50, 70]),
    'n_epochs_kl_warmup': tune.choice([None]),
    'n_epochs_pretrain_ae': tune.choice([0, 1, 3, 5, 10, 30, 50]),
    'adv_steps': tune.choice([2, 3, 5, 10, 15, 20, 25, 30]),
    'mixup_alpha': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    'n_epochs_mixup_warmup': tune.sample_from(
        lambda spec: 0 if spec.config.train_args.mixup_alpha == 0.0 else np.random.choice([0, 1, 3, 5, 10])),
    'n_layers_adv': tune.choice([1, 2, 3, 4, 5]),
    'n_hidden_adv': tune.choice([32, 64, 128, 256]),
    'use_batch_norm_adv': tune.choice([True, False]),
    'use_layer_norm_adv': tune.sample_from(
        lambda spec: False if spec.config.train_args.use_batch_norm_adv else np.random.choice([True, False])),
    'dropout_rate_adv': tune.choice([0.0, 0.1, 0.2, 0.25, 0.3]),
    'pen_adv': tune.loguniform(1e-2, 1e2),
    'reg_adv': tune.loguniform(1e-2, 1e2),
    'lr': tune.loguniform(1e-5, 1e-2),
    'wd': tune.loguniform(1e-8, 1e-5),
    'doser_lr': tune.loguniform(1e-5, 1e-2),
    'doser_wd': tune.loguniform(1e-8, 1e-5),
    'adv_lr': tune.loguniform(1e-5, 1e-2),
    'adv_wd': tune.loguniform(1e-8, 1e-5),
    'adv_loss': tune.choice(['cce']),
    'do_clip_grad': tune.choice([True, False]),
    'gradient_clip_value': tune.choice([1.0]),
    'step_size_lr': tune.choice([10, 25, 45]),
}
plan_kwargs_keys = list(train_args.keys())

# Trainer arguments
trainer_actual_args = {
    'max_epochs': 300,
    'use_gpu': True,
    'early_stopping_patience': 10,
    'check_val_every_n_epoch': 5
}
train_args.update(trainer_actual_args)

# Search space for hyperparameter tuning
search_space = {
    'model_args': model_args,
    'train_args': train_args,
}

# Scheduler settings for ASHA
scheduler_kwargs = {
    'max_t': 1000,
    'grace_period': 5,
    'reduction_factor': 3,
}

# AnnData setup arguments (adapted from combo-sciplex example)
setup_anndata_kwargs = {
    'perturbation_key': 'condition_ID',
    'dosage_key': 'log_dose',
    'control_group': 'CHEMBL504',
    'batch_key': None,
    'is_count_data': True,
    'categorical_covariate_keys': ['cell_type'],
    'deg_uns_key': 'rank_genes_groups_cov',
    'deg_uns_cat_key': 'cov_drug_dose',
    'max_comb_len': 2,
}

# Setup AnnData with preprocessed data
model = cpa.CPA
model.setup_anndata(adata, **setup_anndata_kwargs)

# Resources for training
resources = {
    "cpu": 8,
    "gpu":1,
    "memory": 130 * 1024 * 1024 * 1024  # 183 GiB
}

# Run hyperparameter tuning
EXPERIMENT_NAME = "cpa_autotune_combo"
CHECKPOINT_DIR  = os.path.join(LOGGING_DIR, EXPERIMENT_NAME)

experiment = run_autotune(
        model_cls=model,
        data=adata,
        metrics=["cpa_metric", "r2_mean_deg", "r2_var_deg", "r2_mean_lfc_deg", "r2_var_lfc_deg"],
        mode="max",
        search_space=search_space,
        num_samples=200,
        scheduler="asha",
        searcher="hyperopt",
        seed=1,
        resources=resources,
        experiment_name=EXPERIMENT_NAME,
        logging_dir=LOGGING_DIR,
        adata_path=PREPROCESSED_DATA_PATH,
        sub_sample=None,
        setup_anndata_kwargs=setup_anndata_kwargs,
        use_wandb=True,
        wandb_name="cpa_tune_combo",
        scheduler_kwargs=scheduler_kwargs,
        plan_kwargs_keys=plan_kwargs_keys,
    )

# Save results
result_grid = experiment.result_grid
print(result_grid)
with open(os.path.join(PROJECT_ROOT, 'result_grid_combo.pkl'), 'wb') as f:
    pickle.dump(result_grid, f)
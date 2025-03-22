from cpa import run_autotune
import cpa
import scanpy as sc
from ray import tune
import numpy as np
import pickle
import os

# Define paths based on Kang notebook structure
PROJECT_ROOT = "/home/nmbiedou/Documents/cpa"
ORIGINAL_DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "kang_normalized_hvg.h5ad")
PREPROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "kang_normalized_hvg_preprocessed.h5ad")
LOGGING_DIR = os.getenv("LOGGING_DIR", "/scratch/nmbiedou/autotune")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Check if preprocessed data exists; if not, create and save it
if not os.path.exists(PREPROCESSED_DATA_PATH):
    # Load original data
    adata = sc.read_h5ad(ORIGINAL_DATA_PATH)
    adata.X = adata.layers['counts'].copy()  # Set raw counts

    # Add dose column from Kang notebook preprocessing
    adata.obs['dose'] = adata.obs['condition'].apply(lambda x: '+'.join(['1.0' for _ in x.split('+')]))

    # Save the preprocessed data
    adata.write_h5ad(PREPROCESSED_DATA_PATH)
    print(f"Preprocessed data saved to {PREPROCESSED_DATA_PATH}")
else:
    print(f"Loading preprocessed data from {PREPROCESSED_DATA_PATH}")

# Load the preprocessed data
adata = sc.read_h5ad(PREPROCESSED_DATA_PATH)

# Subsample the data
sc.pp.subsample(adata, fraction=0.1)

# Model hyperparameters for tuning
model_args = {
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
    'split_key': 'split_B',
    'train_split': 'train',
    'valid_split': 'valid',
    'test_split': 'ood',
    'use_intense': True,
    'intense_reg_rate': tune.choice([0.0, 0.001, 0.005, 0.01, 0.05, 0.1]),
    'intense_p': tune.choice([1, 2])
}

# Training hyperparameters for tuning
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
    'max_epochs': 200,
    'use_gpu': True,
    'early_stopping_patience':   10,
    'check_val_every_n_epoch': 5,
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
    'reduction_factor': 4,
}

# AnnData setup arguments (from Kang notebook)
setup_anndata_kwargs = {
    'perturbation_key': 'condition',
    'dosage_key': 'dose',
    'control_group': 'ctrl',
    'batch_key': None,
    'is_count_data': True,
    'categorical_covariate_keys': ['cell_type'],
    'deg_uns_key': 'rank_genes_groups_cov',
    'deg_uns_cat_key': 'cov_cond',
    'max_comb_len': 1,
}

# Setup AnnData with preprocessed data
model = cpa.CPA
model.setup_anndata(adata, **setup_anndata_kwargs)

# Resources matching XEON_SP_4215 node with Tesla V100
resources = {
    "cpu": 40,
    "gpu": 8,
    "memory": 100 * 1024 * 1024 * 1024  # 183 GiB
}

# Run hyperparameter tuning
experiment = run_autotune(
    model_cls=model,
    data=adata,
    metrics=["cpa_metric", "disnt_basal", "disnt_after", "r2_mean", "val_r2_mean", "val_r2_var", "val_recon"],
    mode="max",
    search_space=search_space,
    num_samples=100,
    scheduler="asha",
    searcher="hyperopt",
    seed=1,
    resources=resources,
    experiment_name="kang_autotune_2103_1",
    logging_dir=LOGGING_DIR,
    adata_path=PREPROCESSED_DATA_PATH,  # Use preprocessed data path
    sub_sample=None,
    setup_anndata_kwargs=setup_anndata_kwargs,
    use_wandb=True,
    wandb_name="cpa_kang_tune_2103_1",
    scheduler_kwargs=scheduler_kwargs,
    plan_kwargs_keys=plan_kwargs_keys,
)

# Save results
result_grid = experiment.result_grid
print(result_grid)
with open(os.path.join(PROJECT_ROOT, 'result_grid.pkl'), 'wb') as f:
    pickle.dump(result_grid, f)
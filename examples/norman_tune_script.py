from cpa import run_autotune
import cpa
import scanpy as sc
from ray import tune
import numpy as np
import pickle
import os


# Save the original CUDA_VISIBLE_DEVICES (if set by the cluster)
_original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)

# For setup_anndata: restrict GPU visibility to only one GPU.
if _original_cuda_visible_devices:
    # If CUDA_VISIBLE_DEVICES is already set (possibly multiple GPUs),
    # select only the first one.
    single_device = _original_cuda_visible_devices.split(",")[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = single_device
else:
    # If the variable is not set, try to detect available GPUs and restrict to GPU 0.
    try:
        import torch
        if torch.cuda.device_count() > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    except ImportError:
        pass  # If torch is not available, do nothing.

# Define the path to the Norman2019 dataset
PROJECT_ROOT = "/home/nmbiedou/Documents/cpa"
ORIGINAL_DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "Norman2019_normalized_hvg.h5ad")
PREPROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "Norman2019_normalized_hvg_preprocessed.h5ad")
LOGGING_DIR = os.getenv("LOGGING_DIR", "/scratch/nmbiedou/autotune")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

if not os.path.exists(PREPROCESSED_DATA_PATH):
    # Load original data
    adata = sc.read_h5ad(ORIGINAL_DATA_PATH)
    adata.X = adata.layers['counts'].copy()  # Set raw counts

    # Create data splits: 85% train, 15% valid, and specific conditions as out-of-distribution (ood)
    adata.obs['split'] = np.random.choice(['train', 'valid'], size=adata.n_obs, p=[0.85, 0.15])
    adata.obs.loc[adata.obs['cond_harm'].isin(['DUSP9+ETS2', 'CBL+CNN1']), 'split'] = 'ood'

    # Save the preprocessed data
    adata.write_h5ad(PREPROCESSED_DATA_PATH)
    print(f"Preprocessed data saved to {PREPROCESSED_DATA_PATH}")
else:
    print(f"Loading preprocessed data from {PREPROCESSED_DATA_PATH}")

adata = sc.read_h5ad(PREPROCESSED_DATA_PATH)

# Define model arguments with a search space for hyperparameter tuning
model_args = {
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
    "seed": 8206,
    "use_intense": True,
    "use_rite": False,
    "interaction_order": 3,
    "intense_reg_rate": tune.loguniform(1e-3, 1e-1),
    "intense_p": tune.choice([1, 2]),

    'split_key': 'split',  # Use the custom 'split' column created above
    'train_split': 'train',
    'valid_split': 'valid',
    'test_split': 'ood',

}

# Define training arguments with a search space for hyperparameter tuning
train_args = {
    'n_epochs_adv_warmup': tune.choice([0, 1, 3, 5, 10, 50, 70]),
    'n_epochs_kl_warmup': tune.choice([None]),  # No KL warmup since variational is False
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

    'adv_loss': tune.choice(['cce']),  # Categorical cross-entropy as in the reference script

    'do_clip_grad': tune.choice([True, False]),
    'gradient_clip_value': tune.choice([1.0, 5.0]),  # Include 5.0 from the reference script

    'step_size_lr': tune.choice([10, 25, 45]),
    'momentum': tune.uniform(0.0, 0.99),
}

# Store keys for plan_kwargs
plan_kwargs_keys = list(train_args.keys())

# Additional trainer arguments
trainer_actual_args = {
    'max_epochs': 500,
    'use_gpu': True,
    'early_stopping_patience':  10,
    'batch_size': 2048,
    'check_val_every_n_epoch': 5
}
train_args.update(trainer_actual_args)

# Combine into the search space for Ray Tune
search_space = {
    'model_args': model_args,
    'train_args': train_args,
}

# Define scheduler parameters for ASHA (Asynchronous Successive Halving Algorithm)
scheduler_kwargs = {
    'max_t': 1000,
    'grace_period': 5,
    'reduction_factor': 3,
}

# Setup AnnData for CPA with keys matching the Norman2019 dataset
setup_anndata_kwargs = {
    'perturbation_key': 'cond_harm',       # Perturbation identifier
    'dosage_key': 'dose_value',            # Dosage information
    'control_group': 'ctrl',               # Control condition
    'batch_key': None,                     # No batch effect correction
    'is_count_data': True,                 # Data is count-based
    'categorical_covariate_keys': ['cell_type'],  # Covariate for cell type
    'deg_uns_key': 'rank_genes_groups_cov',       # Differential expression key
    'deg_uns_cat_key': 'cov_cond',         # Category key for DEG
    'max_comb_len': 2,                     # Maximum combination length
}

# Initialize and setup the CPA model with AnnData
model = cpa.CPA
model.setup_anndata(adata, **setup_anndata_kwargs)

# Restore original CUDA_VISIBLE_DEVICES for run_autotune so that all available GPUs are visible.
if _original_cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _original_cuda_visible_devices
else:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

# Run the hyperparameter tuning experiment
EXPERIMENT_NAME = "cpa_autotune_norman_0804"
CHECKPOINT_DIR  = os.path.join(LOGGING_DIR, EXPERIMENT_NAME)
experiment = run_autotune(
        model_cls=model,
        data=adata,
        metrics=["cpa_metric", "disnt_basal", "disnt_after", "r2_mean", "val_r2_mean", "val_r2_var", "val_recon"],
        mode="max",  # Maximize the first metric (cpa_metric)
        search_space=search_space,
        num_samples=200,  # Number of trials to run (adjust as needed)
        scheduler="asha",
        searcher="hyperopt",
        seed=1,
        resources={"cpu": 16,"gpu": 2,"memory": 120 * 1024 * 1024 * 1024},  # Adjust based on hardware
        experiment_name=EXPERIMENT_NAME,
        logging_dir=LOGGING_DIR,  # Update to desired logging directory
        adata_path=PREPROCESSED_DATA_PATH,
        sub_sample=None,
        setup_anndata_kwargs=setup_anndata_kwargs,
        use_wandb=True,
        wandb_name="cpa_tune_norman_0804",
        scheduler_kwargs=scheduler_kwargs,
        plan_kwargs_keys=plan_kwargs_keys,
    )

# Save the tuning results
result_grid = experiment.result_grid
result_file_path = os.path.join(LOGGING_DIR, 'result_grid_norman_newest.pkl')
with open(result_file_path, 'wb') as f:
    pickle.dump(result_grid, f)
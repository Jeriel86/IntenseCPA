from cpa import run_autotune
import cpa
import scanpy as sc
from ray import tune
import numpy as np
import pickle
import os
import gdown


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

# Define paths based on cross_species structure
PROJECT_ROOT = "/home/nmbiedou/Documents/cpa"
ORIGINAL_DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "cross_species_new.h5ad")
PREPROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "cross_species_new_preprocessed.h5ad")
LOGGING_DIR = os.getenv("LOGGING_DIR", "/scratch/nmbiedou/autotune")


os.makedirs(LOGGING_DIR, exist_ok=True)

# Check if preprocessed data exists; if not, create and save it
if not os.path.exists(PREPROCESSED_DATA_PATH):
    # Load original data
    adata = sc.read_h5ad(ORIGINAL_DATA_PATH)
    adata.X = adata.layers['counts'].copy()  # Set raw counts

    # Save the preprocessed data
    adata.write_h5ad(PREPROCESSED_DATA_PATH)
    print(f"Preprocessed data saved to {PREPROCESSED_DATA_PATH}")
else:
    print(f"Loading preprocessed data from {PREPROCESSED_DATA_PATH}")

# Load the preprocessed data
adata = sc.read_h5ad(PREPROCESSED_DATA_PATH)

# Subsample the data
#sc.pp.subsample(adata, fraction=0.1)

# Model hyperparameters for tuning
model_args = {
    'n_latent': tune.choice([32, 64, 128, 256]),
    'recon_loss': tune.choice(['nb']),
    'doser_type': tune.choice(['logsigm']),
    'n_hidden_encoder': tune.choice([128, 256, 512]),
    'n_layers_encoder': tune.choice([1, 2, 3, 4, 5]),
    'n_hidden_decoder': tune.choice([128, 256, 512]),
    'n_layers_decoder': tune.choice([1, 2, 3, 4, 5]),
    'use_batch_norm_encoder': tune.choice([True, False]),
    'use_layer_norm_encoder': tune.sample_from(
        lambda spec: False if spec.config.model_args.use_batch_norm_encoder else np.random.choice([True, False])),
    'use_batch_norm_decoder': tune.choice([True, False]),
    'use_layer_norm_decoder': tune.sample_from(
        lambda spec: False if spec.config.model_args.use_batch_norm_decoder else np.random.choice([True, False])),
    'dropout_rate_encoder': tune.choice([0.0, 0.1, 0.2, 0.25]),
    'dropout_rate_decoder': tune.choice([0.0, 0.1, 0.2, 0.25]),
    'variational': tune.choice([False]),
    'seed': tune.randint(0, 10000),
    'split_key': 'split',
    'train_split': 'train',
    'valid_split': 'test',
    'test_split': 'ood',
    'use_intense': False
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
    'step_size_lr': tune.choice([15, 25, 45]),
    #'momentum': tune.uniform(0.0, 0.99),
}
plan_kwargs_keys = list(train_args.keys())

# Trainer arguments
trainer_actual_args = {
    'max_epochs': 2000,
    'use_gpu': True,
    'early_stopping_patience': tune.choice([5,10,15]),
    'check_val_every_n_epoch': 5,
    'batch_size': tune.choice([128,256,512])
}
train_args.update(trainer_actual_args)

# Search space for hyperparameter tuning
search_space = {
    'model_args': model_args,
    'train_args': train_args,
}

# Scheduler settings for ASHA
scheduler_kwargs = {
    'max_t': 500,
    'grace_period': 5,
    'reduction_factor': 4,
}

# AnnData setup arguments
setup_anndata_kwargs = {
    'perturbation_key': 'condition',
    'dosage_key': 'dose_val',
    'control_group': 'control',
    'batch_key': None,
    'is_count_data': True,
    'categorical_covariate_keys': ['species'],
    'deg_uns_key': 'rank_genes_groups_cov',
    'deg_uns_cat_key': 'cov_drug_dose_name',
    'max_comb_len': 1,
}

# Setup AnnData with preprocessed data
model = cpa.CPA
model.setup_anndata(adata, **setup_anndata_kwargs)

# Resources for training
resources = {
    "cpu": 4,
    "gpu":2,
    "memory": 70 * 1024 * 1024 * 1024  # 183 GiB
}

if _original_cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _original_cuda_visible_devices
else:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

# Run hyperparameter tuning
EXPERIMENT_NAME = "cpa_autotune_cross_species"
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
        experiment_name=EXPERIMENT_NAME,
        logging_dir=LOGGING_DIR,
        adata_path=PREPROCESSED_DATA_PATH,
        sub_sample=None,
        setup_anndata_kwargs=setup_anndata_kwargs,
        use_wandb=True,
        wandb_name="cpa_tune_cross_species",
        scheduler_kwargs=scheduler_kwargs,
        plan_kwargs_keys=plan_kwargs_keys,
    )

# Save results
result_grid = experiment.result_grid
print(result_grid)
with open(os.path.join(PROJECT_ROOT, 'result_grid_cross_species.pkl'), 'wb') as f:
    pickle.dump(result_grid, f)
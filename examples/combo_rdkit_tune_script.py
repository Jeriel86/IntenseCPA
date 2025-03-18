from cpa import run_autotune
import cpa
import scanpy as sc
from ray import tune
import numpy as np
import pickle
import os
import gdown

# Define paths
PROJECT_ROOT = "/home/nmbiedou/Documents/cpa"
ORIGINAL_DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "combo_sciplex_prep_hvg_filtered.h5ad")
PREPROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "datasets",
                                      "combo_sciplex_prep_hvg_filtered_preprocessed_rdkit.h5ad")
LOGGING_DIR = os.path.join(PROJECT_ROOT, "Combo_Rdkit_autotune")



os.makedirs(LOGGING_DIR, exist_ok=True)
# Check if preprocessed data exists; if not, create and save it
if not os.path.exists(PREPROCESSED_DATA_PATH):
    # Load original data
    adata = sc.read_h5ad(ORIGINAL_DATA_PATH)
    adata.X = adata.layers['counts'].copy()  # Set raw counts

    # Add control column as required
    adata.obs['control'] = (adata.obs['condition_ID'] == 'CHEMBL504').astype(int)

    # Save the preprocessed data
    adata.write_h5ad(PREPROCESSED_DATA_PATH)
    print(f"Preprocessed data saved to {PREPROCESSED_DATA_PATH}")
else:
    print(f"Loading preprocessed data from {PREPROCESSED_DATA_PATH}")

# Load the preprocessed data
adata = sc.read_h5ad(PREPROCESSED_DATA_PATH)

# Subsample the data
sc.pp.subsample(adata, fraction=0.1)

# Model hyperparameters for tuning (adapted for RDKit)
model_args = {
    'n_latent': tune.choice([32, 64, 128, 256]),
    'recon_loss': tune.choice(['nb']),
    'doser_type': tune.choice(['linear', 'logsigm']),  # Added linear as per RDKit example
    'n_hidden_encoder': tune.choice([128, 256, 512, 1024]),
    'n_layers_encoder': tune.choice([1, 2, 3, 4, 5]),
    'n_hidden_decoder': tune.choice([128, 256, 512, 1024]),
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
    'split_key': 'split_1ct_MEC',
    'train_split': 'train',
    'valid_split': 'valid',
    'test_split': 'ood',
    'use_rdkit_embeddings': tune.choice([True]),  # Fixed to True for RDKit
    'use_intense': True,
    'intense_reg_rate': tune.choice([0.01, 0.05, 0.1, 0.001])
}

# Training hyperparameters for tuning
train_args = {
    'n_epochs_adv_warmup': tune.choice([0, 1, 3, 5, 10, 50, 100]),  # Added 100 from RDKit example
    'n_epochs_kl_warmup': tune.choice([None]),
    'n_epochs_pretrain_ae': tune.choice([0, 1, 3, 5, 10, 30, 50]),
    'adv_steps': tune.choice([None, 2, 3, 5, 10, 15, 20, 25, 30]),  # Added None from RDKit example
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
    'max_epochs': 2000,
    'use_gpu': False,
    'early_stopping_patience':  10,
    'check_val_every_n_epoch': 5,
    'batch_size': 512,  # Set to match RDKit example
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

# AnnData setup arguments (adapted for RDKit)
setup_anndata_kwargs = {
    'perturbation_key': 'condition_ID',
    'dosage_key': 'log_dose',
    'control_group': 'CHEMBL504',
    'batch_key': None,
    'smiles_key': 'smiles_rdkit',  # Added for RDKit
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
    "memory": 350 * 1024 * 1024 * 1024  # 183 GiB
}

# Run hyperparameter tuning
EXPERIMENT_NAME = "cpa_autotune_combo_rdkit"
CHECKPOINT_DIR  = os.path.join(LOGGING_DIR, EXPERIMENT_NAME)

resume_experiment = os.path.exists(CHECKPOINT_DIR)
if resume_experiment:
    print(f"Resuming experiment from {CHECKPOINT_DIR}")
    # Load the existing experiment to get the tuner and resume
    from ray.tune import Tuner

    tuner = Tuner.restore(CHECKPOINT_DIR, trainable=None)  # trainable will be re-inferred
    result_grid = tuner.fit()
    experiment = AutotuneExperiment(
        model_cls=model,
        data=adata,
        metrics=["cpa_metric", "r2_mean_deg", "r2_var_deg", "r2_mean_lfc_deg", "r2_var_lfc_deg"],
        mode="max",
        search_space=search_space,
        num_samples=500,
        scheduler="asha",
        searcher="hyperopt",
        seed=1,
        resources=resources,
        name=EXPERIMENT_NAME,
        logging_dir=LOGGING_DIR,
        scheduler_kwargs=scheduler_kwargs,
        adata_path=PREPROCESSED_DATA_PATH,
        sub_sample=0.1,
        setup_anndata_kwargs=setup_anndata_kwargs,
        plan_kwargs_keys=plan_kwargs_keys,
    )
    experiment.result_grid = result_grid
else:
    print(f"Starting new experiment at {CHECKPOINT_DIR}")
    experiment = run_autotune(
        model_cls=model,
        data=adata,
        metrics=["cpa_metric", "r2_mean_deg", "r2_var_deg", "r2_mean_lfc_deg", "r2_var_lfc_deg"],
        mode="max",
        search_space=search_space,
        num_samples=500,
        scheduler="asha",
        searcher="hyperopt",
        seed=1,
        resources=resources,
        experiment_name=EXPERIMENT_NAME,
        logging_dir=LOGGING_DIR,
        adata_path=PREPROCESSED_DATA_PATH,
        sub_sample=0.1,
        setup_anndata_kwargs=setup_anndata_kwargs,
        use_wandb=False,
        wandb_name="cpa_tune_combo_rdkit",
        scheduler_kwargs=scheduler_kwargs,
        plan_kwargs_keys=plan_kwargs_keys,
    )
# Save results
result_grid = experiment.result_grid
print(result_grid)
with open(os.path.join(PROJECT_ROOT, 'result_grid_combo_rdkit.pkl'), 'wb') as f:
    pickle.dump(result_grid, f)

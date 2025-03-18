from cpa import run_autotune
import cpa
import scanpy as sc
from ray import tune
import numpy as np
import pickle
import os

# Define the path to the Norman2019 dataset
PROJECT_ROOT = "/home/nmbiedou/Documents/cpa"
ORIGINAL_DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "Norman2019_normalized_hvg.h5ad")
PREPROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "Norman2019_normalized_hvg_preprocessed.h5ad")
LOGGING_DIR = os.path.join(PROJECT_ROOT,"autotune" ,"Norman_autotune")

os.makedirs(LOGGING_DIR, exist_ok=True)

if not os.path.exists(PREPROCESSED_DATA_PATH):
    # Load original data
    adata = sc.read_h5ad(ORIGINAL_DATA_PATH)
    adata.X = adata.layers['counts'].copy()  # Set raw counts

    # Subsample the data to speed up tuning (optional)
    sc.pp.subsample(adata, fraction=0.1)

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
    'n_latent': tune.choice([32, 64, 128, 256]),
    'recon_loss': tune.choice(['nb']),  # Negative binomial loss as used in the reference script
    'doser_type': tune.choice(['linear', 'logsigm']),  # Include 'linear' from the reference script

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

    'variational': tune.choice([False]),  # Non-variational model as in the reference script
    'seed': tune.randint(0, 10000),
    'use_intense': True,
    'intense_reg_rate':  0.05,

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
}

# Store keys for plan_kwargs
plan_kwargs_keys = list(train_args.keys())

# Additional trainer arguments
trainer_actual_args = {
    'max_epochs': 200,
    'use_gpu': False,
    'early_stopping_patience':  10,
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

# Run the hyperparameter tuning experiment
EXPERIMENT_NAME = "cpa_autotune_norman_2"
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
        num_samples=200,
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
        metrics=["cpa_metric", "disnt_basal", "disnt_after", "r2_mean", "val_r2_mean", "val_r2_var", "val_recon"],
        mode="max",  # Maximize the first metric (cpa_metric)
        search_space=search_space,
        num_samples=500,  # Number of trials to run (adjust as needed)
        scheduler="asha",
        searcher="hyperopt",
        seed=1,
        resources={"cpu": 40,"gpu": 4,"memory": 170 * 1024 * 1024 * 1024},  # Adjust based on hardware
        experiment_name=EXPERIMENT_NAME,
        logging_dir=LOGGING_DIR,  # Update to desired logging directory
        adata_path=PREPROCESSED_DATA_PATH,
        sub_sample=0.1,
        setup_anndata_kwargs=setup_anndata_kwargs,
        use_wandb=False,
        wandb_name="cpa_tune_norman",
        scheduler_kwargs=scheduler_kwargs,
        plan_kwargs_keys=plan_kwargs_keys,
    )

# Save the tuning results
result_grid = experiment.result_grid
result_file_path = os.path.join(LOGGING_DIR, 'result_grid_norman_2.pkl')
with open(result_file_path, 'wb') as f:
    pickle.dump(result_grid, f)
import sys
import os
import gdown
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from collections import defaultdict
from tqdm import tqdm
import argparse
import cpa
import scanpy as sc

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Run a single CPA experiment")
parser.add_argument("--seed", type=int, required=True, help="Random seed for the experiment")
parser.add_argument("--use_intense", type=int, choices=[0, 1], required=True, help="0 for original CPA, 1 for intense CPA")
parser.add_argument("--intense_reg_rate", type=float, default=None, help="Regularization rate for intense CPA")
parser.add_argument("--intense_p", type=int, default=None, help="p value for intense CPA")
args = parser.parse_args()

# --- Setting up environment ---
current_dir = "/scratch/nmbiedou"
data_dir = "/home/nmbiedou/Documents/cpa"
sc.settings.set_figure_params(dpi=100)
data_path = os.path.join(data_dir, "datasets", "kang_normalized_hvg.h5ad")

# Save path based on parameters
if args.use_intense:
    save_path = os.path.join(current_dir, 'lightning_logs/Kang_rite', f'Kang_Intense_reg_{str(args.intense_reg_rate).replace(".", "_")}_p_{args.intense_p}_seed_{args.seed}')
else:
    save_path = os.path.join(current_dir, 'lightning_logs/Kang_rite', f'Kang_Original_seed_{args.seed}')
os.makedirs(save_path, exist_ok=True)
sc.settings.figdir = save_path
# --- Loading dataset ---
try:
    adata = sc.read(data_path)
except:
    gdown.download('https://drive.google.com/uc?export=download&id=1z8gGKQ6Doi2blCU2IVihKA38h5fORRp')
    adata = sc.read(data_path)

print(f"Running experiment with seed: {args.seed}, use_intense: {args.use_intense}, intense_reg_rate: {args.intense_reg_rate}, intense_p: {args.intense_p}")
print(adata)

adata.X = adata.layers['counts'].copy()
adata.obs['dose'] = adata.obs['condition'].apply(lambda x: '+'.join(['1.0' for _ in x.split('+')]))

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
if args.use_intense:
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
        "seed": args.seed,
        "use_intense": True,
        "use_rite": True,
        "intense_reg_rate": args.intense_reg_rate,
        "intense_p": args.intense_p,
        "interaction_order":2,
        "rite_factor":0.9
    }
    trainer_params = {
        "n_epochs_adv_warmup": 10,
        "n_epochs_kl_warmup": None,
        "n_epochs_pretrain_ae": 30,
        "adv_steps": 2,
        "mixup_alpha": 0.2,
        "n_epochs_mixup_warmup": 10,
        "n_layers_adv": 2,
        "n_hidden_adv": 64,
        "use_batch_norm_adv": True,
        "use_layer_norm_adv": False,
        "dropout_rate_adv": 0.25,
        "pen_adv": 0.21176187723505563,
        "reg_adv": 3.3191059919680557,
        "lr": 0.00656775854452724,
        "wd": 0.00000017930296245015,
        "doser_lr": 0.00034816913906047706,
        "doser_wd": 0.00000139054966251716,
        "adv_lr": 0.0005237774528387638,
        "adv_wd": 0.00000015681818353966,
        "adv_loss": "cce",
        "do_clip_grad": True,
        "gradient_clip_value": 1.0,
        "step_size_lr": 25,
        "momentum": 0.9007079992500057
    }
else:
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
        "seed": args.seed,
        "use_intense": False
    }
    trainer_params = {
        "n_epochs_kl_warmup": None,
        "n_epochs_pretrain_ae": 30,
        "n_epochs_adv_warmup": 50,
        "n_epochs_mixup_warmup": 0,
        "mixup_alpha": 0.0,
        "adv_steps": None,
        "n_hidden_adv": 64,
        "n_layers_adv": 3,
        "use_batch_norm_adv": True,
        "use_layer_norm_adv": False,
        "dropout_rate_adv": 0.3,
        "reg_adv": 20.0,
        "pen_adv": 5.0,
        "lr": 0.0003,
        "wd": 4e-07,
        "adv_lr": 0.0003,
        "adv_wd": 4e-07,
        "adv_loss": "cce",
        "doser_lr": 0.0003,
        "doser_wd": 4e-07,
        "do_clip_grad": True,
        "gradient_clip_value": 1.0,
        "step_size_lr": 10,
    }

# --- Creating CPA Model ---
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
    use_gpu=True,
    batch_size=512,
    plan_kwargs=trainer_params,
    early_stopping_patience=10 if args.use_intense else 5,
    check_val_every_n_epoch=5,
    save_path=save_path,
)

plot_path = os.path.join(save_path, "history.png")
cpa.pl.plot_history(model, plot_path)

# --- Latent Space Visualization ---
latent_outputs = model.get_latent_representation(adata, batch_size=2048)

# Basal latent space
sc.pp.neighbors(latent_outputs['latent_basal'])
sc.tl.umap(latent_outputs['latent_basal'])
sc.pl.umap(
    latent_outputs['latent_basal'],
    color=['condition', 'cell_type'],
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
    color=['condition', 'cell_type'],
    frameon=False,
    wspace=0.3,
    save=f'latent_after_seed_{args.seed}.png',
    show=False
)

# --- Evaluation ---
model.predict(adata, batch_size=2048)
n_top_degs = [10, 20, 50, None]
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

df = pd.DataFrame(results)
df.to_csv(os.path.join(save_path, f'evaluation_results_seed_{args.seed}.csv'), index=False)

# Save results to a shared location for aggregation
results_dir = os.path.join(current_dir, 'lightning_logs/Kang_rite', 'experiment_results')
os.makedirs(results_dir, exist_ok=True)
if args.use_intense:
    result_file = os.path.join(results_dir, f'result_seed_{args.seed}_intense_{str(args.intense_reg_rate).replace(".", "_")}_{args.intense_p}.csv')
else:
    result_file = os.path.join(results_dir, f'result_seed_{args.seed}_original.csv')
df.to_csv(result_file, index=False)
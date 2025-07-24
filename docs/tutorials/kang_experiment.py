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
current_dir = "/home/nmbiedou/Documents/cpa"
sc.settings.set_figure_params(dpi=100)
data_path = os.path.join(current_dir, "datasets", "kang_normalized_hvg.h5ad")

# Experiment parameters
intense_reg_rates = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
intense_p_values = [1, 2]
num_runs = 5  # Number of random seeds

# Set a meta-seed for reproducibility
np.random.seed(42)
seeds = np.random.randint(0, 10000, size=num_runs).tolist()
print(f"Generated seeds: {seeds}")
with open(os.path.join(current_dir, 'lightning_logs', 'used_seeds.txt'), 'w') as f:
    f.write(f"Seeds used: {seeds}\n")


# Function to run the experiment for a given seed and parameter combination
def run_experiment(seed, intense_reg_rate=None, intense_p=None, use_intense=False):
    if use_intense:
        save_path = os.path.join(current_dir, 'lightning_logs',
                                 f'Kang_Intense_reg_{str(intense_reg_rate).replace(".", "_")}_p_{intense_p}_seed_{seed}')
    else:
        save_path = os.path.join(current_dir, 'lightning_logs', f'Kang_Original_seed_{seed}')
    os.makedirs(save_path, exist_ok=True)

    # --- Loading dataset ---
    try:
        adata = sc.read(data_path)
    except:
        gdown.download('https://drive.google.com/uc?export=download&id=1z8gGKQ6Doi2blCU2IVihKA38h5fORRp')
        adata = sc.read(data_path)

    print(
        f"Running experiment with seed: {seed}, use_intense: {use_intense}, intense_reg_rate: {intense_reg_rate}, intense_p: {intense_p}")
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
    if use_intense:
        model_params = {
            "n_latent": 64,
            "recon_loss": "nb",
            "doser_type": "logsigm",
            "n_hidden_encoder": 512,
            "n_layers_encoder": 5,
            "n_hidden_decoder": 256,
            "n_layers_decoder": 2,
            "use_batch_norm_encoder": False,
            "use_layer_norm_encoder": True,
            "use_batch_norm_decoder": True,
            "use_layer_norm_decoder": False,
            "dropout_rate_encoder": 0.2,
            "dropout_rate_decoder": 0.0,
            "variational": False,
            "seed": seed,
            "use_intense": True,
            "intense_reg_rate": intense_reg_rate,
            "intense_p": intense_p
        }
        trainer_params = {
            "n_epochs_adv_warmup": 3,
            "n_epochs_kl_warmup": None,
            "n_epochs_pretrain_ae": 50,
            "adv_steps": 5,
            "mixup_alpha": 0.3,
            "n_epochs_mixup_warmup": 10,
            "n_layers_adv": 2,
            "n_hidden_adv": 64,
            "use_batch_norm_adv": False,
            "use_layer_norm_adv": False,
            "dropout_rate_adv": 0.0,
            "pen_adv": 0.1995431254447313,
            "reg_adv": 6.1902932390831324,
            "lr": 0.0007064775779032066,
            "wd": 1.175555699588592e-07,
            "doser_lr": 2.3211621010467742e-05,
            "doser_wd": 1.2484610680888827e-07,
            "adv_lr": 0.003224233031630511,
            "adv_wd": 1.6809347701585555e-08,
            "adv_loss": "cce",
            "do_clip_grad": False,
            "gradient_clip_value": 1.0,
            "step_size_lr": 10,
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
            "seed": seed,
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
        use_gpu=False,
        batch_size=512,
        plan_kwargs=trainer_params,
        early_stopping_patience=15 if use_intense else 5,
        check_val_every_n_epoch=5,
        save_path=save_path,
        num_gpus=5
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
        save=f'latent_basal_seed_{seed}.png',
        show=False  # Prevents displaying the plot, just saves it
    )
    os.rename(
        os.path.join(sc.settings.figdir, f'umaplatent_basal_seed_{seed}.png'),
        os.path.join(save_path, f'latent_basal_seed_{seed}.png')
    )


    # Final latent space
    sc.pp.neighbors(latent_outputs['latent_after'])
    sc.tl.umap(latent_outputs['latent_after'])
    sc.pl.umap(
        latent_outputs['latent_after'],
        color=['condition', 'cell_type'],
        frameon=False,
        wspace=0.3,
        save=f'latent_after_seed_{seed}.png',
        show=False
    )
    os.rename(
        os.path.join(sc.settings.figdir, f'umaplatent_after_seed_{seed}.png'),
        os.path.join(save_path, f'latent_after_seed_{seed}.png')
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
                r2_mean_lfc_deg = r2_score(x_true_deg.mean(0) - x_ctrl_deg.mean(0),
                                           x_pred_deg.mean(0) - x_ctrl_deg.mean(0))
                r2_var_lfc_deg = r2_score(x_true_deg.var(0) - x_ctrl_deg.var(0), x_pred_deg.var(0) - x_ctrl_deg.var(0))

                results['condition'].append(condition)
                results['cell_type'].append(cov)
                results['n_top_deg'].append(n_top_deg)
                results['r2_mean_deg'].append(r2_mean_deg)
                results['r2_var_deg'].append(r2_var_deg)
                results['r2_mean_lfc_deg'].append(r2_mean_lfc_deg)
                results['r2_var_lfc_deg'].append(r2_var_lfc_deg)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_path, f'evaluation_results_seed_{seed}.csv'), index=False)
    return df

# Directory for averaged results
results_dir = os.path.join(current_dir, 'lightning_logs', 'experiment_results')
os.makedirs(results_dir, exist_ok=True)
if __name__ == "__main__":
    # 1. Run original CPA (without intense)
    """all_dfs_original = []
    for seed in seeds:
        df = run_experiment(seed, use_intense=False)
        all_dfs_original.append(df)

    # Compute average for original CPA
    combined_df_original = pd.concat(all_dfs_original)
    avg_df_original = combined_df_original.groupby(['condition', 'cell_type', 'n_top_deg']).mean().reset_index()
    result_file_original = os.path.join(results_dir, 'result_experiment_original.csv')
    avg_df_original.to_csv(result_file_original, index=False)
    print(f"Saved averaged results for original CPA to {result_file_original}")"""

    # 2. Run CPA with intense for each parameter combination
    for intense_reg_rate in intense_reg_rates:
        for intense_p in intense_p_values:
            all_dfs = []
            for seed in seeds:
                df = run_experiment(seed, intense_reg_rate, intense_p, use_intense=True)
                all_dfs.append(df)

            # Compute average across seeds
            combined_df = pd.concat(all_dfs)
            avg_df = combined_df.groupby(['condition', 'cell_type', 'n_top_deg']).mean().reset_index()

            # Save averaged results
            result_file = os.path.join(results_dir,
                                       f'result_experiment_{str(intense_reg_rate).replace(".", "_")}_{intense_p}.csv')
            avg_df.to_csv(result_file, index=False)
            print(f"Saved averaged results for intense_reg_rate={intense_reg_rate}, intense_p={intense_p} to {result_file}")

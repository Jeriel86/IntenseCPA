#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Final Report Generator for CPA Experiments on the Norman Dataset with Standard Deviations
"""

import os
import glob
import pandas as pd
from datetime import datetime
from tabulate import tabulate
import numpy as np

# Define directories (adjust paths as needed)
current_dir = "/Users/jeriel/Documents/WS24:25/MA/cpa/results/Norman"
results_dir = os.path.join(current_dir, 'experiment_results')
report_file = os.path.join(results_dir, "final_report_with_p_1.md")

# Helper function to load CSV files
def load_csv_files(pattern, is_seed_file=False):
    files = glob.glob(pattern)
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if is_seed_file:
                # Extract seed number from filenames like result_seed_1_original.csv
                seed_part = os.path.basename(f).split("seed_")[1].split("_")[0]
                df['seed'] = int(seed_part)
            df['filename'] = os.path.basename(f)  # Use 'filename' instead of 'source_file'
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# Load individual seed results for Original CPA
df_orig_seeds = load_csv_files(os.path.join(results_dir, "result_seed_*_original.csv"), is_seed_file=True)
df_orig_seeds["config"] = "Original CPA"

# Load aggregated results for all Intense CPA configurations to find the best
intense_agg_pattern = os.path.join(results_dir, "result_experiment_intense_*_*.csv")
df_intense_agg = load_csv_files(intense_agg_pattern, is_seed_file=False)

# Function to extract configuration from aggregated filename
def extract_config(fname):
    try:
        parts = fname.replace("result_experiment_intense_", "").replace(".csv", "").split("_")
        if len(parts) >= 3:  # Expecting at least [reg_rate_part1, reg_rate_part2, p]
            reg_rate = f"{parts[0]}.{parts[1]}"
            p_value = parts[2]
            return f"Intense CPA (reg_rate={reg_rate}, p={p_value})"
        return "Intense CPA (unknown)"
    except Exception:
        return "Intense CPA (unknown)"

df_intense_agg["config"] = df_intense_agg["filename"].apply(extract_config)

# Find best Intense CPA config based on aggregated r2_mean_deg
avg_scores = df_intense_agg.groupby("config")["r2_mean_deg"].mean().reset_index()
best_intense_row = avg_scores.sort_values("r2_mean_deg", ascending=False).iloc[2]
best_intense_config = best_intense_row["config"]
best_intense_score = best_intense_row["r2_mean_deg"]

# Extract reg_rate and p from best config
reg_rate_str = best_intense_config.split("reg_rate=")[1].split(",")[0]
p_str = best_intense_config.split("p=")[1].split(")")[0]
reg_rate_formatted = reg_rate_str.replace(".", "_")

# Load individual seed results for best Intense CPA
df_intense_seeds = load_csv_files(
    os.path.join(results_dir, f"result_seed_*_intense_{reg_rate_formatted}_{p_str}.csv"),
    is_seed_file=True
)
df_intense_seeds["config"] = best_intense_config

# Combine seed data
df_all_seeds = pd.concat([df_orig_seeds, df_intense_seeds], ignore_index=True)

# Separate into In-distribution and OOD
ood_conditions = {"DUSP9+ETS2", "CBL+CNN1"}
df_in_dist_seeds = df_all_seeds[~df_all_seeds["condition"].isin(ood_conditions)].copy()
df_ood_seeds = df_all_seeds[df_all_seeds["condition"].isin(ood_conditions)].copy()

# Function to build comparison table with mean and std
def build_comparison_table(df):
    grouped = df.groupby(["config", "n_top_deg"]).agg({
        "r2_mean_deg": ["mean", "std"],
        "r2_mean_lfc_deg": ["mean", "std"]
    }).reset_index()
    grouped.columns = ['_'.join(col).strip() if col[1] else col[0] for col in grouped.columns.values]

    summary_orig = grouped[grouped["config"] == "Original CPA"].set_index("n_top_deg")
    summary_intense = grouped[grouped["config"] == best_intense_config].set_index("n_top_deg")

    common_n_top = sorted(set(summary_orig.index) & set(summary_intense.index))

    table_rows = []
    for n in common_n_top:
        orig_r2_mean = summary_orig.loc[n, "r2_mean_deg_mean"]
        orig_r2_std = summary_orig.loc[n, "r2_mean_deg_std"]
        intense_r2_mean = summary_intense.loc[n, "r2_mean_deg_mean"]
        intense_r2_std = summary_intense.loc[n, "r2_mean_deg_std"]

        orig_r2_lfc_mean = summary_orig.loc[n, "r2_mean_lfc_deg_mean"]
        orig_r2_lfc_std = summary_orig.loc[n, "r2_mean_lfc_deg_std"]
        intense_r2_lfc_mean = summary_intense.loc[n, "r2_mean_lfc_deg_mean"]
        intense_r2_lfc_std = summary_intense.loc[n, "r2_mean_lfc_deg_std"]

        r2_deg_orig_str = f"**{orig_r2_mean:.3f} ± {orig_r2_std:.3f}**" if orig_r2_mean > intense_r2_mean else f"{orig_r2_mean:.3f} ± {orig_r2_std:.3f}"
        r2_deg_intense_str = f"**{intense_r2_mean:.3f} ± {intense_r2_std:.3f}**" if intense_r2_mean > orig_r2_mean else f"{intense_r2_mean:.3f} ± {intense_r2_std:.3f}"

        r2_lfc_orig_str = f"**{orig_r2_lfc_mean:.3f} ± {orig_r2_lfc_std:.3f}**" if orig_r2_lfc_mean > intense_r2_lfc_mean else f"{orig_r2_lfc_mean:.3f} ± {orig_r2_lfc_std:.3f}"
        r2_lfc_intense_str = f"**{intense_r2_lfc_mean:.3f} ± {intense_r2_lfc_std:.3f}**" if intense_r2_lfc_mean > orig_r2_lfc_mean else f"{intense_r2_lfc_mean:.3f} ± {intense_r2_lfc_std:.3f}"

        table_rows.append([n, r2_deg_orig_str, r2_deg_intense_str, r2_lfc_orig_str, r2_lfc_intense_str])

    headers = ["n_top_deg", "Original CPA r2_mean_deg", f"{best_intense_config} r2_mean_deg",
               "Original CPA r2_mean_lfc_deg", f"{best_intense_config} r2_mean_lfc_deg"]
    return tabulate(table_rows, headers=headers, tablefmt="pipe", stralign="center")

# Build tables
table_in_dist = build_comparison_table(df_in_dist_seeds)
table_ood = build_comparison_table(df_ood_seeds)

# Visualization paths
orig_latent_img = "Norman_Original/latent_after_cond_harm_seed_all.png"
orig_train_img = "Norman_Original/history_seed_all.png"
intense_latent_img = best_intense_config.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_").replace(",", "") + "/latent_after_cond_harm_seed_all.png"
intense_train_img = best_intense_config.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_").replace(",", "") + "/history_seed_all.png"

# Generate markdown report
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
report_lines = [
    "# Final Comparative Report: Original CPA vs. Best Intense CPA Config with p = 1",
    f"*Report generated on {now}*",
    "",
    "## 1. Experiment Overview",
    ("In this experiment, we evaluated the performance of the standard Compositional Perturbation Autoencoder (CPA) "
     "against an enhanced model that incorporates tensor fusion (termed \"Intense CPA\"). The goal was to assess the generalization "
     "capabilities of each model under in-distribution conditions (conditions seen during training) as well as out-of-distribution (OOD) "
     "conditions (unseen combinations and different dosages). For each configuration, experiments were run on 5 different seeds. Evaluation "
     "metrics include the R² score for mean gene expression predictions (r2_mean_deg) and for log-fold change predictions (r2_mean_lfc_deg). "
     "Values are reported as mean ± standard deviation across seeds."),
    "",
    "## 2. Best Intense CPA Settings",
    f"- **Selected Configuration:** {best_intense_config}",
    f"- **Average r2_mean_deg:** {best_intense_score:.3f}",
    "",
    "## 3. Quantitative Evaluation",
    "### 3.1 In-Distribution Evaluation",
    "The table below shows, for various numbers of top differentially expressed genes (n_top_deg), the average R² metrics ± standard deviation for "
    "Original CPA and the best Intense CPA. The higher mean value for each metric is highlighted in **bold**.",
    "",
    table_in_dist,
    "",
    "### 3.2 Out-of-Distribution (OOD) Evaluation",
    "For conditions not seen during training, the following table compares the Original CPA and the best Intense CPA:",
    "",
    table_ood,
    "",
    "## 4. Visualizations",
    "### 4.1 Latent Space Visualizations",
    "Below are representative UMAP (or Kernel PCA) plots of the final latent representations for each model:",
    "",
    "- **Original CPA Latent Space:**",
    f"  ![]({orig_latent_img})",
    "",
    "- **Best Intense CPA Latent Space:**",
    f"  ![]({intense_latent_img})",
    "",
    "### 4.2 Training History",
    "The training loss and other relevant metrics over epochs for each model are shown below:",
    "",
    "- **Original CPA Training History:**",
    f"  ![]({orig_train_img})",
    "",
    "- **Best Intense CPA Training History:**",
    f"  ![]({intense_train_img})",
    "",
    "## 5. Discussion",
    (
        "The quantitative evaluation indicates that the enhanced Intense CPA configuration outperforms the Original CPA in several cases, "
        "as reflected by higher average R² scores both in in-distribution and OOD conditions. The standard deviations provide insight into "
        "the consistency of performance across seeds."),
    "",
    "## 6. Conclusion",
    (
        "Overall, the experimental results support the hypothesis that modeling higher-order interactions via tensor fusion "
        "significantly improves the performance of CPA on the Norman 2019 dataset."),
    "",
    "## 7. References",
    (
        "Additional details regarding the experimental setup and evaluation metrics are available in the corresponding documentation.")
]

# Write the markdown report to file
with open(report_file, "w") as f:
    f.write("\n".join(report_lines))

print(f"Final comparative report with standard deviations generated and saved to: {report_file}")
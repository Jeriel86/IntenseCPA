import os
import pandas as pd
import numpy as np

# Set up directories
current_dir = "/home/nmbiedou/Documents/cpa"
results_dir = os.path.join(current_dir, 'lightning_logs', 'experiment_results')
os.makedirs(results_dir, exist_ok=True)

# Load original CPA averaged results
avg_df_original = pd.read_csv(os.path.join(results_dir, 'result_experiment_original.csv'))
print(f"Loaded averaged results for original CPA from {os.path.join(results_dir, 'result_experiment_original.csv')}")

# Define intense parameters
intense_reg_rates = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
intense_p_values = [1, 2]

# Load intense CPA averaged results and track the best performer
best_intense_score = -float('inf')  # Higher is better for r2_mean_deg
best_intense_config = None
best_intense_df = None

for reg_rate in intense_reg_rates:
    for p in intense_p_values:
        file_name = f'result_experiment_{str(reg_rate).replace(".", "_")}_{p}.csv'
        file_path = os.path.join(results_dir, file_name)
        avg_df = pd.read_csv(file_path)
        print(f"Loaded averaged results for intense_reg_rate={reg_rate}, intense_p={p} from {file_path}")

        # Evaluate performance (mean r2_mean_deg across all n_top_deg)
        mean_r2 = avg_df['r2_mean_deg'].mean()
        if mean_r2 > best_intense_score:
            best_intense_score = mean_r2
            best_intense_config = (reg_rate, p)
            best_intense_df = avg_df

# Compare best intense with original CPA
original_r2_mean = avg_df_original['r2_mean_deg'].mean()
print(f"\nBest intense configuration: intense_reg_rate={best_intense_config[0]}, intense_p={best_intense_config[1]}")
print(f"Best intense mean r2_mean_deg: {best_intense_score:.4f}")
print(f"Original CPA mean r2_mean_deg: {original_r2_mean:.4f}")
print(f"Difference (Best Intense - Original): {best_intense_score - original_r2_mean:.4f}")

# Generate Markdown report
report_file = os.path.join(current_dir, 'lightning_logs', 'report.md')
with open(report_file, 'w') as f:
    f.write("# CPA Experiment Report\n\n")
    f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Summary
    f.write("## Summary\n\n")
    f.write(
        f"**Best Intense Configuration**: `intense_reg_rate={best_intense_config[0]}, intense_p={best_intense_config[1]}`\n\n")
    f.write(f"**Best Intense Mean R² (r2_mean_deg)**: {best_intense_score:.4f}\n")
    f.write(f"**Original CPA Mean R² (r2_mean_deg)**: {original_r2_mean:.4f}\n")
    f.write(f"**Difference (Best Intense - Original)**: {best_intense_score - original_r2_mean:.4f}\n\n")
    f.write(
        "The best intense configuration was determined by averaging the `r2_mean_deg` metric across all `n_top_deg` values for each combination of `intense_reg_rate` and `intense_p`. All results below are averaged across seeds (7270, 860, 5390, 5191, 5734) by the experiment script.\n\n")

    # Evaluation Results
    f.write("## Evaluation Results\n\n")
    f.write(
        "The tables below summarize the averaged evaluation metrics for the original CPA and the best intense CPA configuration, grouped by `n_top_deg`.\n\n")

    # Numeric columns to average
    numeric_cols = ['r2_mean_deg', 'r2_var_deg', 'r2_mean_lfc_deg', 'r2_var_lfc_deg']

    # Original CPA Metrics
    f.write("### Original CPA Metrics\n")
    f.write("| n_top_deg | r2_mean_deg | r2_var_deg | r2_mean_lfc_deg | r2_var_lfc_deg |\n")
    f.write("|-----------|-------------|------------|-----------------|----------------|\n")
    grouped_original = avg_df_original.groupby('n_top_deg')[numeric_cols].mean().reset_index()
    grouped_original['n_top_deg'] = grouped_original['n_top_deg'].apply(lambda x: -1 if x == 'all' else int(x))
    grouped_original = grouped_original.sort_values('n_top_deg').reset_index(drop=True)
    grouped_original['n_top_deg'] = grouped_original['n_top_deg'].apply(lambda x: 'all' if x == -1 else x)
    for _, row in grouped_original.iterrows():
        n_top_deg = row['n_top_deg']
        f.write(
            f"| {n_top_deg} | {row['r2_mean_deg']:.4f} | {row['r2_var_deg']:.4f} | {row['r2_mean_lfc_deg']:.4f} | {row['r2_var_lfc_deg']:.4f} |\n")
    f.write("\n")

    # Best Intense CPA Metrics
    f.write(
        f"### Best Intense CPA Metrics (intense_reg_rate={best_intense_config[0]}, intense_p={best_intense_config[1]})\n")
    f.write("| n_top_deg | r2_mean_deg | r2_var_deg | r2_mean_lfc_deg | r2_var_lfc_deg |\n")
    f.write("|-----------|-------------|------------|-----------------|----------------|\n")
    grouped_intense = best_intense_df.groupby('n_top_deg')[numeric_cols].mean().reset_index()
    grouped_intense['n_top_deg'] = grouped_intense['n_top_deg'].apply(lambda x: -1 if x == 'all' else int(x))
    grouped_intense = grouped_intense.sort_values('n_top_deg').reset_index(drop=True)
    grouped_intense['n_top_deg'] = grouped_intense['n_top_deg'].apply(lambda x: 'all' if x == -1 else x)
    for _, row in grouped_intense.iterrows():
        n_top_deg = row['n_top_deg']
        f.write(
            f"| {n_top_deg} | {row['r2_mean_deg']:.4f} | {row['r2_var_deg']:.4f} | {row['r2_mean_lfc_deg']:.4f} | {row['r2_var_lfc_deg']:.4f} |\n")
    f.write("\n")

print(f"Generated report at {report_file}")
import os
import pandas as pd
from glob import glob

current_dir = "/Users/jeriel/Documents/WS24:25/MA/cpa/results/Kang_rite"
results_dir = os.path.join(current_dir, 'experiment_results')
os.makedirs(results_dir, exist_ok=True)

# Aggregate original CPA results
original_files = glob(os.path.join(results_dir, "result_seed_*_original.csv"))
all_dfs_original = [pd.read_csv(f) for f in original_files]
combined_df_original = pd.concat(all_dfs_original)
avg_df_original = combined_df_original.groupby(['condition', 'cell_type', 'n_top_deg']).mean().reset_index() #, 'cell_type'
avg_df_original.to_csv(os.path.join(results_dir, 'result_experiment_original.csv'), index=False)
print(f"Saved averaged results for original CPA to {os.path.join(results_dir, 'result_experiment_original.csv')}")

# Aggregate intense CPA results
intense_reg_rates = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
intense_p_values = [1, 2]
for reg_rate in intense_reg_rates:
    for p in intense_p_values:
        intense_files = glob(os.path.join(results_dir, f"result_seed_*_intense_{str(reg_rate).replace('.', '_')}_{p}.csv"))
        all_dfs = [pd.read_csv(f) for f in intense_files]
        combined_df = pd.concat(all_dfs)
        avg_df = combined_df.groupby(['condition','cell_type', 'n_top_deg']).mean().reset_index() #, 'cell_type'
        result_file = os.path.join(results_dir, f'result_experiment_intense_{str(reg_rate).replace(".", "_")}_{p}.csv')
        avg_df.to_csv(result_file, index=False)
        print(f"Saved averaged results for intense_reg_rate={reg_rate}, intense_p={p} to {result_file}")
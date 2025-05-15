import pickle
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG: path to your pickle file ===
PICKLE_PATH = "result_grid_sciplex3.pkl"

# 1. Load the serialized ResultGrid
with open(PICKLE_PATH, "rb") as f:
    result_grid = pickle.load(f)

# 2. Convert to DataFrame
df = result_grid.get_dataframe()

# 3. Top-5 runs by highest CPA metric
top_cpa = df.sort_values("cpa_metric", ascending=False).head(10)
print("Top 5 by CPA metric:")
print(top_cpa[["trial_id", "cpa_metric", "disnt_basal"]])

# 4. Top-5 runs by lowest distance baseline
top_distn = df.sort_values("disnt_basal", ascending=True).head(5)
print("\nTop 5 by lowest distn_baseline:")
print(top_distn[["trial_id", "cpa_metric", "disnt_basal"]])

# 5. Scatter plot: CPA vs Distance Baseline
plt.figure()
plt.scatter(df["disnt_basal"], df["cpa_metric"])
plt.xlabel("disnt_basal (lower is better)")
plt.ylabel("cpa_metric (higher is better)")
plt.title("Trade-off: CPA Metric vs. Distance Baseline")
plt.show()

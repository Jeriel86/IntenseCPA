import os
import json
import numpy as np
import pandas as pd
from ray.tune import ExperimentAnalysis
import argparse


def convert_to_python_types(obj):
    """
    Recursively convert NumPy types in an object to standard Python types.

    Args:
        obj: The object to convert (e.g., dict, list, or scalar).

    Returns:
        The object with all NumPy types converted to Python types.
    """
    if isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(value) for value in obj]
    elif isinstance(obj, np.generic):
        return obj.item()  # Converts NumPy scalar to Python scalar
    else:
        return obj


def get_config(experiment_type):
    """Generate configuration based on experiment type."""
    PROJECT_ROOT = "/home/nmbiedou/Documents/cpa"
    LOGGING_DIR_BASE = os.getenv("SLURM_TMPDIR", os.path.join(PROJECT_ROOT, "autotune"))

    configs = {
        "norman": {
            "logging_dir": LOGGING_DIR_BASE,
            "experiment_name": "cpa_autotune_norman",
            "config_file": "best_config.json",
            "results_file": "sorted_results.csv"
        },
        "combo": {
            "logging_dir": os.path.join(PROJECT_ROOT, "Combo_autotune"),
            "experiment_name": "cpa_autotune_combo",
            "config_file": "best_config_combo.json",
            "results_file": "sorted_results_combo.csv"
        },
        "combo_rdkit": {
            "logging_dir": os.path.join(PROJECT_ROOT, "Combo_Rdkit_autotune"),
            "experiment_name": "cpa_autotune_combo_rdkit",
            "config_file": "best_config_combo_rdkit.json",
            "results_file": "sorted_results_combo_rdkit.csv"
        },
        "kang": {
            "logging_dir": os.path.join(PROJECT_ROOT, "autotune"),
            "experiment_name": "kang_autotune",
            "config_file": "best_config_kang.json",
            "results_file": "sorted_results_kang.csv"
        }
    }

    if experiment_type not in configs:
        raise ValueError(f"Unknown experiment type. Choose from {list(configs.keys())}")

    config = configs[experiment_type]
    os.makedirs(config["logging_dir"], exist_ok=True)
    config["experiment_path"] = os.path.join(config["logging_dir"], config["experiment_name"])
    config["sorted_csv_path"] = os.path.join(PROJECT_ROOT, config["results_file"])
    return config


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Analyze hyperparameter tuning results")
    parser.add_argument(
        "--experiment",
        type=str,
        default=os.getenv("EXPERIMENT_TYPE", "kang"),  # Default from env var or combo_rdkit
        help="Experiment type: norman, combo, or combo_rdkit"
    )
    args = parser.parse_args()

    # Get configuration for the specified experiment
    config = get_config(args.experiment)

    # Load the hyperparameter tuning results
    analysis = ExperimentAnalysis(config["experiment_path"])

    # Find the best trial based on the 'cpa_metric' in maximization mode
    best_trial = analysis.get_best_trial(metric="cpa_metric", mode="max")
    best_config = analysis.get_best_config(metric="cpa_metric", mode="max")

    # Print the best hyperparameters and the best metric value
    print(f"\nExperiment: {args.experiment}")
    print("Best Hyperparameters:", best_config)
    print("Best Metric Value:", best_trial.last_result["cpa_metric"])

    # Convert NumPy types to Python types and save the best configuration to a JSON file
    best_config_python = convert_to_python_types(best_config)
    with open(config["config_file"], "w") as f:
        json.dump(best_config_python, f, indent=4)
    print(f"Best configuration saved to: {config['config_file']}")

    # Load all results into a pandas DataFrame
    df = analysis.dataframe()

    # Sort the DataFrame by 'cpa_metric' in descending order (max first)
    sorted_df = df.sort_values(by="cpa_metric", ascending=False)

    # Save the sorted DataFrame to a CSV file
    sorted_df.to_csv(config["sorted_csv_path"], index=False)
    print(f"Sorted results saved to: {config['sorted_csv_path']}")

    # Optionally print the unsorted DataFrame
    print("\nAll Results as DataFrame:")
    print(df)


if __name__ == "__main__":
    main()
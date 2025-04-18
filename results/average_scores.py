import os
from typing import Optional
from collections import defaultdict

import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import pandas as pd
import glob


def plot_relevance_scores(relevance_dict, save_path: Optional[str] = None):
    """
    Plot an aesthetically enhanced bar plot of relevance scores from an MKLFusion instance.

    Args:
        mkl_fusion_module: An instance of MKLFusion that implements a scores() method. The scores() method
                           should return a dictionary with modality keys and normalized relevance scores as values.

    The y-axis is fixed to [0,1] and the plot is styled using a modern Seaborn theme.
    """
    sns.set_theme(context="talk", style="white", palette="muted")

    modalities = list(relevance_dict.keys())
    scores = list(relevance_dict.values())

    # Choose a pleasant color palette for the bars. Here we use Set2 from Seaborn.
    custom_colors = sns.color_palette("Set2", len(modalities))

    # Create the figure with an appropriate size for a thesis.
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw the bar plot
    bars = ax.bar(modalities, scores, color=custom_colors, edgecolor="black", linewidth=1.5)

    # Set axis labels and title with custom fonts and weights.
    ax.set_xlabel("Modalities", fontsize=16, fontweight="bold")
    ax.set_ylabel("Relevance Score", fontsize=16, fontweight="bold")
    ax.set_title("Relevance Scores of Modalities", fontsize=18, fontweight="bold")

    # Force the y-axis to range from 0 to 1.
    ax.set_ylim(0, 1)

    # Annotate each bar with its numerical value.
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,  # add a little offset above the bar
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color="black"
        )

    # Improve layout for publication.
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=100)

    plt.show()



def calculate_average_scores(results_dir, intense_reg_rate=None, intense_p=None):
    """
    Calculate average scores from CSV files in the results directory.

    Args:
        results_dir (str): Directory containing the scores CSV files
        intense_reg_rate (float, optional): Filter by specific intense_reg_rate
        intense_p (float, optional): Filter by specific intense_p value

    Returns:
        dict: Average scores for each column
    """
    # Create pattern to match files
    pattern = f'scores_reg_*_p_*.csv'
    if intense_reg_rate is not None and intense_p is not None:
        pattern = f'scores_reg_{str(intense_reg_rate).replace(".", "_")}_p_{intense_p}.csv'

    # Get all matching files
    file_pattern = os.path.join(results_dir, pattern)
    csv_files = glob.glob(file_pattern)

    if not csv_files:
        print(f"No CSV files found matching pattern: {file_pattern}")
        return None

    # Initialize list to store all dataframes
    all_dfs = []

    # Read all CSV files
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    if not all_dfs:
        print("No valid dataframes to process")
        return None

    # Concatenate all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Calculate averages
    averages = combined_df.mean().to_dict()

    # Print results
    print(f"Processed {len(csv_files)} files with {len(combined_df)} total rows")
    print("Average scores:")
    for key, value in averages.items():
        print(f"{key}: {value:.4f}")

    return averages


def main():

    current_dir = "/Users/jeriel/Documents/WS24:25/MA/cpa/results/Combo_Order_2"
    results_dir = os.path.join(current_dir, 'experiment_results')
    plot_path_scores = os.path.join(results_dir, "scores_p_2.png")

    print("\nCalculating averages for specific parameters:")
    specific_averages = calculate_average_scores(results_dir,intense_reg_rate=0.1, intense_p=2 )
    plot_relevance_scores(specific_averages, plot_path_scores)



if __name__ == "__main__":
    main()
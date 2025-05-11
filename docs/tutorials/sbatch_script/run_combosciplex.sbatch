#!/bin/bash
#SBATCH --mem=150G              # Memory: 128 GB
#SBATCH -J Combo_Training       # Job name
#SBATCH --mail-type=END,FAIL    # Email notifications
#SBATCH -n 1                    # 1 task
#SBATCH --cpus-per-task=4       # 8 CPU cores per task
#SBATCH -N 1                    # 1 node
#SBATCH --gpus=v100:2           # 1 Tesla A30 GPU
#SBATCH --partition=gpuidle     # Target gpuidle partition
#SBATCH --output=combosciplex_intense_%j.out     # Output file
#SBATCH --error=combosciplex_intense_%j.err      # Error file

# Load required modules
source /home/nmbiedou/miniconda3/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate newcpa

# Run your Python script
python  /home/nmbiedou/Documents/cpa/docs/tutorials/combosciplex.py

# Deactivate Conda environment
conda deactivate
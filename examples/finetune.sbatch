#!/bin/bash
#SBATCH -t 2-00:00:00           # Time limit: 2 days
#SBATCH --mem=128G              # Memory: 128 GB
#SBATCH -J Kang_finetune       # Job name
#SBATCH --mail-type=END,FAIL    # Email notifications
#SBATCH -n 1                    # 1 task
#SBATCH --cpus-per-task=8       # 8 CPU cores per task
#SBATCH -N 1                    # 1 node
#SBATCH --gpus=v100:8           # 1 Tesla V100 GPU
#SBATCH --partition=gpuidle     # Target gpuidle partition
#SBATCH --output=/home/nmbiedou/Documents/cpa/examples/slurm_%A_%a.out  # Output log
#SBATCH --error=/home/nmbiedou/Documents/cpa/examples/slurm_%A_%a.err   # Error log


# Load modules

source /home/nmbiedou/miniconda3/etc/profile.d/conda.sh
conda activate newcpa  # Replace with your actual environment name

python /home/nmbiedou/Documents/cpa/examples/kang_tune_script.py

conda deactivate
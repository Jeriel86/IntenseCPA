#!/bin/bash
#SBATCH -t 2-00:00:00           # Time limit: 2 days
#SBATCH --mem=128G              # Memory: 128 GB
#SBATCH -J Kang_Training       # Job name
#SBATCH --mail-type=END,FAIL    # Email notifications
#SBATCH --ntasks-per-node=4                    # 1 task
#SBATCH --cpus-per-task=4       # 8 CPU cores per task
#SBATCH -N 1                    # 1 node
#SBATCH --gres=gpu:v100:2
#SBATCH --partition=gpuidle     # Target gpuidle partition
#SBATCH --output=kang_%j.out     # Output file
#SBATCH --error=kang_%j.err      # Error file

# Load required modules
source /home/nmbiedou/miniconda3/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate newcpa

# Verify GPU availability
nvidia-smi

# Run your Python script
srun python  /home/nmbiedou/Documents/cpa/docs/tutorials/Kang.py

# Deactivate Conda environment
conda deactivate
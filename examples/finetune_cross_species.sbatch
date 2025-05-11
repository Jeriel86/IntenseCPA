#!/bin/bash
#SBATCH --mem=80G              # Memory: 128 GB
#SBATCH -J Cross_species_finetune        # Job name
#SBATCH --mail-type=BEGIN,END,FAIL    # Email notifications
#SBATCH -n 1                    # 1 task
#SBATCH --cpus-per-task=4       # 8 CPU cores per task
#SBATCH -N 1                    # 1 node
#SBATCH --gres=gpu:v100:2           # 2 Tesla a100 GPUs
#SBATCH --partition=gpuidle     # Target gpuidle partition
#SBATCH --output=/home/nmbiedou/Documents/cpa/examples/tuning_slurm_logs/slurm_%j.out  # Output log
#SBATCH --error=/home/nmbiedou/Documents/cpa/examples/tuning_slurm_logs/slurm_%j.err   # Error log

# Set logging directory explicitly to /scratch
export LOGGING_DIR="/scratch/nmbiedou/autotune"  # Replace 'nmbiedou' with your actual username
mkdir -p "$LOGGING_DIR"  # Create the directory if it doesn't exist

# Load Conda environment
source /home/nmbiedou/miniconda3/etc/profile.d/conda.sh
conda activate newcpa  # Replace with your actual environment name

# Run the hyperparameter tuning script
echo "Running hyperparameter tuning..."
srun python /home/nmbiedou/Documents/cpa/examples/cross_species_tune_script.py
if [ $? -ne 0 ]; then
    echo "Error: Hyperparameter tuning script failed."
    exit 1
fi

# Deactivate Conda environment
conda deactivate

echo "Job completed successfully."
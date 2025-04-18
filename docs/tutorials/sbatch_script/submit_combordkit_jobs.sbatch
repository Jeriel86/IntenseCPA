#!/bin/bash
#SBATCH --mem=100G              # Memory: 128 GB
#SBATCH -J Combo_Experiment       # Job name
#SBATCH --mail-type=END,FAIL    # Email notifications
#SBATCH -n 1                    # 1 task
#SBATCH --cpus-per-task=4       # 8 CPU cores per task
#SBATCH -N 1                    # 1 node
#SBATCH --gpus=v100:2           # 1 Tesla A30 GPU
#SBATCH --partition=gpuidle     # Target gpuidle partition
#SBATCH --output=/home/nmbiedou/Documents/cpa/slurm_logs/Combo_rdkit/slurm_%A_%a.out  # Output log
#SBATCH --error=/home/nmbiedou/Documents/cpa/slurm_logs/Combo_rdkit/slurm_%A_%a.err   # Error log
#SBATCH --array=0-74            # Array for 65 jobs (5 original + 5*6*2 intense)

# Load modules
source /home/nmbiedou/miniconda3/etc/profile.d/conda.sh
conda activate newcpa  # Replace with your actual environment name

# Experiment parameters
INTENSE_REG_RATES=(0.001 0.0022706495959443195 0.003 0.005 0.01 0.05 0.1)  # 6 rates
INTENSE_P_VALUES=(1 2)  # 2 p values

# Number of runs per configuration
RUNS_PER_CONFIG=5
NUM_ORIGINAL=5  # 5 runs for original CPA
NUM_INTENSE_CONFIGS=$(( ${#INTENSE_REG_RATES[@]} * ${#INTENSE_P_VALUES[@]} ))  # 6 * 2 = 12 intense configs
TOTAL_JOBS=$(( NUM_ORIGINAL + NUM_INTENSE_CONFIGS * RUNS_PER_CONFIG ))  # 5 + 12*5 = 65

# Map array index to experiment parameters
TASK_ID=$SLURM_ARRAY_TASK_ID

if [ $TASK_ID -lt $NUM_ORIGINAL ]; then
    # Original CPA experiments (task_id 0-4)
    python /home/nmbiedou/Documents/cpa/docs/tutorials/run_experiment_combordkit.py --use_intense 0
else
    # Intense CPA experiments (task_id 5-64)
    INTENSE_RUN_IDX=$(( TASK_ID - NUM_ORIGINAL ))  # 0 to 59
    CONFIG_IDX=$(( INTENSE_RUN_IDX / RUNS_PER_CONFIG ))  # 0 to 11 (12 configs)
    REG_RATE_IDX=$(( CONFIG_IDX / ${#INTENSE_P_VALUES[@]} ))  # 0 to 5
    P_IDX=$(( CONFIG_IDX % ${#INTENSE_P_VALUES[@]} ))  # 0 to 1

    INTENSE_REG_RATE=${INTENSE_REG_RATES[$REG_RATE_IDX]}
    INTENSE_P=${INTENSE_P_VALUES[$P_IDX]}

    python /home/nmbiedou/Documents/cpa/docs/tutorials/run_experiment_combordkit.py --use_intense 1 --intense_reg_rate $INTENSE_REG_RATE --intense_p $INTENSE_P
fi

conda deactivate
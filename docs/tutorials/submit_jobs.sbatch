#!/bin/bash
#SBATCH -t 2-00:00:00           # Time limit: 2 days
#SBATCH --mem=128G              # Memory: 128 GB
#SBATCH -J CPA_Experiment       # Job name
#SBATCH --mail-type=END,FAIL    # Email notifications
#SBATCH -n 1                    # 1 task
#SBATCH --cpus-per-task=8       # 8 CPU cores per task
#SBATCH -N 1                    # 1 node
#SBATCH --gpus=a100:1           # 1 Tesla V100 GPU
#SBATCH --partition=gpuidle     # Target gpuidle partition
#SBATCH --output=/home/nmbiedou/Documents/cpa/lightning_logs/slurm_%A_%a.out  # Output log
#SBATCH --error=/home/nmbiedou/Documents/cpa/lightning_logs/slurm_%A_%a.err   # Error log
#SBATCH --array=0-64            # Array for 65 jobs (5 original + 5*6*2 intense)

# Load modules
source /home/nmbiedou/miniconda3/etc/profile.d/conda.sh
conda activate newcpa  # Replace with your actual environment name

# Experiment parameters
SEEDS=(1234 2345 3456 4567 5678)  # 5 seeds
INTENSE_REG_RATES=(0.0 0.001 0.005 0.01 0.05 0.1)  # 6 rates
INTENSE_P_VALUES=(1 2)  # 2 p values

# Calculate total jobs
NUM_ORIGINAL=5  # 5 seeds for original CPA
NUM_INTENSE=$(( ${#SEEDS[@]} * ${#INTENSE_REG_RATES[@]} * ${#INTENSE_P_VALUES[@]} ))  # 5 * 6 * 2 = 60
TOTAL_JOBS=$(( NUM_ORIGINAL + NUM_INTENSE ))  # 65 jobs

# Map array index to experiment parameters
TASK_ID=$SLURM_ARRAY_TASK_ID

if [ $TASK_ID -lt $NUM_ORIGINAL ]; then
    # Original CPA experiments
    SEED=${SEEDS[$TASK_ID]}
    python /home/nmbiedou/Documents/cpa/docs/tutorials/run_experiment.py --seed $SEED --use_intense 0
else
    # Intense CPA experiments
    INTENSE_IDX=$(( TASK_ID - NUM_ORIGINAL ))
    SEED_IDX=$(( INTENSE_IDX / (${#INTENSE_REG_RATES[@]} * ${#INTENSE_P_VALUES[@]}) ))
    REG_RATE_IDX=$(( (INTENSE_IDX % (${#INTENSE_REG_RATES[@]} * ${#INTENSE_P_VALUES[@]})) / ${#INTENSE_P_VALUES[@]} ))
    P_IDX=$(( INTENSE_IDX % ${#INTENSE_P_VALUES[@]} ))

    SEED=${SEEDS[$SEED_IDX]}
    INTENSE_REG_RATE=${INTENSE_REG_RATES[$REG_RATE_IDX]}
    INTENSE_P=${INTENSE_P_VALUES[$P_IDX]}

    python /home/nmbiedou/Documents/cpa/docs/tutorials/run_experiment.py --seed $SEED --use_intense 1 --intense_reg_rate $INTENSE_REG_RATE --intense_p $INTENSE_P
fi

conda deactivate
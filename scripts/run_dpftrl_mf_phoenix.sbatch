#!/bin/bash
#SBATCH --job-name=dpftrl_mf
#SBATCH --account=gts-vzikas3
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dpftrl_mf_%A_%a.out
#SBATCH --error=logs/dpftrl_mf_%A_%a.err
#SBATCH --array=0-3

# MF-DP-FTRL audit on ScatterLinear.

module load anaconda3
conda activate /storage/scratch1/6/aagrawal609/conda_envs/audit

EPSILONS=(1 2 4 8)
EPS=${EPSILONS[$SLURM_ARRAY_TASK_ID]}
DATABASE_SEED=277640617625635548156432291773894477542

echo "Running MF-DP-FTRL audit for epsilon: $EPS"
python scripts/gen_scores_DP_FTRL_scatter.py \
    --mechanism mf \
    --mf-bands 0 \
    --database-seed ${DATABASE_SEED} \
    --target-steps 128 \
    --logical-batch-size 380 \
    --max-physical-batch-size 390 \
    --aug-multiplicity 1 \
    --canary-count 5000 \
    --pkeep 0.5 \
    --epsilon ${EPS} \
    --lr 1.0 \
    --data-dir ./data \
    --log-dir ./logs

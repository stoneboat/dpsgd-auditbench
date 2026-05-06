#!/bin/bash
#SBATCH --job-name=dpftrl_scatter
#SBATCH --account=gts-vzikas3
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dpftrl_scatter_%j.out
#SBATCH --error=logs/dpftrl_scatter_%j.err
#SBATCH --array=0-4

# === Module and env setup ===
module load anaconda3
conda activate /storage/scratch1/6/aagrawal609/conda_envs/dpsgd_bench

# One-time dependency install (kymatio for the scattering transform):
#   pip install kymatio

# === Epsilon values (parity with DP-SGD / DP-FTRL-WRN sweeps) ===
EPSILONS=(1 2 4 8)

if [ $SLURM_ARRAY_TASK_ID -lt 4 ]; then
    # Tasks 0, 1, 2, 3 handle the actual audit sweep
    EPS=${EPSILONS[$SLURM_ARRAY_TASK_ID]}
    echo "Running Audit for Epsilon: $EPS"
    python scripts/gen_scores_DP_FTRL_scatter.py \
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

else
    # Task 4 handles the baseline only
    echo "Running Baseline (Canary Count 0)"
    python scripts/gen_scores_DP_FTRL_scatter.py \
    --target-steps 128 \
    --logical-batch-size 390 \
    --max-physical-batch-size 390 \
    --aug-multiplicity 1 \
    --canary-count 0 \
    --pkeep 0.5 \
    --epsilon 8 \
    --lr 1.0 \
    --data-dir ./data \
    --log-dir ./logs
fi
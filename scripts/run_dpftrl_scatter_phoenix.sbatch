#!/bin/bash
#SBATCH --job-name=dpftrl_scatter
#SBATCH --account=gts-vzikas3
#SBATCH --partition=gpu-h200
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
EPS=${EPSILONS[$SLURM_ARRAY_TASK_ID]}

echo "Running DP-FTRL+ScatterLinear with target epsilon=${EPS} (array task ${SLURM_ARRAY_TASK_ID})"

# Single-pass DP-FTRL on CIFAR-10:
#   N=50000, B=500  ->  T = 100 leaves (one full pass).
# Each canary fires at exactly one leaf; matches Kairouz Theorem 4.1
# single-firing semantics, with per-leaf SNR-optimal ancestor-sum scoring.
python scripts/gen_scores_DP_FTRL_scatter.py \
    --target-steps 100 \
    --logical-batch-size 500 \
    --max-physical-batch-size 500 \
    --aug-multiplicity 1 \
    --canary-count 5000 \
    --pkeep 0.5 \
    --epsilon ${EPS} \
    --lr 1.0 \
    --data-dir ./data \
    --log-dir ./logs

python scripts/gen_scores_DP_FTRL_scatter.py \
    --target-steps 100 \
    --logical-batch-size 500 \
    --max-physical-batch-size 500 \
    --aug-multiplicity 1 \
    --canary-count 0 \
    --pkeep 0.5 \
    --epsilon 8 \
    --lr 1.0 \
    --data-dir ./data \
    --log-dir ./logs
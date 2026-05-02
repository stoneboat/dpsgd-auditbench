#!/bin/bash
#SBATCH --job-name=dpftrl_auditbench
#SBATCH --account=gts-vzikas3
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dpftrl_audit_%j.out
#SBATCH --error=logs/dpftrl_audit_%j.err
#SBATCH --array=0-3

# === Module and env setup ===
module load anaconda3
conda activate /storage/scratch1/6/aagrawal609/conda_envs/dpsgd_bench

# === Epsilon values (parity with DP-SGD sweep) ===
EPSILONS=(1 2 4 8)
EPS=${EPSILONS[$SLURM_ARRAY_TASK_ID]}

echo "Running DP-FTRL with target epsilon=${EPS} (array task ${SLURM_ARRAY_TASK_ID})"

# === Run training (Sander/Mahloujifar recipe: 2500 leaves, K=16, EMA 0.9999) ===
# Note: DP-FTRL has no Poisson sampling -- each leaf is a full mini-batch
# gradient. Privacy comes from the tree-aggregated Gaussian mechanism.
python scripts/gen_scores_DP_FTRL_whitebox.py \
    --target-steps 2500 \
    --ema-decay 0.9999 \
    --ckpt-interval 20 \
    --canary-count 5000 \
    --pkeep 0.5 \
    --epsilon ${EPS} \
    --logical-batch-size 4096 \
    --max-physical-batch-size 128 \
    --aug-multiplicity 16 \
    --lr 4.0 \
    --data-dir ./data \
    --log-dir ./logs

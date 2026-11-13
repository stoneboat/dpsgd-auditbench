#!/bin/bash
#SBATCH --job-name=dpsgd_ablation
#SBATCH --account=gts-vzikas3
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dp_audit_hyperparams_%j.out
#SBATCH --error=logs/dp_audit_hyperparams_%j.err
#SBATCH --array=0-2

# Hyperparameter ablation at fixed eps=8, T=500 (matches run_whitebox_phoenix).
#   tasks 0-1: batch size 1024 / 2048 (4096 baseline run separately).
#              At fixed eps the accountant re-derives sigma from q=B/N, so this
#              is simultaneously the noise-multiplier sweep.
#   task 2:    clipping norm C=10 (C does not enter accounting; it scales signal
#              and noise together, so the audit should be invariant).

# === Module and env setup ===
module load anaconda3
conda activate /storage/scratch1/6/aagrawal609/conda_envs/audit

# === Ablation grid: tag | batch size | clipping norm ===
TAGS=(bs1024      bs2048      clip10)
BATCH=(1024       2048        4096)
CLIP=(1.0         1.0         10.0)

TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
B=${BATCH[$SLURM_ARRAY_TASK_ID]}
C=${CLIP[$SLURM_ARRAY_TASK_ID]}

echo "Ablation task ${SLURM_ARRAY_TASK_ID}: tag=${TAG} batch=${B} clip=${C}"

python scripts/gen_scores_DP_whitebox.py \
    --epochs 300 \
    --target-steps 2500 \
    --ema-decay 0.9999 \
    --ckpt-interval 20 \
    --canary-count 5000 \
    --pkeep 0.5 \
    --epsilon 8 \
    --logical-batch-size ${B} \
    --max-physical-batch-size 128 \
    --aug-multiplicity 16 \
    --max-grad-norm ${C} \
    --lr 4.0 \
    --data-dir ./data \
    --log-dir ./logs \
    --log-file logs/whitebox_eps8_${TAG}_${SLURM_ARRAY_JOB_ID}.log
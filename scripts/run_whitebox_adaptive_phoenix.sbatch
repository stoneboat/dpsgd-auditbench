#!/bin/bash
#SBATCH --job-name=dpsgd_adaptive
#SBATCH --account=gts-vzikas3
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dp_adaptive_%j.out
#SBATCH --error=logs/dp_adaptive_%j.err

# White-box audit under quantile adaptive clipping (Andrew et al. 2021) at eps=8.

module load anaconda3
conda activate /storage/scratch1/6/aagrawal609/conda_envs/audit

DATABASE_SEED=277640617625635548156432291773894477542
TARGET_STEPS=2500        # drop to 500 for a smoke test
EPS=8

echo "=== adaptive clipping, eps=${EPS}, T=${TARGET_STEPS}, seed=${DATABASE_SEED} ==="

python scripts/gen_scores_DP_whitebox.py \
    --adaptive-clipping \
    --target-unclipped-quantile 0.5 \
    --clipbound-lr 0.2 \
    --min-clipbound 0.1 \
    --max-clipbound 50.0 \
    --database-seed ${DATABASE_SEED} \
    --target-steps ${TARGET_STEPS} \
    --ema-decay 0.9999 \
    --ckpt-interval 20 \
    --canary-count 5000 \
    --pkeep 0.5 \
    --epsilon ${EPS} \
    --max-grad-norm 1.0 \
    --logical-batch-size 4096 \
    --max-physical-batch-size 128 \
    --aug-multiplicity 16 \
    --lr 4.0 \
    --data-dir ./data \
    --log-dir ./logs

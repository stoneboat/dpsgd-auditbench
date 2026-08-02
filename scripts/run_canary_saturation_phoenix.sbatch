#!/bin/bash
#SBATCH --job-name=canary_sat
#SBATCH --account=gts-vzikas3
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/canary_sat_%A_%a.out
#SBATCH --error=logs/canary_sat_%A_%a.err
#SBATCH --array=0-1

# Appendix E assumption check on a real DP-SGD run.
#   task 0: input-space canary (constant red, label 6) -- tests A2 (saturation) and A1.
#   task 1: dirac canary baseline -- A2 exact by construction, measures A1 only.
# Same recipe as run_whitebox_phoenix.sbatch so the numbers transfer.

module load anaconda3
conda activate /storage/scratch1/6/aagrawal609/conda_envs/audit

EPS=8
STEPS=500

COMMON="--epsilon ${EPS} --delta 1e-5 --target-steps ${STEPS} \
    --logical-batch-size 4096 --max-physical-batch-size 128 \
    --aug-multiplicity 16 --max-grad-norm 1.0 --lr 4.0 \
    --data-dir ./data --log-dir ./logs"

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo "Input-space canary arm (constant red, label 6), eps=${EPS}, ${STEPS} steps"
    python scripts/canary_saturation_run.py ${COMMON} \
        --canary-color 1,0,0 --canary-label 6 \
        --out ./logs/canary_sat_red6_eps${EPS}.csv
else
    echo "Dirac canary arm (coord 0), eps=${EPS}, ${STEPS} steps"
    python scripts/canary_saturation_run.py ${COMMON} \
        --dirac 0 \
        --out ./logs/canary_sat_dirac_eps${EPS}.csv
fi

#!/bin/bash
#SBATCH -J dpsgd-whitebox           # Job name
#SBATCH -A p-vzikas3                # Account/project
#SBATCH -q inferno                  # Queue
#SBATCH -N 1                        # Number of nodes
#SBATCH --ntasks-per-node=1         # Tasks per node
#SBATCH --gpus-per-node=V100:1      # GPU type and count (adjust: A100, H100, etc.)
#SBATCH --mem=32G                   # Memory
#SBATCH -t 24:00:00                 # Wall time (adjust as needed)
#SBATCH -o logs/slurm-%j.out       # Stdout
#SBATCH -e logs/slurm-%j.err       # Stderr

# === Module and env setup ===
module purge
module load pytorch/25
module load anaconda3

eval "$(conda shell.bash hook)"
conda activate /tmp/python-venv/dpsgd-auditbench_venv

# If env doesn't exist on this node, create it
if [ ! -d "/tmp/python-venv/dpsgd-auditbench_venv" ]; then
    echo "Env not found, running install script..."
    bash scripts/local_scripts/cluster_install_gpu.sh
    eval "$(conda shell.bash hook)"
    conda activate /tmp/python-venv/dpsgd-auditbench_venv
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $SLURM_SUBMIT_DIR

# === Run training ===
python scripts/gen_scores_DP_whitebox.py \
    --epochs 200 \
    --ckpt-interval 20 \
    --canary-count 5000 \
    --pkeep 0.5 \
    --noise-multiplier 3.0 \
    --logical-batch-size 4096 \
    --max-physical-batch-size 128 \
    --aug-multiplicity 1 \
    --lr 4.0 \
    --data-dir ./data \
    --log-dir ./logs
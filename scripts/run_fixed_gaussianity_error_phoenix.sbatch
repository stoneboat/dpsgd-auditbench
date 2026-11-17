#!/bin/bash
#SBATCH --job-name=fixed_gaussianity_error
#SBATCH --account=gts-vzikas3
#SBATCH --partition=cpu-small
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=fig/fixed_gaussianity_error/slurm-%A_%a.out
#SBATCH --error=fig/fixed_gaussianity_error/slurm-%A_%a.err

set -e
cd "$SLURM_SUBMIT_DIR"
module load anaconda3
conda activate "${AUDIT_ENV:-/storage/scratch1/6/aagrawal609/conda_envs/audit}"
set -uo pipefail
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

data_dir="${AUDIT_DATA_DIR:-$PWD/data}"
result_dir="$PWD/fig/fixed_gaussianity_error/${AUDIT_COMPLEXITY_MERGE_JOB_ID:-$SLURM_ARRAY_JOB_ID}"
srun python -u - "$data_dir" "$result_dir" "$SLURM_ARRAY_TASK_ID" "$SLURM_ARRAY_TASK_COUNT" <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from run_auditing_comparison import get_target_T, run_corrected_bootstrap_complexity, run_corrected_bootstrap_report

data_dir, result_dir, task, task_count = sys.argv[1:]

def run_dir(seed):
    return str(Path(data_dir) / f"mislabeled-canaries-{seed}-5000-0.5-cifar10")

sgd8 = run_dir("277640617625635548156432291773894477542")
sharded = os.environ.get("AUDIT_COMPLEXITY_SHARD") == "1"
merge = bool(os.environ.get("AUDIT_COMPLEXITY_MERGE_JOB_ID"))
if task == "2" or sharded or merge:
    output = Path(result_dir) / "canaries"
    output.mkdir(parents=True, exist_ok=True)
    run_corrected_bootstrap_complexity(
        sgd8, delta=1e-5, significance=0.05, fig_dir=output,
        n_trials=int(os.environ.get("AUDIT_COMPLEXITY_TRIALS", "50")), noise_multiplier_floor=2.0,
        shard=(int(task), int(task_count)) if sharded else None, merge_shards=merge,
        resume_csv=os.environ.get("AUDIT_COMPLEXITY_RESUME"),
    )
    sys.exit(0)

output = Path(result_dir) / ("epsilon", "steps", "canaries", "tables")[int(task)]
output.mkdir(parents=True, exist_ok=True)
axis = None
if task == "0":
    noise_floors = {
        run_dir("150826909777158496980686348097484970563"): 15.0,
        run_dir("33983531732666956501283061792610382617"): 8.0,
        run_dir("194358092235300364301779860499483305667"): 4.5,
        sgd8: 2.0,
    }
    axis = "target_epsilon"
elif task == "1":
    step_runs = [
        run_dir("227413209713036813868017938866853468203"),
        run_dir("6361188182970055010550653208276212089"),
        run_dir("60281807569336015233236270129373243223"),
        run_dir("32554249179507031055760171971516307803"),
        sgd8,
    ]
    floors_by_target_steps = {50: 0.75, 100: 0.85, 500: 1.30, 1000: 1.70, 2500: 2.0}
    noise_floors = {exp_dir: floors_by_target_steps[get_target_T(exp_dir)] for exp_dir in step_runs}
    axis = "T"
else:
    batch1024 = os.environ.get("AUDIT_BATCH1024") or run_dir("228508263780865377273727074192580881208")
    batch2048 = os.environ.get("AUDIT_BATCH2048") or run_dir("131088123116516713197276447333128882228")
    clip10 = (
        os.environ.get("AUDIT_CLIP10") or os.environ.get("AUDIT_CLIP10_NORMALIZED")
        or run_dir("314474712029583471132696488640968510043")
    )
    noise_floors = {
        sgd8: 2.0,
        f"{sgd8}-fault_clip_mismatch": 1.25,
        f"{sgd8}-fault_sigma_stale": 1.90,
        f"{sgd8}-fault_q_mismatch": 2.0,
        f"{sgd8}-adaptive": 2.0,
        f"{sgd8}-lr2.0": 2.0,
        f"{sgd8}-lr8.0": 2.0,
        f"{sgd8}-lr16.0": 2.0,
        batch1024: 0.90,
        batch2048: 1.40,
        clip10: 2.0,
    }

noise_floors = {str(Path(exp_dir).resolve()): floor for exp_dir, floor in noise_floors.items()}
success = run_corrected_bootstrap_report(
    noise_floors, delta=1e-5, significance=0.05, fig_dir=output, axis=axis, noise_floors=noise_floors,
)
sys.exit(0 if success else 1)
PY

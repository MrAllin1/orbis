#!/bin/bash
#SBATCH --job-name=fvd_calc_balanced_seed42
#SBATCH --partition=lmbhiwi_gpu-rtx2080
#SBATCH --account=lmbhiwi-dlc
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail

echo "=== FVD job start ==="
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"

# Activate conda
source /work/dlclarge2/alidemaa-text-control-orbis/miniconda3/etc/profile.d/conda.sh
conda activate orbis_env

echo "Python: $(which python)"
python - <<'EOF'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
EOF

nvidia-smi || true

# IMPORTANT: cd to repo root so imports work consistently
cd /work/dlclarge2/alidemaa-text-control-orbis/orbis

COMMON_ARGS="--num_frames 16 --height 192 --width 336 --max_videos 20 --batch_size 4"

echo "=== LEFT FVD ==="
python FVD/tools/compute_fvd_from_folders.py \
  --real_dir /work/dlclarge2/alidemaa-text-control-orbis/orbis/FVD/real_by_class/left \
  --fake_dir /work/dlclarge2/alidemaa-text-control-orbis/orbis/FVD/generated_data/rollout_runs/turning_left \
  ${COMMON_ARGS}

echo "=== RIGHT FVD ==="
python FVD/tools/compute_fvd_from_folders.py \
  --real_dir /work/dlclarge2/alidemaa-text-control-orbis/orbis/FVD/real_by_class/right \
  --fake_dir /work/dlclarge2/alidemaa-text-control-orbis/orbis/FVD/generated_data/rollout_runs/turning_right \
  ${COMMON_ARGS}

echo "=== STRAIGHT FVD ==="
python FVD/tools/compute_fvd_from_folders.py \
  --real_dir /work/dlclarge2/alidemaa-text-control-orbis/orbis/FVD/real_by_class/straight \
  --fake_dir /work/dlclarge2/alidemaa-text-control-orbis/orbis/FVD/generated_data/rollout_runs/training_caption \
  ${COMMON_ARGS}


echo "=== LEFT RIGHT FVD ==="
python FVD/tools/compute_fvd_from_folders.py \
  --real_dir /work/dlclarge2/alidemaa-text-control-orbis/orbis/FVD/real_by_class/left \
  --fake_dir /work/dlclarge2/alidemaa-text-control-orbis/orbis/FVD/generated_data/rollout_runs/turning_left_right \
  ${COMMON_ARGS}


echo "=== RIGHT LEFT FVD ==="
python FVD/tools/compute_fvd_from_folders.py \
  --real_dir /work/dlclarge2/alidemaa-text-control-orbis/orbis/FVD/real_by_class/right \
  --fake_dir /work/dlclarge2/alidemaa-text-control-orbis/orbis/FVD/generated_data/rollout_runs/turning_right_left \
  ${COMMON_ARGS}

echo "=== FVD job finished ==="
echo "Date: $(date)"

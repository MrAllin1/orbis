#!/bin/bash
#SBATCH --job-name=orbis_adaln
#SBATCH --partition=lmbhiwi_gpu-rtx2080
#SBATCH --account=lmbhiwi-dlc
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# -----------------------------------------------------
echo "=== Starting SLURM Job on lmbhiwi_gpu-rtx2080 ==="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
export TK_WORK_DIR=/work/dlclarge2/alidemaa-text-control-orbis/orbis/logs_tk/tokenizer_192x336
# -----------------------------------------------------
# ACTIVATE your miniconda installation in HOME
# -----------------------------------------------------
source /work/dlclarge2/alidemaa-text-control-orbis/miniconda3/etc/profile.d/conda.sh
conda activate orbis_env

echo "Python = $(which python)"
echo "Torch CUDA available?:"
python - <<EOF
import torch
print(torch.cuda.is_available())
EOF

nvidia-smi || true

# -----------------------------------------------------
# Run your script
# -----------------------------------------------------
python -m finetuning.train_adaln_debug

echo "=== Job Finished ==="

#!/bin/bash
#SBATCH --job-name=orbis_adaln
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

echo "=== Starting SLURM Job on dev_gpu_h100 ==="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
# -----------------------------------------------------
# ACTIVATE your miniconda installation in HOME
# -----------------------------------------------------
source /home/fr/fr_fr/fr_aa533/miniconda3/bin/activate
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
python -m finetuning.train_text_finetune

echo "=== Job Finished ==="

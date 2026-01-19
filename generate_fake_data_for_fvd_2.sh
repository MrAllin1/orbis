#!/bin/bash
#SBATCH --job-name=fvd_fake_compact_dataset
#SBATCH --partition=lmbhiwi_gpu-rtx2080
#SBATCH --account=lmbhiwi-dlc
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail

echo "=== Starting SLURM Job ==="
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"

export TK_WORK_DIR=/work/dlclarge2/alidemaa-text-control-orbis/orbis/logs_tk/tokenizer_192x336

source /work/dlclarge2/alidemaa-text-control-orbis/miniconda3/etc/profile.d/conda.sh
conda activate orbis_env

echo "Python: $(which python)"
python - <<'EOF'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
EOF

nvidia-smi || true

# IMPORTANT: cd to repo root so `finetuning` is importable
cd /work/dlclarge2/alidemaa-text-control-orbis/orbis

python -m FVD.tools.inference_to_generate_fake_clips_balanced_dataset \
  --video_source /work/dlclarge2/alidemaa-text-control-orbis/orbis/data/covla_videos_balanced_val \
  --num_videos 20 \
  --unique_videos \
  --seed 42 \
  --desired_seconds 20 \
  --num_context_frames 5 \
  --stride 10 \
  --base_out_dir /work/dlclarge2/alidemaa-text-control-orbis/orbis/FVD/compact_dataset/generated_data/rollout_runs \
  --collect_root /work/dlclarge2/alidemaa-text-control-orbis/orbis/FVD/compact_dataset/generated_data/fvd_fake_by_prompt \
  --job_tag compact_dataset_seed42 \
  --use_finetuned_weights \
  --save_path /work/dlclarge2/alidemaa-text-control-orbis/orbis/finetuning/compact_train.ckpt

echo "=== Job Finished ==="
echo "Date: $(date)"

#!/bin/bash
#SBATCH --job-name=orbis_rollout
#SBATCH --output=rollout_%j.out
#SBATCH --error=rollout_%j.err
#SBATCH --partition=gpu_a100_il         # ✅ correct partition name
#SBATCH --gres=gpu:1                    # request 1 GPU
#SBATCH --cpus-per-task=8               # good for PyTorch dataloading
#SBATCH --mem=64G                       # adjust if needed
#SBATCH --time=00:10:00                 # adjust to expected runtime

# ---- Load modules (if required)
module purge
module load cuda/12.1                   # ✅ IMPORTANT on HPC
# (If they have a pytorch module, you can load it)
export TK_WORK_DIR="/home/fr/fr_fr/fr_aa533/work/orbis/logs_tk"
export WM_WORK_DIR="/home/fr/fr_fr/fr_aa533/work/orbis/logs_wm"
# ---- Activate conda environment
source /home/fr/fr_fr/fr_aa533/miniconda3/etc/profile.d/conda.sh
conda activate orbis_env

# ---- Go to the project directory
cd /pfs/data6/home/fr/fr_fr/fr_aa533/work/orbis

# ---- Run rollout
python evaluate/rollout.py \
  --exp_dir logs_wm/orbis_288x512 \
  --num_gen_frames 120 \
  --num_steps 30

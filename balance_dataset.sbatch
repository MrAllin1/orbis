#!/bin/bash
#SBATCH --job-name=balancing_dataset
#SBATCH --partition=lmbhiwi_gpu-rtx2080
#SBATCH --account=lmbhiwi-dlc
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# (If they have a pytorch module, you can load it)
export TK_WORK_DIR="/work/dlclarge2/alidemaa-text-control-orbis/orbis/logs_tk"
export WM_WORK_DIR="/work/dlclarge2/alidemaa-text-control-orbis/orbis/logs_wm"

source /work/dlclarge2/alidemaa-text-control-orbis/miniconda3/etc/profile.d/conda.sh
conda activate orbis_env

# ================= RUN =================
python /work/dlclarge2/alidemaa-text-control-orbis/orbis/data/preprocessing/balance_covla_dataset.py
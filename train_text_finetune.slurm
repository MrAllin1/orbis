#!/bin/bash
#SBATCH --job-name=orbis_text_ft        # Job name  
#SBATCH --output=text_ft_%j.out         # STDOUT log
#SBATCH --error=text_ft_%j.err          # STDERR log
#SBATCH --partition=gpu_a100_il         # GPU partition
#SBATCH --gres=gpu:1                    # 1x A100 GPU
#SBATCH --cpus-per-task=8               # Good for DataLoader workers
#SBATCH --mem=64G                       # Memory
#SBATCH --time=00:10:00                 # Adjust based on training length

### --- Load modules required for GPU acceleration ---
module purge
module load cuda/12.1

### --- Activate conda environment ---
source /home/fr/fr_fr/fr_aa533/miniconda3/etc/profile.d/conda.sh
conda activate orbis_env

### --- Move into project directory ---
cd /home/fr/fr_fr/fr_aa533/work/orbis

### --- RUN TRAINING ---
python finetuning/train_text_finetune.py

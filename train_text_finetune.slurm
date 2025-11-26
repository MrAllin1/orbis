#!/bin/bash
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=10:00:00

### --- Load modules required for GPU acceleration ---
module purge
module load cuda/12.1

### --- Activate conda environment ---
source /home/fr/fr_fr/fr_aa533/miniconda3/etc/profile.d/conda.sh
conda activate orbis_env

### --- Move into project directory ---
cd /home/fr/fr_fr/fr_aa533/work/orbis

### --- RUN TRAINING ---
python -m finetuning.train_text_finetune

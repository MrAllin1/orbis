#!/bin/bash
#SBATCH --job-name=covla20
#SBATCH --output=covla20.out
#SBATCH --error=covla20.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "=== CoVLA Tutorial Downloader (20 samples) ==="
echo "Node: $(hostname)"
echo "Start: $(date)"

TARGET_DIR="/home/fr/fr_fr/fr_aa533/work/orbis/covla_20_samples"
mkdir -p "$TARGET_DIR"

# ------------------------------
# Activate Conda
# ------------------------------
source ~/.bashrc
conda activate orbis_env

echo "Using Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Prefer decord backend (even if the env var is ignored, it's harmless)
export HF_DATASETS_VIDEO_DECODER=decord
echo "HF_DATASETS_VIDEO_DECODER=$HF_DATASETS_VIDEO_DECODER"

# ------------------------------
# Python block
# ------------------------------
python3 << EOF
import os
import sys
import cv2

print("[INFO] Python version:", sys.version)

try:
    import datasets
    from datasets import load_dataset
    print("[INFO] datasets version:", datasets.__version__)
except Exception as e:
    print("[FATAL] Could not import 'datasets':", e)
    sys.exit(1)

try:
    import decord
    print("[INFO] decord version:", decord.__version__)
except Exception as e:
    print("[WARN] Could not import decord:", e)

OUT = "/home/fr/fr_fr/fr_aa533/work/orbis/covla_20_samples"
os.makedirs(OUT, exist_ok=True)

print("[1/2] Loading CoVLA dataset in STREAMING mode…")
dataset = load_dataset(
    "turing-motors/CoVLA-Dataset",
    split="train",
    streaming=True,
)

print("[2/2] Downloading exactly 20 videos using the tutorial logic…")

num_to_download = 20
count = 0

for scene_idx, scene in enumerate(dataset):
    if count >= num_to_download:
        break

    video = scene["video"]      # decord.VideoReader (on your Mac) / other backend here
    video_id = scene["video_id"]

    output_file = os.path.join(OUT, f"{count:02d}_{video_id}.mp4")
    print(f"[INFO] [{count+1}/{num_to_download}] Saving video_id={video_id} -> {output_file}")

    # --- Tutorial-style re-encoding to MP4 ---
    # Get first frame to determine resolution
    frame0 = video[0].asnumpy()
    height, width, _ = frame0.shape
    fps = 20

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for frame_idx in range(len(video)):
        frame = video[frame_idx].asnumpy()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    print(f"[OK] Saved: {output_file}")
    count += 1

print(f"\n[DONE] Saved {count} videos to {OUT}")
EOF

echo "Finished: $(date)"

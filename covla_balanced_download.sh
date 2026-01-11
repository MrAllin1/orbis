#!/bin/bash
#SBATCH --job-name=covla_balanced_download
#SBATCH --output=covla_balanced_download.out
#SBATCH --error=covla_balanced_download.err
#SBATCH --time=24:00:00
#SBATCH --partition=lmbhiwi_gpu-rtx2080
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

echo "=== CoVLA Balanced Videos Downloader ==="
echo "Node: $(hostname)"
echo "Start: $(date)"

# ------------------------------
# Paths
# ------------------------------
CAPTIONS_DIR="/work/dlclarge2/alidemaa-text-control-orbis/orbis/data/covla_captions_balanced"
OUT_DIR="/work/dlclarge2/alidemaa-text-control-orbis/orbis/data/covla_videos_balanced"

mkdir -p "$OUT_DIR"

# ------------------------------
# Activate Conda
# ------------------------------
source ~/.bashrc
conda activate orbis_env

echo "Using Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Prefer decord backend (harmless if ignored)
export HF_DATASETS_VIDEO_DECODER=decord
echo "HF_DATASETS_VIDEO_DECODER=$HF_DATASETS_VIDEO_DECODER"

# Optional chunking (recommended for SLURM arrays)
# - If you submit as an array job, each task downloads a slice of IDs.
# - Example:
#   sbatch --array=0-20 covla_balanced_download.sh
#   (with CHUNK_SIZE=200 => ~21 tasks for ~4120 videos)
export CHUNK_SIZE="${CHUNK_SIZE:-200}"

python3 << 'EOF'
import os
import sys
import glob
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

CAPTIONS_DIR = os.environ.get("CAPTIONS_DIR", "/work/dlclarge2/alidemaa-text-control-orbis/orbis/data/covla_captions_balanced")
OUT_DIR = os.environ.get("OUT_DIR", "/work/dlclarge2/alidemaa-text-control-orbis/orbis/data/covla_videos_balanced")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "200"))

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------
# Collect balanced IDs from filenames
# ------------------------------
jsonl_files = sorted(glob.glob(os.path.join(CAPTIONS_DIR, "*.jsonl")))
if not jsonl_files:
    print(f"[FATAL] No .jsonl files found in: {CAPTIONS_DIR}")
    sys.exit(1)

all_ids = [os.path.splitext(os.path.basename(p))[0] for p in jsonl_files]
total_ids = len(all_ids)
print(f"[INFO] Found {total_ids} balanced IDs from captions dir.")

# ------------------------------
# Optional SLURM array slicing
# ------------------------------
slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
if slurm_task_id is not None:
    tid = int(slurm_task_id)
    start = tid * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, total_ids)
    if start >= total_ids:
        print(f"[DONE] Task {tid}: start={start} >= total_ids={total_ids}. Nothing to do.")
        sys.exit(0)
    target_ids = all_ids[start:end]
    print(f"[INFO] SLURM_ARRAY_TASK_ID={tid} => downloading slice [{start}:{end}) ({len(target_ids)} IDs)")
else:
    target_ids = all_ids
    print("[INFO] No SLURM_ARRAY_TASK_ID set => downloading ALL balanced videos in one job.")

target_set = set(target_ids)

# Resume: remove IDs whose videos already exist
# (assumes output filename is {video_id}.mp4)
existing = 0
for vid in list(target_set):
    out_path = os.path.join(OUT_DIR, f"{vid}.mp4")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        target_set.remove(vid)
        existing += 1

print(f"[INFO] Resume check: {existing} already present, {len(target_set)} remaining to download.")

if not target_set:
    print("[DONE] Nothing to download (all targets already exist).")
    sys.exit(0)

# ------------------------------
# Stream dataset and download matches
# ------------------------------
print("[INFO] Loading CoVLA dataset in STREAMING mode…")
dataset = load_dataset(
    "turing-motors/CoVLA-Dataset",
    split="train",
    streaming=True,
)

def save_video_tutorial_style(video, out_file: str, fps: int = 20):
    # Determine resolution from first frame
    frame0 = video[0].asnumpy()
    height, width, _ = frame0.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open: {out_file}")

    try:
        for i in range(len(video)):
            frame = video[i].asnumpy()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()

found = 0
seen = 0
remaining = len(target_set)
print(f"[INFO] Starting streaming scan. Need to fetch {remaining} videos...")

for scene_idx, scene in enumerate(dataset):
    seen += 1

    video_id = scene.get("video_id")
    if video_id not in target_set:
        continue

    out_file = os.path.join(OUT_DIR, f"{video_id}.mp4")
    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        # Another process/job might have created it.
        target_set.remove(video_id)
        remaining -= 1
        continue

    print(f"[DL] ({found+1}) video_id={video_id} -> {out_file}")

    try:
        video = scene["video"]
        save_video_tutorial_style(video, out_file, fps=20)
        print(f"[OK] Saved: {out_file}")
        found += 1
        target_set.remove(video_id)
        remaining -= 1
    except Exception as e:
        # Keep it in the set so a retry run can attempt again
        print(f"[ERR] Failed video_id={video_id}: {e}")

    if remaining == 0:
        print("[DONE] All target videos downloaded.")
        break

print("\n=== Summary ===")
print(f"[INFO] Streamed scenes seen: {seen}")
print(f"[INFO] Videos successfully saved this run: {found}")
print(f"[INFO] Still missing after run: {len(target_set)}")

if target_set:
    missing_path = os.path.join(OUT_DIR, "missing_video_ids.txt")
    with open(missing_path, "w") as f:
        for vid in sorted(target_set):
            f.write(vid + "\n")
    print(f"[WARN] Wrote missing IDs to: {missing_path}")
EOF

echo "Finished: $(date)"

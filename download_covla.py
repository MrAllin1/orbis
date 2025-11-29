#!/usr/bin/env python

import os
import shutil
from huggingface_hub import list_repo_files, hf_hub_download

# ===== SETTINGS =====
N_NEW = 80  # how many *additional* videos you want
OUT_DIR = "covla_20_videos"   # same folder as before
REPO_ID = "turing-motors/CoVLA-Dataset"
# ====================

os.makedirs(OUT_DIR, exist_ok=True)

print(f"[INFO] Output directory: {OUT_DIR}")
print("[1/4] Checking already downloaded videos...")

# Existing local .mp4 filenames (just basenames, e.g. '0000b7dc6478371b.mp4')
existing_local = {
    f for f in os.listdir(OUT_DIR)
    if f.lower().endswith(".mp4")
}
print(f"[INFO] Found {len(existing_local)} existing local videos.")

print("[2/4] Listing files in HF dataset repo (no downloads yet)...")

# Get all files in the dataset repo
all_files = list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset",
)

# Filter only mp4 video files under "videos/"
video_files = sorted(
    f for f in all_files
    if f.startswith("videos/") and f.endswith(".mp4")
)

print(f"[INFO] Found {len(video_files)} video files in repo.")
if len(video_files) == 0:
    raise RuntimeError("No .mp4 files found under 'videos/' in the dataset repo.")

# Filter out files whose basename is already in OUT_DIR
unique_candidates = [
    f for f in video_files
    if os.path.basename(f) not in existing_local
]

print(f"[INFO] Unique candidates (not yet downloaded): {len(unique_candidates)}")

if len(unique_candidates) == 0:
    raise RuntimeError("No new videos available to download (all already present locally?).")

# Take only the first N_NEW unique ones
selected_files = unique_candidates[:N_NEW]
print(f"[INFO] Will download {len(selected_files)} new files (requested {N_NEW}).")

print("[3/4] Downloading files via hf_hub_download...")

for idx, rel_path in enumerate(selected_files, start=1):
    print(f"[INFO] ({idx}/{len(selected_files)}) file: {rel_path}")

    # Download this single file into HF cache
    local_fp = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=rel_path,
    )

    # Copy to OUT_DIR with original filename
    dst_name = os.path.basename(rel_path)
    dst_path = os.path.join(OUT_DIR, dst_name)
    shutil.copy2(local_fp, dst_path)
    print(f"[OK] Saved to {dst_path}")

print(f"[4/4] Done. Downloaded {len(selected_files)} *new* video files into {OUT_DIR}.")
print(f"[INFO] Total local videos now: {len(os.listdir(OUT_DIR))}")

import os
import json
import time
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from decord import VideoReader, cpu 


class CoVLAOrbisMultiFrame(Dataset):
    """
    Local CoVLA dataset compatible with Orbis pipeline (simplified):

    - No random crops / center crops.
    - Every frame is simply resized to (H, W) and normalized to [-1, 1].

    Args:
        num_frames: number of frames per sample (context + target)
        stored_data_frame_rate: fps of stored videos (e.g. 20)
        target_frame_rate: logical fps for the model (e.g. 5)
        size: (H, W) final resolution of frames (default: 192x336 to match tokenizer/FM)
        captions_dir: directory containing <video_id>.jsonl caption files
        videos_dir: directory containing .mp4 videos
        num_samples: optional cap on number of videos used
        debug: if True, prints debug info
        aug: kept only for logging (no behavior differences)
        scale_min/scale_max: unused (kept for compatibility)
    """

    def __init__(
        self,
        num_frames: int = 6,
        stored_data_frame_rate: int = 20,
        target_frame_rate: int = 5,
        size=(192, 336),
        captions_dir: str | None = None,
        videos_dir: str = "data/covla_100_videos",
        num_samples: int | None = None,
        debug: bool = False,
        aug: str = "resize",   # kept for compatibility / logging only
        scale_min: float = 0.75,  # unused
        scale_max: float = 1.0,   # unused
    ):
        t0 = time.time() if debug else None

        # ----------- basic settings -----------
        self.num_frames = num_frames
        self.stored_rate = stored_data_frame_rate
        self.target_rate = target_frame_rate
        self.frame_interval = max(1, round(self.stored_rate / self.target_rate))

        # normalize size → (H, W)
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = (int(size[0]), int(size[1]))  # (H, W)

        self.captions_dir = captions_dir
        self.videos_dir = videos_dir
        self.debug = debug
        self.aug = aug  # only for logging

        # ----------- transforms (NO cropping, only resize) -----------
        # Deterministic and identical for train/val.
        self.base_transform = transforms.Compose(
            [
                transforms.Resize(self.size),   # directly to (H, W) = (192,336)
                transforms.ToTensor(),
            ]
        )

        # ----------- collect video_ids -----------
        if debug:
            print("\n================= [INIT] CoVLAOrbisMultiFrame (LOCAL) =================")
            print(f"[INIT] Scanning videos in: {videos_dir}")

        all_files = [f for f in os.listdir(videos_dir) if f.endswith(".mp4")]
        all_files = sorted(all_files)
        self.video_ids = [os.path.splitext(f)[0] for f in all_files]

        if len(self.video_ids) == 0:
            raise RuntimeError(f"No .mp4 files found in {videos_dir}")

        # limit samples if requested
        if num_samples is None:
            self.num_samples = len(self.video_ids)
        else:
            self.num_samples = min(num_samples, len(self.video_ids))

        if debug:
            print(f"[INIT] Found {len(self.video_ids)} videos.")
            print(f"[INIT] Using num_samples = {self.num_samples}")
            print(f"[INIT] num_frames={num_frames}, frame_interval={self.frame_interval}")
            print(f"[INIT] final size = {self.size} (H, W)")
            print(f"[INIT] aug (ignored for cropping) = {self.aug}")
            print(f"[INIT] Completed in {time.time() - t0:.2f} seconds")
            print("================================================================\n")

    # ------------------------------------------------------------------
    def load_captions(self, video_id: str) -> dict[int, str]:
        """Load captions from <video_id>.jsonl if available."""
        if not self.captions_dir:
            return {}

        path = os.path.join(self.captions_dir, f"{video_id}.jsonl")
        if not os.path.exists(path):
            return {}

        caps: dict[int, str] = {}
        with open(path, "r") as f:
            for line in f:
                entry = json.loads(line)
                frame_idx_str = list(entry.keys())[0]
                frame_idx = int(frame_idx_str)
                caps[frame_idx] = entry[frame_idx_str].get("plain_caption", "")
        return caps

    # ------------------------------------------------------------------
    def _sample_frame_indices(self, total_frames: int) -> list[int]:
        """
        Sample a contiguous subsequence of `num_frames` with step `frame_interval`,
        like the original multi-frame setup.
        """
        frames_needed = self.num_frames * self.frame_interval
        if total_frames <= 0:
            raise RuntimeError("Video has zero frames")

        max_start = max(0, total_frames - frames_needed)
        start = random.randint(0, max_start) if max_start > 0 else 0

        raw_idxs = [start + i * self.frame_interval for i in range(self.num_frames)]
        idxs = [min(i, total_frames - 1) for i in raw_idxs]
        return idxs

    # ------------------------------------------------------------------
    def _apply_transforms_to_frames(self, pil_frames: list[Image.Image]) -> torch.Tensor:
        """
        Just resize each frame to (H, W), stack, and normalize to [-1, 1].
        No random cropping, no center cropping.
        """
        transformed = [self.base_transform(img) for img in pil_frames]
        frames = torch.stack(transformed, dim=0)  # (F, C, H, W)
        frames = frames * 2 - 1  # [0,1] → [-1,1]
        return frames

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.num_samples}")

        video_id = self.video_ids[idx]
        video_path = os.path.join(self.videos_dir, f"{video_id}.mp4")

        if self.debug:
            print(f"\n================= [GETITEM] idx={idx} =================")
            print(f"[GETITEM] video_id={video_id}")
            print(f"[GETITEM] video_path={video_path}")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # ---- open video with decord ----
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        if self.debug:
            print(f"[GETITEM] total_frames={total_frames}")

        # ---- captions ----
        captions = self.load_captions(video_id)

        # ---- frame indices (random subsequence) ----
        idxs = self._sample_frame_indices(total_frames)
        if self.debug:
            print(f"[GETITEM] Sampled indices: {idxs}")

        pil_frames: list[Image.Image] = []
        texts: list[str] = []

        for i in idxs:
            frame = vr[i].asnumpy()
            img = Image.fromarray(frame).convert("RGB")
            pil_frames.append(img)
            texts.append(captions.get(i, ""))

        # apply resize + stack
        frames = self._apply_transforms_to_frames(pil_frames)

        # choose global caption (like Orbis often does)
        global_caption = texts[0] if len(texts) > 0 and texts[0] != "" else ""
        if global_caption == "":
            global_caption = "no caption"

        if self.debug:
            print(f"[GETITEM] global_caption='{global_caption[:50]}…'")
            print(f"[GETITEM] frames.shape={tuple(frames.shape)} (F,C,H,W)")
            print("[GETITEM] Returning sample.\n")

        return {
            "images": frames,             # (F, 3, H, W) in [-1, 1]
            "caption": global_caption,
            "video_id": video_id,
            "frame_rate": self.target_rate,  # scalar, used to build [B]-tensor in training
        }

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.num_samples

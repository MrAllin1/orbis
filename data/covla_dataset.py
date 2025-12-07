import os
import json
import time
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from decord import VideoReader, cpu  # make sure `pip install decord` is done


class RandomResizedCenterCropRect:
    """
    Rectangular version of RandomResizedCenterCrop (like in Orbis),
    but with params shared across all frames in a sequence.

    - size: (H, W) or int (then square)
    - scale: (min_area_frac, max_area_frac)
    """

    def __init__(self, size, scale=(0.5, 1.0), interpolation=Image.BILINEAR):
        if isinstance(size, (list, tuple)):
            self.crop_h, self.crop_w = int(size[0]), int(size[1])
        else:
            self.crop_h = self.crop_w = int(size)
        self.scale = scale
        self.interpolation = interpolation

        # cached parameters so all frames in a clip get the same crop
        self.fixed_params = None

    def _get_params(self, img: Image.Image):
        if self.fixed_params is not None:
            return self.fixed_params

        width, height = img.size
        area = height * width
        aspect_ratio = width / height

        target_area = random.uniform(*self.scale) * area

        new_width = int(round((target_area * aspect_ratio) ** 0.5))
        new_height = int(round((target_area / aspect_ratio) ** 0.5))

        # ensure crop fits
        if new_width < self.crop_w or new_height < self.crop_h:
            scale = max(self.crop_w / max(new_width, 1), self.crop_h / max(new_height, 1))
            new_width = int(new_width * scale)
            new_height = int(new_height * scale)

        img_w, img_h = width, height
        new_width = min(new_width, img_w)
        new_height = min(new_height, img_h)

        # resize around center, then center-crop to (crop_h, crop_w)
        x1 = (new_width - self.crop_w) // 2
        y1 = (new_height - self.crop_h) // 2
        x1 = max(0, x1)
        y1 = max(0, y1)

        self.fixed_params = (new_width, new_height, x1, y1)
        return self.fixed_params

    def reset(self):
        """Reset cached crop parameters for a new video clip."""
        self.fixed_params = None

    def __call__(self, img: Image.Image):
        new_width, new_height, x1, y1 = self._get_params(img)
        img = img.resize((new_width, new_height), self.interpolation)
        x2 = x1 + self.crop_w
        y2 = y1 + self.crop_h
        return img.crop((x1, y1, x2, y2))


class CoVLAOrbisMultiFrame(Dataset):
    """
    Local CoVLA dataset compatible with Orbis pipeline:
    - num_frames: number of frames per sample (context + target)
    - stored_data_frame_rate: fps of stored videos (e.g. 20)
    - target_frame_rate: logical fps for the model (e.g. 5)
    - size: (H, W) final resolution
    - Returns:
        {
          "images": (F, 3, H, W) in [-1, 1],
          "caption": str,
          "video_id": str,
        }
    """

    def __init__(
        self,
        num_frames=6,
        stored_data_frame_rate=20,
        target_frame_rate=5,
        size=(288, 512),
        captions_dir=None,
        videos_dir="data/covla_100_videos",
        num_samples=None,
        debug=False,
        aug="random_resize_center",
        scale_min=0.75,
        scale_max=1.0,
    ):
        t0 = time.time() if debug else None

        # ----------- basic settings -----------
        self.num_frames = num_frames
        self.stored_rate = stored_data_frame_rate
        self.target_rate = target_frame_rate
        self.frame_interval = max(1, round(self.stored_rate / self.target_rate))

        # normalize size → (H, W)
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = (int(size[0]), int(size[1]))

        self.captions_dir = captions_dir
        self.videos_dir = videos_dir
        self.debug = debug
        self.aug = aug

        # ----------- transforms (match Orbis style) -----------
        if aug == "random_resize_center":
            # like RandomResizedCenterCrop in Orbis
            self.custom_crop = RandomResizedCenterCropRect(
                size=self.size, scale=(scale_min, scale_max)
            )
            self.base_transform = transforms.Compose(
                [
                    self.custom_crop,
                    transforms.ToTensor(),
                ]
            )
        elif aug == "resize_center":
            # deterministic resize + center crop (validation-like)
            self.custom_crop = None
            self.base_transform = transforms.Compose(
                [
                    transforms.Resize(min(self.size)),
                    transforms.CenterCrop(self.size),
                    transforms.ToTensor(),
                ]
            )
        else:
            raise ValueError(f"Unknown augmentation type: {aug}")

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
            print(f"[INIT] aug = {self.aug}")
            print(f"[INIT] Completed in {time.time() - t0:.2f} seconds")
            print("================================================================\n")

    # ------------------------------------------------------------------
    def load_captions(self, video_id):
        """Load captions from <video_id>.jsonl if available."""
        if not self.captions_dir:
            return {}

        path = os.path.join(self.captions_dir, f"{video_id}.jsonl")
        if not os.path.exists(path):
            return {}

        caps = {}
        with open(path, "r") as f:
            for line in f:
                entry = json.loads(line)
                frame_idx_str = list(entry.keys())[0]
                frame_idx = int(frame_idx_str)
                caps[frame_idx] = entry[frame_idx_str].get("plain_caption", "")
        return caps

    # ------------------------------------------------------------------
    def _sample_frame_indices(self, total_frames: int):
        """
        Sample a contiguous subsequence of `num_frames` with step `frame_interval`,
        like the HDF5 multi-frame datasets.
        """
        frames_needed = self.num_frames * self.frame_interval
        if total_frames <= 0:
            raise RuntimeError("Video has zero frames")

        # maximum starting frame that still allows enough frames
        max_start = max(0, total_frames - frames_needed)
        start = random.randint(0, max_start) if max_start > 0 else 0

        raw_idxs = [start + i * self.frame_interval for i in range(self.num_frames)]
        idxs = [min(i, total_frames - 1) for i in raw_idxs]
        return idxs

    # ------------------------------------------------------------------
    def _apply_transforms_to_frames(self, pil_frames):
        """
        Apply same random crop to all frames (for random_resize_center),
        then stack to a tensor and normalize to [-1, 1].
        """
        if self.aug == "random_resize_center" and self.custom_crop is not None:
            # reset crop params for this clip
            self.custom_crop.reset()

        transformed = [self.base_transform(img) for img in pil_frames]
        frames = torch.stack(transformed, dim=0)  # (F, C, H, W)
        frames = frames * 2 - 1  # [0,1] → [-1,1]
        return frames

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
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

        pil_frames = []
        texts = []

        for i in idxs:
            frame = vr[i].asnumpy()
            img = Image.fromarray(frame).convert("RGB")
            pil_frames.append(img)
            texts.append(captions.get(i, ""))

        # apply shared crop + stack
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
            "images": frames,       # (num_frames, 3, H, W) in [-1, 1]
            "caption": global_caption,
            "video_id": video_id,
        }

    # ------------------------------------------------------------------
    def __len__(self):
        return self.num_samples

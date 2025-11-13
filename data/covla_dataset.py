import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image

from torchvision import transforms
import random
import matplotlib.pyplot as plt


class CoVLAOrbisMultiFrame(Dataset):
    """
    CoVLA → Orbis-compatible multi-frame dataset.
    Reuses EXACT transforms from Orbis (Resize + CenterCrop + ToTensor).
    """

    def __init__(
        self,
        split="train",
        num_frames=8,
        stored_data_frame_rate=10,
        target_frame_rate=5,
        size=288,
        streaming=True,
        num_samples=None,
        captions_dir='data/covla_captions',
    ):
        self.dataset = load_dataset(
            "turing-motors/CoVLA-Dataset",
            split=split,
            streaming=streaming,
        )

        if not streaming:
            self.dataset = list(self.dataset)
            if num_samples:
                self.dataset = self.dataset[:num_samples]

        self.num_frames = num_frames
        self.stored_rate = stored_data_frame_rate
        self.target_rate = target_frame_rate

        # EXACT frame skipping logic reused from Orbis MultiHDF5DatasetMultiFrame
        self.frame_interval = max(1, round(self.stored_rate / self.target_rate))

        self.streaming = streaming
        self.captions_dir = captions_dir

        # ----- IMPORTED FROM ORBIS (not rewritten) -----
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ])
        # -------------------------------------------------

    def __len__(self):
        if self.streaming:
            raise TypeError("Streaming dataset does not support len().")
        return len(self.dataset)

    def load_captions(self, video_id):
        if not self.captions_dir:
            return None

        path = os.path.join(self.captions_dir, f"{video_id}.jsonl")
        if not os.path.exists(path):
            return None

        caps = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
            # Each line is: {"33": {...}} or {"0": {...}}
            # So extract the only key inside this line
                fid_str = next(iter(data.keys()))
                entry = data[fid_str]
                fid = int(fid_str)
                caps[fid] = entry.get("plain_caption", "")
        return caps

    def __getitem__(self, idx):
        sample = next(iter(self.dataset)) if self.streaming else self.dataset[idx]
        video = sample["video"]
        video_id = sample["video_id"]

        frame_captions = self.load_captions(video_id)

        total_frames = len(video)
        needed = self.num_frames * self.frame_interval

        if total_frames < needed:
            start = 0
        else:
            start = np.random.randint(0, total_frames - needed + 1)

        indices = [start + i * self.frame_interval for i in range(self.num_frames)]

        frames = []
        captions = []

        for i in indices:
            ft = video[i]
            arr = ft.numpy()

            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = np.transpose(arr, (1, 2, 0))

            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)

            img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")

            img = self.transform(img) * 2 - 1
            frames.append(img)

            if frame_captions:
                captions.append(frame_captions.get(i, ""))

        frames = torch.stack(frames, dim=0)

        return {
            "images": frames,
            "frame_rate": self.target_rate,
            "caption": captions if captions else f"Video {video_id}",
        }


if __name__ == "__main__":
    print("▶ Loading CoVLA dataset...")
    
    ds = CoVLAOrbisMultiFrame(
        split="train",
        num_frames=8,
        streaming=True,            # CoVLA must be streaming=True
        captions_dir="data/covla_captions",
    )

    # Random sample iterator
    it = iter(ds)
    sample = next(it)

    print("\n=== SAMPLE LOADED ===")
    print("Frame rate:", sample["frame_rate"])

    # Print captions
    print("\n=== CAPTIONS (one per sampled frame) ===")
    for i, cap in enumerate(sample["caption"]):
        print(f"Frame {i}: {cap}")

    # Show first image to visually confirm
    img = sample["images"][0]
    img = (img + 1) / 2    # back to [0,1]
    img_np = img.permute(1, 2, 0).numpy()

    plt.imshow(img_np)
    plt.title("First sampled frame")
    plt.axis("off")
    plt.show()

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
from torchvision import transforms

CACHE_DIR = "/home/fr/fr_fr/fr_aa533/work/covla_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def cache_path(video_id, frame_idx):
    return os.path.join(CACHE_DIR, f"{video_id}_{frame_idx}.pt")


class CoVLAOrbisMultiFrame(Dataset):
    def __init__(
        self,
        split="train",
        num_frames=6,
        stored_data_frame_rate=10,
        target_frame_rate=5,
        size=288,
        streaming=True,
        num_samples=None,
        captions_dir="data/covla_captions",
    ):
        self.dataset = load_dataset(
            "turing-motors/CoVLA-Dataset",
            split=split,
            streaming=streaming,
        )

        self.streaming = streaming

        if not streaming:
            self.dataset = list(self.dataset)
            if num_samples:
                self.dataset = self.dataset[:num_samples]

        self.iterator = iter(self.dataset)

        self.num_frames = num_frames
        self.stored_rate = stored_data_frame_rate
        self.target_rate = target_frame_rate
        self.frame_interval = max(1, round(self.stored_rate / self.target_rate))

        self.captions_dir = captions_dir

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ])

    def fetch_sample(self):
        while True:
            try:
                return next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataset)
            except Exception:
                continue

    def load_captions(self, video_id):
        path = os.path.join(self.captions_dir, f"{video_id}.jsonl")
        if not os.path.exists(path):
            return None

        caps = {}
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                fid_str = next(iter(data.keys()))
                caps[int(fid_str)] = data[fid_str].get("plain_caption", "")
        return caps

    def decode_frame(self, video_id, video, i):
        cp = cache_path(video_id, i)
        if os.path.exists(cp):
            return torch.load(cp)

        arr = video[i].numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)

        img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        img = self.transform(img) * 2 - 1

        torch.save(img, cp)
        return img

    def __getitem__(self, idx):
        sample = self.fetch_sample() if self.streaming else self.dataset[idx]

        video = sample["video"]
        video_id = sample.get("video_id")
        captions = self.load_captions(video_id)

        total = len(video)
        need = self.num_frames * self.frame_interval

        if total < need:
            idxs = [min(i, total - 1) for i in range(0, need, self.frame_interval)]
        else:
            start = np.random.randint(0, total - need + 1)
            idxs = [start + i * self.frame_interval for i in range(self.num_frames)]

        frames, texts = [], []
        for i in idxs:
            frames.append(self.decode_frame(video_id, video, i))
            texts.append(captions.get(i, "") if captions else "")

        frames = torch.stack(frames, 0)
        global_caption = texts[0] if len(texts) > 0 else ""
        return {"images": frames, "caption": global_caption, "frame_rate": self.target_rate}

    def __len__(self):
        raise TypeError("Streaming dataset does not have length.")

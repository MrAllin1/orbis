import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class CoVLADataset(Dataset):
    def __init__(
        self,
        split="train",
        streaming=True,
        resize=(288, 512),
        num_samples=None,
        auth_token=None
    ):
        """
        CoVLA dataset loader with video decoding, image preprocessing, and caption access.

        Args:
            split (str): Dataset split to use ("train", "test", etc.).
            streaming (bool): Use Hugging Face streaming API.
            resize (tuple): Target (height, width) resolution.
            num_samples (int): Limit number of samples (for debugging).
            auth_token (str): Optional Hugging Face auth token if dataset is gated.
        """

        self.dataset = load_dataset(
            "turing-motors/CoVLA-Dataset",
            split=split,
            streaming=streaming,
            use_auth_token=auth_token
        )

        if not streaming:
            self.dataset = list(self.dataset)
            if num_samples:
                self.dataset = self.dataset[:num_samples]

        self.streaming = streaming
        self.resize = resize

        self.transform = A.Compose([
            A.Resize(height=resize[0], width=resize[1]),
            A.CenterCrop(height=resize[0], width=resize[1]),
            A.Normalize(mean=0.5, std=0.5),  # Normalize to [-1, 1]
            ToTensorV2()
        ])

    def __len__(self):
        if self.streaming:
            raise TypeError("Streaming datasets do not support len(). Use iter().")
        return len(self.dataset)

    def __getitem__(self, idx):
        max_attempts = 10
        attempts = 0

        while attempts < max_attempts:
            sample = next(iter(self.dataset)) if self.streaming else self.dataset[idx]
            video = sample["video"]
            frame_tensor = video[0]

            # Sanity check: should be a 3D tensor (HWC or CHW)
            if isinstance(frame_tensor, torch.Tensor):
                arr = frame_tensor.numpy()

                if arr.ndim == 3:
                    # Handle CHW
                    if arr.shape[0] == 3:
                        arr = np.transpose(arr, (1, 2, 0))

                    # Grayscale → RGB
                    elif arr.shape[2] == 1:
                        arr = np.repeat(arr, 3, axis=2)

                    # Invalid shape
                    elif arr.shape[2] != 3:
                        attempts += 1
                        continue

                    try:
                        frame = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
                        frame_np = np.array(frame)
                        image_tensor = self.transform(image=frame_np)["image"]

                        caption = f"Video ID: {sample['video_id']}"

                        return {
                            "image": image_tensor,
                            "caption": caption
                        }

                    except Exception as e:
                        print(f"[Warning] Failed to convert frame: {e}")

            attempts += 1

        raise RuntimeError("Failed to load a valid frame after multiple attempts.")


def load_frame_captions(jsonl_path):
    frame_captions = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            frame_obj = json.loads(line)
            for frame_idx, data in frame_obj.items():
                frame_captions[int(frame_idx)] = data.get("plain_caption", "")
    return frame_captions


def show_covla_video_with_captions(captions_dir="captions"):
    # Load HF dataset
    dataset = load_dataset("turing-motors/CoVLA-Dataset", split="train", streaming=True)
    sample = next(iter(dataset))
    video = sample["video"]
    video_id = sample["video_id"]

    caption_path = os.path.join(captions_dir, f"{video_id}.jsonl")

    # Load per-frame captions
    if not os.path.exists(caption_path):
        print(f"Caption file not found for video_id {video_id}")
        return

    captions = load_frame_captions(caption_path)

    frames = []
    text_labels = []

    # Decode frames
    for i in range(len(video)):
        frame_tensor = video[i]
        arr = frame_tensor.numpy()

        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim != 3 or arr.shape[2] != 3:
            continue

        img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        frames.append(np.array(img))
        text_labels.append(captions.get(i, f"Frame {i}: No caption"))

    # Animate video
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.axis("off")

    im = ax.imshow(frames[0])
    text_obj = ax.text(
        10,
        10,
        text_labels[0],
        color="white",
        fontsize=10,
        backgroundcolor="black",
        verticalalignment="top",
        wrap=True
    )

    def update(frame_idx):
        im.set_array(frames[frame_idx])
        text_obj.set_text(text_labels[frame_idx])
        return [im, text_obj]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=100
    )

    plt.title(f"Video ID: {video_id}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_covla_video_with_captions(captions_dir="/Users/andialidema/Desktop/ORBIS-text-conditioning/orbis/data/covla_captions")

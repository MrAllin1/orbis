import json
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import time


class CoVLAOrbisMultiFrame(Dataset):
    """
    EXTENSIVE DEBUG VERSION
    Adds logs everywhere to detect slowdown / stuck points.
    """

    def __init__(
        self,
        num_frames=6,
        stored_data_frame_rate=20,
        target_frame_rate=5,
        size=288,
        captions_dir=None,
        num_samples=1000,
    ):
        print("\n================= [INIT] CoVLAOrbisMultiFrame =================")
        t0 = time.time()
        print("[INIT] Loading dataset from HuggingFace (streaming=True)…")

        # Load HF dataset in streaming mode
        self.dataset = load_dataset(
            "turing-motors/CoVLA-Dataset",
            split="train",
            streaming=True,
        )
        print("[INIT] HuggingFace dataset loaded. (streaming generator object)")

        self.num_frames = num_frames
        self.stored_rate = stored_data_frame_rate
        self.target_rate = target_frame_rate
        self.frame_interval = max(1, round(self.stored_rate / self.target_rate))

        print(f"[INIT] num_frames={num_frames}, frame_interval={self.frame_interval}")

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ])
        print("[INIT] Transforms initialized.")

        self.captions_dir = captions_dir
        self.num_samples = num_samples

        print("[INIT] Creating dataset iterator…")
        self.iterator = iter(self.dataset)
        print("[INIT] Iterator ready.")

        print(f"[INIT] Completed in {time.time() - t0:.2f} seconds")
        print("================================================================\n")

    # --------------------------------------------------------------
    def load_captions(self, video_id):
        print(f"[CAPTION] Loading captions for video_id={video_id}")

        if not self.captions_dir:
            print("[CAPTION] No captions_dir set → return empty dict.")
            return {}

        path = f"{self.captions_dir}/{video_id}.jsonl"
        print(f"[CAPTION] Caption file path: {path}")

        try:
            with open(path, "r") as f:
                caps = {
                    int(list(entry.keys())[0]): entry[list(entry.keys())[0]]["plain_caption"]
                    for entry in map(json.loads, f)
                }
            print(f"[CAPTION] Loaded {len(caps)} caption lines.")
            return caps

        except FileNotFoundError:
            print("[CAPTION] Caption JSONL file missing → return empty dict.")
            return {}

    # --------------------------------------------------------------
    def __getitem__(self, idx):

        print(f"\n================= [GETITEM] idx={idx} =================")

        # --- Retrieve sample -------------------------
        print("[GETITEM] Fetching next sample from iterator…")
        try:
            sample = next(self.iterator)
            print("[GETITEM] Received sample OK.")
        except StopIteration:
            print("[GETITEM] Iterator exhausted — restarting dataset iterator.")
            self.iterator = iter(load_dataset(
                "turing-motors/CoVLA-Dataset",
                split="train",
                streaming=True,
            ))
            sample = next(self.iterator)

        # --- Parse sample ----------------------------
        video = sample["video"]
        video_id = sample["video_id"]

        print(f"[GETITEM] video_id={video_id}")
        print("[GETITEM] Counting frames…")
        total_frames = len(video)
        print(f"[GETITEM] total_frames={total_frames}")

        # --- Load captions ---------------------------
        captions = self.load_captions(video_id)

        # --- Select frame indices --------------------
        print("[GETITEM] Computing frame indices…")
        need = self.num_frames * self.frame_interval
        idxs = [min(i, total_frames - 1) for i in range(0, need, self.frame_interval)]
        print(f"[GETITEM] Selected indices: {idxs}")

        frames = []
        texts = []

        # --- Decode frames ---------------------------
        print("[GETITEM] Decoding frames now…")

        for i in idxs:
            print(f"[GETITEM] Decoding frame {i}…")

            try:
                frame = video[i].asnumpy()
                print("[GETITEM] Frame decoded to NumPy.")
            except Exception as e:
                print(f"[ERROR] Failed decoding frame {i}: {e}")
                raise e

            print("[GETITEM] Converting NumPy → PIL")
            img = Image.fromarray(frame).convert("RGB")

            print("[GETITEM] Applying transforms…")
            img = self.transform(img) * 2 - 1

            frames.append(img)
            texts.append(captions.get(i, ""))

            print(f"[GETITEM] Frame {i} processed.")

        # --- Stack frames -----------------------------
        print("[GETITEM] Stacking frames into a tensor…")
        frames = torch.stack(frames, 0)

        global_caption = texts[0] if texts else ""
        print(f"[GETITEM] global_caption='{global_caption[:50]}…'")

        print("[GETITEM] Returning output dict.\n")
        return {
            "images": frames,
            "caption": global_caption,
            "video_id": video_id,
        }

    # --------------------------------------------------------------
    def __len__(self):
        return self.num_samples

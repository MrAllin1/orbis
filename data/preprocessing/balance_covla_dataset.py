import json
import shutil
import random
from pathlib import Path
from collections import Counter, defaultdict

# ================= CONFIG =================
INPUT_DIR = Path("/work/dlclarge2/alidemaa-text-control-orbis/orbis/data/covla_captions")
OUTPUT_DIR = Path("/work/dlclarge2/alidemaa-text-control-orbis/orbis/data/covla_captions_balanced")
LABELS_TXT = OUTPUT_DIR.parent / "covla_video_labels.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CONFIDENCE = 0.5     # majority threshold
FILTER_NIGHT = True
BALANCE_MODE = "ratio"  # "equal" or "ratio"

CUSTOM_RATIOS = {
    "left": 0.35,
    "right": 0.35,
    "straight": 0.30,
}

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ================= HELPERS =================
NIGHT_KEYWORDS = ["night", "dark", "evening", "sunset", "dusk"]

def is_day(text):
    text = text.lower()
    return not any(k in text for k in NIGHT_KEYWORDS)

def extract_direction(text):
    text = text.lower()
    if "turning left" in text or "turn left" in text:
        return "left"
    if "turning right" in text or "turn right" in text:
        return "right"
    if "straight" in text or "going straight" in text:
        return "straight"
    return None

# ================= CLASSIFY VIDEOS =================
video_info = {}

for jsonl_file in INPUT_DIR.glob("*.jsonl"):
    counts = Counter()

    with open(jsonl_file) as f:
        for line in f:
            entry = json.loads(line)
            for _, data in entry.items():
                caption = (
                    data.get("plain_caption", "") +
                    " " +
                    data.get("rich_caption", "")
                )

                if FILTER_NIGHT and not is_day(caption):
                    continue

                direction = extract_direction(caption)
                if direction:
                    counts[direction] += 1

    total = sum(counts.values())

    if total == 0:
        label = "unknown"
        confidence = 0.0
    else:
        label, votes = counts.most_common(1)[0]
        confidence = votes / total
        if confidence < MIN_CONFIDENCE:
            label = "ambiguous"

    video_info[jsonl_file.name] = {
        "path": jsonl_file,
        "label": label,
        "counts": dict(counts),
        "confidence": round(confidence, 3),
        "frames_used": total,
    }

# ================= FILTER VALID VIDEOS =================
valid_videos = {
    k: v for k, v in video_info.items()
    if v["label"] in {"left", "right", "straight"}
}

by_label = defaultdict(list)
for name, info in valid_videos.items():
    by_label[info["label"]].append(info)

print("Before balancing:", {k: len(v) for k, v in by_label.items()})

# ================= BALANCE VIDEOS =================
balanced = []

if BALANCE_MODE == "equal":
    min_count = min(len(v) for v in by_label.values())
    for label, items in by_label.items():
        balanced.extend(random.sample(items, min_count))

elif BALANCE_MODE == "ratio":
    total = sum(len(v) for v in by_label.values())
    for label, ratio in CUSTOM_RATIOS.items():
        k = int(total * ratio)
        balanced.extend(random.sample(by_label[label], min(k, len(by_label[label]))))

else:
    raise ValueError("Invalid BALANCE_MODE")

print("After balancing:", Counter(v["label"] for v in balanced))

# ================= COPY FILES =================
for info in balanced:
    shutil.copy2(info["path"], OUTPUT_DIR / info["path"].name)

# ================= WRITE LABELS TXT =================
with open(LABELS_TXT, "w") as f:
    for info in sorted(balanced, key=lambda x: x["path"].name):
        f.write(
            f"{info['path'].name}\t"
            f"{info['label']}\t"
            f"confidence={info['confidence']}\t"
            f"frames={info['frames_used']}\t"
            f"counts={info['counts']}\n"
        )

print(f"\nBalanced dataset written to: {OUTPUT_DIR}")
print(f"Video labels saved to: {LABELS_TXT}")

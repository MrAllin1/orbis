import argparse
import json
import re
from pathlib import Path
from collections import Counter

SPEED_RE = re.compile(r"\b(high|moderate|low)\s+speed\b", re.I)
TURN_RE  = re.compile(r"\b(turning\s+(left|right)|moving\s+(straight))\b", re.I)
DECEL_RE = re.compile(r"\bdeceleration\b", re.I)
TL_NONE_RE = re.compile(r"\bno\s+traffic\s+light\b", re.I)
TL_RE = re.compile(r"\btraffic\s+light(s)?\b", re.I)
COLOR_RE = re.compile(r"\b(red|green|yellow)\b", re.I)
ARROW_RE = re.compile(r"\b(left|right|straight)\s+arrow\b", re.I)

SPEED_ORDER = {"low": 0, "moderate": 1, "high": 2}

def iter_jsonl_files(p: Path):
    if p.is_file():
        yield p
    else:
        for f in sorted(p.glob("*.jsonl")):
            if f.is_file():
                yield f

def parse_line(line: str):
    obj = json.loads(line)
    if not isinstance(obj, dict) or len(obj) != 1:
        return None
    k = list(obj.keys())[0]
    v = obj[k] if isinstance(obj[k], dict) else {}
    cap = v.get("plain_caption", "") or v.get("rich_caption", "") or ""
    return str(k), str(cap)

def extract_state(caption: str):
    c = caption.strip()

    # turn
    turn = "unknown"
    m = TURN_RE.search(c)
    if m:
        if m.group(2):
            turn = m.group(2).lower()   # left/right
        elif m.group(3):
            turn = "straight"

    # speed
    speed = "unknown"
    m = SPEED_RE.search(c)
    if m:
        speed = m.group(1).lower()

    # deceleration (kept, but no longer drives "after a while" by itself)
    decel = bool(DECEL_RE.search(c))

    # traffic light
    if TL_NONE_RE.search(c):
        tl = "none"
        colors = []
        arrows = []
    else:
        tl = "present" if TL_RE.search(c) else "unknown"
        colors = [m.group(1).lower() for m in COLOR_RE.finditer(c)]
        arrows = [m.group(1).lower() for m in ARROW_RE.finditer(c)]

    return {
        "turn": turn,
        "speed": speed,
        "decel": decel,
        "tl": tl,
        "colors": colors,
        "arrows": arrows,
    }

def dominant(items):
    """
    Returns (value, ratio) for the most common non-unknown item.
    ratio is wrt number of non-unknown items.
    """
    vals = [x for x in items if x and x != "unknown"]
    if not vals:
        return "unknown", 0.0
    c = Counter(vals)
    val, cnt = c.most_common(1)[0]
    return val, cnt / len(vals)

def build_later_clause(states, max_frame=300, min_late_ratio=0.6, min_total_frames=60):
    """
    Add a 'Later ...' clause ONLY if there is a sustained change between early and late windows.
    - early window: first 40%
    - late window: last 40%
    """
    n = len(states)
    if n < min_total_frames:
        return ""

    e1 = int(0.4 * n)
    l0 = int(0.6 * n)

    early = states[:e1]
    late = states[l0:]

    early_turn, _ = dominant([st["turn"] for _, st in early])
    late_turn, late_turn_ratio = dominant([st["turn"] for _, st in late])

    early_speed, _ = dominant([st["speed"] for _, st in early])
    late_speed, late_speed_ratio = dominant([st["speed"] for _, st in late])

    events = []

    # Turn change (e.g., straight -> left)
    if late_turn != "unknown" and late_turn_ratio >= min_late_ratio and late_turn != early_turn:
        events.append(f"turns {late_turn}")

    # Speed regime change (low/moderate/high)
    if (
        late_speed != "unknown"
        and late_speed_ratio >= min_late_ratio
        and early_speed in SPEED_ORDER
        and late_speed in SPEED_ORDER
        and late_speed != early_speed
    ):
        if SPEED_ORDER[late_speed] < SPEED_ORDER[early_speed]:
            events.append("slows down")
        else:
            events.append("speeds up")

    if not events:
        return ""
    if len(events) == 1:
        return f"Later it {events[0]}."
    return f"Later it {events[0]} and {events[1]}."

def build_compact(frames_caps, max_frame=300):
    """
    frames_caps: list[(frame_idx_int, caption_str)] sorted by frame_idx
    """
    window = [(i, cap) for i, cap in frames_caps if i <= max_frame]
    if not window:
        return "no caption"

    states = []
    for i, cap in window:
        st = extract_state(cap)
        states.append((i, st))

    # choose dominant (turn,speed) over the full window
    ts_counter = Counter(
        (st["turn"], st["speed"]) for _, st in states
        if st["turn"] != "unknown" and st["speed"] != "unknown"
    )
    if ts_counter:
        (turn, speed), _ = ts_counter.most_common(1)[0]
    else:
        turn = next((st["turn"] for _, st in states if st["turn"] != "unknown"), "straight")
        speed = next((st["speed"] for _, st in states if st["speed"] != "unknown"), "moderate")

    # traffic lights summary
    tl_vals = [st["tl"] for _, st in states]
    if all(t == "none" for t in tl_vals if t != "unknown") and any(t == "none" for t in tl_vals):
        tl_sentence = "No traffic lights."
    else:
        colors = []
        arrows = []
        for _, st in states:
            colors += st["colors"]
            arrows += st["arrows"]
        colors = [c for c, _ in Counter(colors).most_common(2)]
        arrows = [a for a, _ in Counter(arrows).most_common(2)]

        if colors or arrows:
            parts = []
            if colors:
                parts.append("signals: " + ", ".join(colors))
            if arrows:
                parts.append("arrows: " + ", ".join(arrows))
            tl_sentence = "Traffic lights present (" + "; ".join(parts) + ")."
        else:
            tl_sentence = "No traffic lights." if "none" in tl_vals else ""

    # NEW: later clause based on sustained early->late change (not just "deceleration" keyword)
    later = build_later_clause(states, max_frame=max_frame)

    # build final short caption
    sents = []
    if turn == "straight":
        sents.append(f"Moving straight at {speed} speed.")
    else:
        sents.append(f"Turning {turn} at {speed} speed.")

    if tl_sentence:
        sents.append(tl_sentence)

    if later:
        sents.append(later)

    return " ".join(sents).strip()

def process_file(in_file: Path, out_file: Path, max_frame=300):
    keys = []
    frames_caps = []
    with in_file.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_line(line)
            if not parsed:
                continue
            k, cap = parsed
            keys.append(k)
            try:
                idx = int(k)
            except ValueError:
                idx = len(frames_caps)
            frames_caps.append((idx, cap))

    frames_caps.sort(key=lambda x: x[0])
    compact = build_compact(frames_caps, max_frame=max_frame)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for k in keys:
            f.write(json.dumps({k: {"plain_caption": compact}}, ensure_ascii=False) + "\n")

    return compact, len(keys)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="Input .jsonl file or directory of .jsonl files")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--max_frame", type=int, default=300, help="Ignore captions after this frame index")
    ap.add_argument("--min_late_ratio", type=float, default=0.6,
                    help="How dominant a late-state must be to count as a real change")
    ap.add_argument("--min_total_frames", type=int, default=60,
                    help="Minimum frames (within 0..max_frame) to attempt early/late change detection")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in iter_jsonl_files(in_path):
        out_file = out_dir / f.name
        compact, n = process_file(f, out_file, max_frame=args.max_frame)
        print(f"[OK] {f.name}: frames={n} | max_frame={args.max_frame}")
        print(f"     compact: {compact}")

if __name__ == "__main__":
    main()

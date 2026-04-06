"""
Upload processed Gurbani Kirtan data to HuggingFace as a dataset.
Converts audio to FLAC, runs OCR on slide frames, matches against STTM
for canonical text, builds dataset with train/val/test splits.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pytesseract
import soundfile as sf
from PIL import Image
from huggingface_hub import HfApi, create_repo

from sttm_matcher import STTMMatcher

REPO_ID = "surindersinghssj/gurbani-kirtan-dataset-v2"
MAX_SEGMENT_DURATION = 30.0  # Whisper's preferred window size


def ocr_slide(frame_path, threshold_val=150):
    """Extract Gurmukhi and English text from a slide frame."""
    img = cv2.imread(frame_path)
    if img is None:
        return "", ""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    pil_img = Image.fromarray(thresh)

    combined = pytesseract.image_to_string(pil_img, lang="pan+eng", config="--psm 6")
    lines = [l.strip() for l in combined.strip().split("\n") if l.strip()]

    gurmukhi_lines = []
    english_lines = []
    for line in lines:
        has_gurmukhi = any("\u0A00" <= ch <= "\u0A7F" for ch in line)
        if has_gurmukhi:
            gurmukhi_lines.append(line)
        else:
            english_lines.append(line)

    return "\n".join(gurmukhi_lines), "\n".join(english_lines)


def wav_to_flac(wav_path):
    """Convert WAV to FLAC using ffmpeg."""
    flac_path = wav_path.rsplit(".", 1)[0] + ".flac"
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-c:a", "flac", "-ar", "16000", "-ac", "1",
         flac_path, "-loglevel", "error"],
        check=True,
    )
    return flac_path


def split_audio_segment(wav_path, max_duration=MAX_SEGMENT_DURATION):
    """Split a WAV file into chunks <= max_duration seconds.
    Returns list of (chunk_path, offset_seconds) tuples."""
    data, sr = sf.read(wav_path)
    total_duration = len(data) / sr

    if total_duration <= max_duration:
        return [(wav_path, 0.0)]

    chunks = []
    max_samples = int(max_duration * sr)
    base_name = wav_path.rsplit(".", 1)[0]

    for i, start in enumerate(range(0, len(data), max_samples)):
        chunk_data = data[start:start + max_samples]
        if len(chunk_data) / sr < 0.5:  # skip tiny trailing chunks
            break
        chunk_path = f"{base_name}_chunk{i:02d}.wav"
        sf.write(chunk_path, chunk_data, sr)
        chunks.append((chunk_path, start / sr))

    return chunks


def assign_splits(video_ids, train_ratio=0.8, val_ratio=0.1):
    """Assign video IDs to train/val/test splits deterministically.
    Uses hash of video_id for stable, reproducible assignment."""
    splits = {}
    for vid in video_ids:
        h = int(hashlib.md5(vid.encode()).hexdigest(), 16) % 100
        if h < int(train_ratio * 100):
            splits[vid] = "train"
        elif h < int((train_ratio + val_ratio) * 100):
            splits[vid] = "validation"
        else:
            splits[vid] = "test"
    return splits


def build_rows(processed_dir, video_id, shabad_title, matcher=None,
               channel="rajikaurrr", kirtan_style="studio"):
    """Build dataset rows from processed video output.

    Args:
        processed_dir: Path to processed video directory
        video_id: YouTube video ID
        shabad_title: Title of the shabad
        matcher: STTMMatcher instance for canonical text correction
        channel: YouTube channel name
        kirtan_style: One of: studio, gurdwara_live, akj, rain_sabai, etc.
    """
    manifest_path = os.path.join(processed_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    rows = []
    segments = manifest["segments"]

    frames_dir = os.path.join(processed_dir, "frames")
    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.endswith(".jpg")]
    ) if os.path.isdir(frames_dir) else []

    for seg in segments:
        idx = seg["index"]

        gurmukhi_ocr = ""
        english_ocr = ""
        slide_image_path = None

        frame_idx = idx - 1
        if 0 <= frame_idx < len(frame_files):
            slide_image_path = os.path.join(frames_dir, frame_files[frame_idx])
            gurmukhi_ocr, english_ocr = ocr_slide(slide_image_path)
            print(f"  Slide {idx}: {gurmukhi_ocr[:60]}...")

        wav_path = seg["audio_path"]
        if not os.path.exists(wav_path):
            continue

        # Match against STTM for canonical text
        gurmukhi_canonical = ""
        english_canonical = ""
        match_score = 0.0

        if matcher:
            # Try Gurmukhi match first, then English fallback
            result = matcher.match(gurmukhi_ocr)
            if not result and english_ocr:
                result = matcher.match_by_english(english_ocr)
            if result:
                gurmukhi_canonical = result["gurmukhi_canonical"]
                english_canonical = result["english_translation"]
                match_score = result["score"]

        # Split long segments for Whisper compatibility
        chunks = split_audio_segment(wav_path)

        for chunk_path, offset in chunks:
            flac_path = wav_to_flac(chunk_path)
            audio_data, sr = sf.read(flac_path)
            chunk_duration = len(audio_data) / sr

            rows.append({
                "audio_path": flac_path,
                "image_path": slide_image_path,
                "gurmukhi_ocr": gurmukhi_ocr,
                "gurmukhi_text": gurmukhi_canonical or gurmukhi_ocr,
                "english_translation": english_canonical or english_ocr,
                "english_ocr": english_ocr,
                "match_score": match_score,
                "start_time": round(seg["start"] + offset, 3),
                "end_time": round(seg["start"] + offset + chunk_duration, 3),
                "duration": round(chunk_duration, 3),
                "slide_index": idx,
                "video_id": video_id,
                "shabad_title": shabad_title,
                "channel": channel,
                "kirtan_style": kirtan_style,
            })

    return rows


def upload_dataset(rows, output_dir="/tmp/hf_dataset"):
    """Upload dataset files and metadata to HuggingFace Hub."""
    api = HfApi()

    # Create repo if needed
    create_repo(REPO_ID, repo_type="dataset", exist_ok=True)

    # Assign splits by video ID
    video_ids = list(set(r["video_id"] for r in rows))
    split_map = assign_splits(video_ids)

    # Prepare split directories
    for split in ["train", "validation", "test"]:
        os.makedirs(os.path.join(output_dir, "data", split, "audio"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "data", split, "images"), exist_ok=True)

    # Copy files and build metadata per split
    split_rows = {"train": [], "validation": [], "test": []}

    for row in rows:
        split = split_map[row["video_id"]]

        # Copy audio
        audio_fname = f"{row['video_id']}_{row['slide_index']:03d}_{row['start_time']:.1f}s.flac"
        audio_dest = os.path.join(output_dir, "data", split, "audio", audio_fname)
        shutil.copy2(row["audio_path"], audio_dest)

        # Copy slide image if exists
        image_fname = ""
        if row["image_path"] and os.path.exists(row["image_path"]):
            image_fname = f"{row['video_id']}_{row['slide_index']:03d}.jpg"
            image_dest = os.path.join(output_dir, "data", split, "images", image_fname)
            if not os.path.exists(image_dest):
                shutil.copy2(row["image_path"], image_dest)

        split_rows[split].append({
            "file_name": f"audio/{audio_fname}",
            "image": f"images/{image_fname}" if image_fname else "",
            "gurmukhi_ocr": row["gurmukhi_ocr"],
            "gurmukhi_text": row["gurmukhi_text"],
            "english_translation": row["english_translation"],
            "english_ocr": row["english_ocr"],
            "match_score": row["match_score"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "duration": row["duration"],
            "slide_index": row["slide_index"],
            "video_id": row["video_id"],
            "shabad_title": row["shabad_title"],
            "channel": row["channel"],
            "kirtan_style": row["kirtan_style"],
        })

    # Write metadata.jsonl per split
    for split, srows in split_rows.items():
        if not srows:
            continue
        metadata_path = os.path.join(output_dir, "data", split, "metadata.jsonl")
        with open(metadata_path, "w") as f:
            for r in srows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {split}: {len(srows)} rows")

    print(f"\nPrepared {len(rows)} total rows in {output_dir}")
    print("Uploading to HuggingFace Hub...")

    # Upload each split directory
    for split in ["train", "validation", "test"]:
        split_dir = os.path.join(output_dir, "data", split)
        if not os.path.exists(os.path.join(split_dir, "metadata.jsonl")):
            continue
        api.upload_folder(
            folder_path=split_dir,
            repo_id=REPO_ID,
            repo_type="dataset",
            path_in_repo=f"data/{split}",
        )

    # Upload README
    readme = create_readme(rows, split_rows)
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    print(f"\nUploaded to https://huggingface.co/datasets/{REPO_ID}")


def create_readme(rows, split_rows):
    total_duration = sum(r["duration"] for r in rows)
    num_videos = len(set(r["video_id"] for r in rows))
    num_segments = len(rows)
    matched = sum(1 for r in rows if r["match_score"] > 0)
    avg_duration = total_duration / num_segments if num_segments else 0
    styles = set(r.get("kirtan_style", "") for r in rows)

    train_n = len(split_rows.get("train", []))
    val_n = len(split_rows.get("validation", []))
    test_n = len(split_rows.get("test", []))

    return f"""---
language:
  - pa
  - en
license: cc-by-4.0
task_categories:
  - automatic-speech-recognition
  - audio-classification
tags:
  - gurbani
  - kirtan
  - sikh
  - punjabi
  - gurmukhi
  - speech
  - music
  - whisper
size_categories:
  - {"n<1K" if num_segments < 1000 else "1K<n<10K" if num_segments < 10000 else "10K<n<100K"}
---

# Gurbani Kirtan Dataset V2

A timestamped Gurbani Kirtan dataset with line-level audio segments, canonical Gurmukhi text
(matched against SikhiToTheMax database), and English translations extracted from YouTube kirtan videos.

## Dataset Description

Each row represents a single Gurbani line (slide) from a kirtan video, with:
- **Audio segment** (FLAC, 16kHz mono, <=30s for Whisper compatibility)
- **Gurmukhi text** — canonical text matched against STTM database
- **Gurmukhi OCR** — raw OCR output for comparison
- **English translation** — from Dr. Sant Singh Khalsa (via STTM)
- **Slide image** — original text overlay from the video
- **Kirtan style** — studio, gurdwara_live, akj, etc.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total segments | {num_segments} |
| Total audio duration | {total_duration:.1f}s ({total_duration/3600:.1f}h) |
| Number of videos | {num_videos} |
| STTM matched | {matched}/{num_segments} ({100*matched/num_segments:.0f}%) |
| Avg segment duration | {avg_duration:.1f}s |
| Audio format | FLAC, 16kHz, mono |
| Languages | Punjabi (Gurmukhi), English |
| Kirtan styles | {', '.join(sorted(styles))} |

## Splits

| Split | Segments | Note |
|-------|----------|------|
| train | {train_n} | 80% of videos |
| validation | {val_n} | 10% of videos |
| test | {test_n} | 10% of videos |

Splits are by **video ID** (not segment) to prevent data leakage.

## Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `file_name` | string | Path to FLAC audio file |
| `image` | string | Path to slide image (JPG) |
| `gurmukhi_text` | string | Canonical Gurbani line (STTM-corrected) |
| `gurmukhi_ocr` | string | Raw OCR output from slide |
| `english_translation` | string | English translation (Dr. Sant Singh Khalsa) |
| `english_ocr` | string | Raw English OCR from slide |
| `match_score` | float | STTM fuzzy match confidence (0-100) |
| `start_time` | float | Start time in source video (seconds) |
| `end_time` | float | End time in source video (seconds) |
| `duration` | float | Duration of audio segment (seconds) |
| `slide_index` | int | Index of the slide in the video |
| `video_id` | string | YouTube video ID |
| `shabad_title` | string | Title of the shabad |
| `channel` | string | YouTube channel name |
| `kirtan_style` | string | Style: studio, gurdwara_live, akj, rain_sabai |

## How to Use

```python
from datasets import load_dataset

ds = load_dataset("surindersinghssj/gurbani-kirtan-dataset-v2")

# Access splits
train = ds["train"]
val = ds["validation"]
test = ds["test"]

# Filter high-confidence matches only
high_quality = train.filter(lambda x: x["match_score"] >= 80)
```

## Pipeline

1. **Download** kirtan videos from YouTube using `yt-dlp`
2. **Detect slide transitions** using OpenCV frame differencing (auto-threshold: median + 4*std)
3. **Extract audio segments** between transitions using `ffmpeg` (FLAC 16kHz mono)
4. **OCR slide text** using Tesseract (`pan+eng`)
5. **Match against STTM** database (141K lines) for canonical Gurmukhi + English translations
6. **Split segments >30s** for Whisper ASR compatibility
7. **Assign train/val/test** splits by video ID (80/10/10)

## Source

Videos sourced from YouTube kirtan channels that display each Gurbani line
as a slide synchronized to the audio.

## License

CC-BY-4.0. Please credit the original kirtan artists and channels.

## Citation

```bibtex
@dataset{{gurbani_kirtan_v2,
  title={{Gurbani Kirtan Dataset V2}},
  author={{Surinder Singh}},
  year={{2026}},
  url={{https://huggingface.co/datasets/surindersinghssj/gurbani-kirtan-dataset-v2}}
}}
```
"""


if __name__ == "__main__":
    processed_dir = sys.argv[1] if len(sys.argv) > 1 else "/root/gurbani_kirtan/processed"
    video_id = sys.argv[2] if len(sys.argv) > 2 else "ObPNeuIN17c"
    shabad_title = sys.argv[3] if len(sys.argv) > 3 else "Mango Daan Thakur Naam"
    db_path = sys.argv[4] if len(sys.argv) > 4 else "database.sqlite"

    # Initialize STTM matcher
    matcher = None
    if os.path.exists(db_path):
        print("=== Loading STTM matcher ===")
        matcher = STTMMatcher(db_path)
        matcher.load()

    print("\n=== Building dataset rows ===")
    rows = build_rows(processed_dir, video_id, shabad_title, matcher=matcher)
    print(f"Built {len(rows)} rows")

    print("\n=== Uploading to HuggingFace ===")
    upload_dataset(rows)

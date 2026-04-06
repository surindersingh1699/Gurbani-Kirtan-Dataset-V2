"""Update the HF dataset README."""

import os
from huggingface_hub import HfApi

REPO_ID = "surindersinghssj/gurbani-kirtan-dataset-v2"

readme = """---
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
size_categories:
  - n<1K
builder_name: audiofolder
configs:
  - config_name: default
    data_dir: data
    default: true
---

# Gurbani Kirtan Dataset V2

A timestamped Gurbani Kirtan dataset with line-level audio segments, Gurmukhi text, and English translations extracted from YouTube kirtan videos.

## Dataset Description

Each row represents a single Gurbani line (slide) from a kirtan video, with:
- **Audio segment** (FLAC, 16kHz mono) of that line being sung
- **Gurmukhi text** (Punjabi script) of the line
- **English translation** of the line
- **Timestamps** (start/end time in the source video)
- **Video ID** for source attribution

## How to Use

```python
from datasets import load_dataset

ds = load_dataset("surindersinghssj/gurbani-kirtan-dataset-v2")

# Play audio and see text
print(ds["train"][0]["gurmukhi_text"])
print(ds["train"][0]["english_translation"])
print(ds["train"][0]["audio"])
```

## Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `audio` | Audio | FLAC audio segment (16kHz, mono) |
| `gurmukhi_text` | string | Gurbani line in Gurmukhi script |
| `english_translation` | string | English translation of the line |
| `start_time` | float | Start time in source video (seconds) |
| `end_time` | float | End time in source video (seconds) |
| `duration` | float | Duration of audio segment (seconds) |
| `slide_index` | int | Index of the slide in the video |
| `video_id` | string | YouTube video ID (source attribution) |
| `shabad_title` | string | Title of the shabad |
| `channel` | string | YouTube channel name |

## Pipeline

This dataset was built using an automated pipeline:
1. **Download** kirtan videos from YouTube using `yt-dlp`
2. **Detect slide transitions** using OpenCV frame differencing (auto-threshold)
3. **Extract audio segments** between transitions using `ffmpeg` (FLAC, 16kHz mono)
4. **OCR slide text** using Tesseract with Punjabi + English models
5. **Package** as a HuggingFace dataset

## Source

Videos sourced from YouTube kirtan channels that display each Gurbani line as a slide synchronized to the audio.

## License

CC-BY-4.0. Please credit the original kirtan artists and channels.

## Citation

```bibtex
@dataset{gurbani_kirtan_v2,
  title={Gurbani Kirtan Dataset V2},
  author={Surinder Singh},
  year={2026},
  url={https://huggingface.co/datasets/surindersinghssj/gurbani-kirtan-dataset-v2}
}
```
"""

api = HfApi()
api.upload_file(
    path_or_fileobj=readme.encode(),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
)
print("README updated")

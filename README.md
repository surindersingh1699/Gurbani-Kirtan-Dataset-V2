# Gurbani Kirtan Dataset V2

Automated pipeline for building a timestamped Gurbani Kirtan audio dataset from YouTube kirtan videos where each line is displayed as a slide synced to audio.

## What It Does

Takes kirtan videos like [this format](https://www.youtube.com/@rajikaurrr) where each Gurbani line appears as a slide, and produces:

- **FLAC audio segments** (16kHz mono, <=30s) for each line
- **Canonical Gurmukhi text** matched against STTM database (141K lines)
- **English translations** from Dr. Sant Singh Khalsa
- **Slide images** with original text overlays
- **Train/val/test splits** by video ID for ML training

Output is uploaded to [HuggingFace](https://huggingface.co/datasets/surindersinghssj/gurbani-kirtan-dataset-v2).

## Pipeline

```
YouTube Video → yt-dlp download
    → OpenCV slide detection (frame diff, auto-threshold)
    → ffmpeg audio extraction (FLAC 16kHz mono)
    → Tesseract OCR (pan+eng)
    → STTM fuzzy matching (canonical text + translation)
    → HuggingFace upload (train/val/test splits)
```

## Quick Start

### 1. Process a single video

```bash
python process_video.py /path/to/video.mp4 /path/to/output/
```

### 2. Build dataset and upload to HuggingFace

```bash
python upload_to_hf.py /path/to/processed/ VIDEO_ID "Shabad Title"
```

### 3. Batch process entire channels (8 workers)

```bash
python batch_process.py --workers 8 --channels channels.txt --cookies cookies.txt
```

## Files

| File | Description |
|------|-------------|
| `process_video.py` | Slide detection + frame/audio extraction pipeline |
| `upload_to_hf.py` | Dataset builder with STTM matching + HF uploader |
| `sttm_matcher.py` | Fuzzy matcher against SikhiToTheMax Gurbani database |
| `batch_process.py` | Parallel batch processor with checkpointing |
| `database.sqlite` | STTM database (141K Gurbani lines + translations) |
| `channels.txt` | YouTube channel URLs to process |
| `cookies.txt` | YouTube cookies for yt-dlp (not committed) |

## STTM Matching Strategy

OCR'd text is matched against the STTM database using a 2-tier approach:

1. **First-letters index** — Extract first Gurmukhi letter of each word, map to STTM ASCII, query the indexed `first_letters` column to narrow candidates, then fuzzy-match among them
2. **Full corpus search** — Fallback using `rapidfuzz` token_sort_ratio across all 141K lines
3. **English fallback** — If Gurmukhi OCR is too poor, match English text against 60K translations

Uses `anvaad-py` to convert STTM ASCII Gurmukhi to Unicode.

## Dataset Fields

| Field | Description |
|-------|-------------|
| `gurmukhi_text` | Canonical text (STTM-corrected) |
| `gurmukhi_ocr` | Raw OCR output from slide |
| `english_translation` | English translation (Dr. Sant Singh Khalsa) |
| `english_ocr` | Raw English from slide OCR |
| `match_score` | STTM match confidence (0-100) |
| `kirtan_style` | studio, gurdwara_live, akj, rain_sabai |
| `audio` | FLAC 16kHz mono, <=30s segments |
| `duration` | Segment duration in seconds |
| `video_id` | YouTube source video ID |

## Performance

| Scenario | Estimated Time (300h kirtan) |
|----------|------------------------------|
| Single-threaded | ~18.6 hours |
| 4 workers | ~4.7 hours |
| 8 workers | ~2.3 hours |

## Server

- SSH: see `.env` for `SERVER_SSH`
- Working dir: `/root/gurbani_kirtan/`
- yt-dlp requires: `--js-runtimes node` flag
- CPU-only processing (no GPU needed)

## Dependencies

```
opencv-python
pytesseract
soundfile
numpy
Pillow
huggingface_hub
yt-dlp
ffmpeg
anvaad-py
rapidfuzz
```

## License

CC-BY-4.0

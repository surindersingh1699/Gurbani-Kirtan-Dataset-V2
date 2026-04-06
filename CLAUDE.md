# Gurbani Kirtan Dataset V2

## Project
Building a timestamped Gurbani Kirtan audio dataset from YouTube kirtan videos where each line is displayed as a slide synced to audio.

## Pipeline
1. **Download** — `yt-dlp` with Firefox cookies (`cookies.txt`, refresh every ~2 weeks)
2. **Slide detection** — OpenCV pixel-diff with auto-threshold (median + 4*std)
3. **OCR** — Tesseract (`pan+eng`) with threshold=150 preprocessing on extracted frames
4. **Audio extraction** — ffmpeg splits audio at slide transitions, output as FLAC 16kHz mono
5. **Upload** — HuggingFace `datasets` library to `surindersinghssj/gurbani-kirtan-dataset-v2`

## Server
- SSH: `root@138.199.174.101` (from `.env` SERVER_SSH)
- Working dir: `/root/gurbani_kirtan/`
- yt-dlp requires: `--js-runtimes node` flag (Node 22 installed)
- No GPU — all CPU processing

## Key Files
- `process_video.py` — slide detection + frame/audio extraction pipeline
- `upload_to_hf.py` — dataset builder + HF uploader
- `cookies.txt` — Firefox YouTube cookies (do NOT commit)
- `channels.txt` — YouTube channel URLs to process

## YouTube Channels
- `@rajikaurrr` — slide-per-line kirtan format

## Performance
- Processing speed: ~16x realtime (single-threaded)
- 300h of kirtan ≈ 18.6h processing (single) / 4.7h (4 workers)

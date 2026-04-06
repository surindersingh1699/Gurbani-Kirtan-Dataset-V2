"""
Batch processor for Gurbani Kirtan videos.
Downloads videos from YouTube channels, processes them through the
slide detection + audio extraction pipeline, with parallel workers
and checkpointing for resume support.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from process_video import process_video

# Defaults
DEFAULT_WORKERS = 8
DEFAULT_BASE_DIR = "/root/gurbani_kirtan"
PROGRESS_FILE = "progress.json"
FAILED_FILE = "failed_videos.json"


def load_progress(base_dir):
    """Load processing progress from checkpoint file."""
    path = os.path.join(base_dir, PROGRESS_FILE)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"completed": [], "in_progress": [], "skipped": []}


def save_progress(base_dir, progress):
    """Save processing progress to checkpoint file."""
    path = os.path.join(base_dir, PROGRESS_FILE)
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


def load_failures(base_dir):
    """Load list of failed videos for retry."""
    path = os.path.join(base_dir, FAILED_FILE)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_failure(base_dir, video_id, error_msg):
    """Append a failure record."""
    failures = load_failures(base_dir)
    failures.append({
        "video_id": video_id,
        "error": str(error_msg),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    path = os.path.join(base_dir, FAILED_FILE)
    with open(path, "w") as f:
        json.dump(failures, f, indent=2)


def fetch_video_list(channel_url, cookies_path=None):
    """Fetch list of video IDs from a YouTube channel using yt-dlp."""
    cmd = [
        "yt-dlp", "--flat-playlist", "--print", "%(id)s\t%(title)s",
        "--no-warnings",
    ]
    if cookies_path and os.path.exists(cookies_path):
        cmd.extend(["--cookies", cookies_path])

    cmd.append(channel_url)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"Warning: yt-dlp failed for {channel_url}: {result.stderr[:200]}")
        return []

    videos = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        video_id = parts[0].strip()
        title = parts[1].strip() if len(parts) > 1 else ""
        if video_id:
            videos.append({"id": video_id, "title": title})

    return videos


def download_video(video_id, download_dir, cookies_path=None):
    """Download a single video using yt-dlp. Returns path to downloaded file."""
    output_template = os.path.join(download_dir, f"{video_id}.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-warnings",
        "--js-runtimes", "node",
    ]
    if cookies_path and os.path.exists(cookies_path):
        cmd.extend(["--cookies", cookies_path])

    cmd.append(f"https://www.youtube.com/watch?v={video_id}")

    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)

    # Find the downloaded file
    for ext in ["mp4", "mkv", "webm"]:
        path = os.path.join(download_dir, f"{video_id}.{ext}")
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"Downloaded file not found for {video_id}")


def process_single_video(video_id, title, base_dir, cookies_path=None):
    """Download and process a single video. Returns result dict."""
    download_dir = os.path.join(base_dir, "downloads")
    output_dir = os.path.join(base_dir, "processed", video_id)

    # Check if already processed (checkpoint)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.exists(manifest_path):
        return {"video_id": video_id, "status": "already_done", "title": title}

    os.makedirs(download_dir, exist_ok=True)

    try:
        # Download
        video_path = download_video(video_id, download_dir, cookies_path)

        # Process
        result = process_video(video_path, output_dir)

        # Store video metadata
        meta = {
            "video_id": video_id,
            "title": title,
            "num_slides": result["num_slides"],
            "duration": result["duration"],
            "num_segments": len(result["segments"]),
        }
        meta_path = os.path.join(output_dir, "video_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # Clean up downloaded video to save disk
        if os.path.exists(video_path):
            os.remove(video_path)

        return {"video_id": video_id, "status": "success", "title": title,
                "num_slides": result["num_slides"], "duration": result["duration"]}

    except Exception as e:
        return {"video_id": video_id, "status": "failed", "title": title,
                "error": str(e)}


def run_batch(channel_urls, base_dir, cookies_path=None, workers=DEFAULT_WORKERS,
              limit=None):
    """Run the full batch processing pipeline."""
    print(f"=== Gurbani Kirtan Batch Processor ===")
    print(f"Workers: {workers}")
    print(f"Base dir: {base_dir}")
    print()

    # Load progress
    progress = load_progress(base_dir)
    completed_ids = set(progress["completed"])
    print(f"Already completed: {len(completed_ids)} videos")

    # Fetch video lists from all channels
    all_videos = []
    for url in channel_urls:
        print(f"Fetching videos from {url}...")
        videos = fetch_video_list(url, cookies_path)
        print(f"  Found {len(videos)} videos")
        all_videos.extend(videos)

    # Filter out already-completed videos
    pending = [v for v in all_videos if v["id"] not in completed_ids]
    if limit:
        pending = pending[:limit]

    print(f"\nTotal videos: {len(all_videos)}")
    print(f"Pending: {len(pending)}")
    print()

    if not pending:
        print("Nothing to process!")
        return

    # Process with parallel workers
    succeeded = 0
    failed = 0
    skipped = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for video in pending:
            future = executor.submit(
                process_single_video,
                video["id"], video["title"], base_dir, cookies_path,
            )
            futures[future] = video

        for future in as_completed(futures):
            video = futures[future]
            try:
                result = future.result()
                video_id = result["video_id"]
                status = result["status"]

                if status == "success":
                    succeeded += 1
                    progress["completed"].append(video_id)
                    save_progress(base_dir, progress)
                    dur = result.get("duration", 0)
                    print(f"  [OK] {video_id} - {result.get('title', '')[:40]} "
                          f"({result.get('num_slides', 0)} slides, {dur:.0f}s)")

                elif status == "already_done":
                    skipped += 1
                    if video_id not in progress["completed"]:
                        progress["completed"].append(video_id)
                        save_progress(base_dir, progress)
                    print(f"  [SKIP] {video_id} - already processed")

                elif status == "failed":
                    failed += 1
                    save_failure(base_dir, video_id, result.get("error", "unknown"))
                    print(f"  [FAIL] {video_id} - {result.get('error', '')[:60]}")

            except Exception as e:
                failed += 1
                save_failure(base_dir, video["id"], str(e))
                print(f"  [FAIL] {video['id']} - {str(e)[:60]}")

    elapsed = time.time() - t0
    print(f"\n=== Batch Complete ===")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


def main():
    parser = argparse.ArgumentParser(description="Batch process Gurbani Kirtan videos")
    parser.add_argument("--channels", default="channels.txt",
                        help="Path to channels.txt with one URL per line")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR,
                        help="Base working directory")
    parser.add_argument("--cookies", default="cookies.txt",
                        help="Path to YouTube cookies file")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of videos to process")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry previously failed videos")
    args = parser.parse_args()

    # Read channel URLs
    channels_path = args.channels
    if not os.path.exists(channels_path):
        print(f"Error: channels file not found: {channels_path}")
        sys.exit(1)

    with open(channels_path) as f:
        channel_urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not channel_urls:
        print("No channels found in channels.txt")
        sys.exit(1)

    # If retrying failures, clear the failed list
    if args.retry_failed:
        failures = load_failures(args.base_dir)
        if failures:
            progress = load_progress(args.base_dir)
            failed_ids = {f["video_id"] for f in failures}
            progress["completed"] = [v for v in progress["completed"] if v not in failed_ids]
            save_progress(args.base_dir, progress)
            # Clear failures file
            with open(os.path.join(args.base_dir, FAILED_FILE), "w") as f:
                json.dump([], f)
            print(f"Cleared {len(failed_ids)} failed videos for retry")

    run_batch(
        channel_urls=channel_urls,
        base_dir=args.base_dir,
        cookies_path=args.cookies,
        workers=args.workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

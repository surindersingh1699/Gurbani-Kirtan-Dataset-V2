"""
Gurbani Kirtan Video Processor
Detects slide transitions in kirtan videos and extracts timestamps + key frames.
"""

import cv2
import numpy as np
import json
import os
import subprocess
import sys
from pathlib import Path


def compute_frame_diff(frame1, frame2):
    """Compute pixel-level mean absolute difference between two frames."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    return float(np.mean(diff)) / 255.0  # Normalize to 0-1


def detect_transitions(video_path, sample_fps=2, threshold=None, min_gap_sec=3.0):
    """
    Detect slide transitions in a video.

    Args:
        video_path: Path to video file
        sample_fps: Frames to sample per second (2 is enough for slides)
        threshold: Histogram diff threshold to detect a transition
        min_gap_sec: Minimum gap between transitions to avoid duplicates

    Returns:
        List of transition timestamps in seconds
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video: {video_path}")
    print(f"FPS: {fps:.1f}, Duration: {duration:.1f}s, Total frames: {total_frames}")

    frame_interval = int(fps / sample_fps)
    transitions = []
    prev_frame = None
    diffs = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps

            if prev_frame is not None:
                diff = compute_frame_diff(prev_frame, frame)
                diffs.append((timestamp, diff))

            prev_frame = frame.copy()

        frame_idx += 1

    cap.release()

    # Auto-threshold: use median + N*std to find outlier jumps
    if diffs:
        vals = np.array([d for _, d in diffs])
        if threshold is None:
            median = np.median(vals)
            std = np.std(vals)
            threshold = median + 4 * std
            print(f"Auto threshold: {threshold:.5f} (median={median:.5f}, std={std:.5f})")

        for ts, diff in diffs:
            if diff > threshold:
                if not transitions or (ts - transitions[-1]) >= min_gap_sec:
                    transitions.append(ts)

    return transitions, diffs, duration


def extract_key_frames(video_path, timestamps, output_dir):
    """Extract a frame at each transition timestamp."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    for i, ts in enumerate(timestamps):
        # Seek slightly after transition to get the new slide
        target_frame = int((ts + 0.5) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(output_dir, f"slide_{i:03d}_{ts:.1f}s.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append({"index": i, "timestamp": ts, "frame_path": frame_path})

    cap.release()
    return frames


def extract_audio_segments(video_path, timestamps, duration, output_dir):
    """Extract audio segments between transitions using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)
    segments = []

    # Add start and end boundaries
    boundaries = [0.0] + timestamps + [duration]

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        seg_duration = end - start

        if seg_duration < 0.5:
            continue

        output_path = os.path.join(output_dir, f"segment_{i:03d}_{start:.1f}s.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ss", f"{start:.3f}",
             "-t", f"{seg_duration:.3f}", "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", output_path, "-loglevel", "error"],
            check=True,
        )

        if os.path.exists(output_path):
            segments.append({
                "index": i,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(seg_duration, 3),
                "audio_path": output_path,
            })

    return segments


def process_video(video_path, output_dir):
    """Full pipeline: detect transitions, extract frames and audio."""
    video_path = str(video_path)
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Detect transitions
    print("\n=== Step 1: Detecting slide transitions ===")
    transitions, diffs, duration = detect_transitions(video_path)
    print(f"Found {len(transitions)} transitions")
    for i, ts in enumerate(transitions):
        print(f"  Slide {i+1}: {ts:.1f}s")

    # Step 2: Extract key frames
    print("\n=== Step 2: Extracting key frames ===")
    frames_dir = os.path.join(output_dir, "frames")
    frames = extract_key_frames(video_path, transitions, frames_dir)
    print(f"Extracted {len(frames)} key frames")

    # Step 3: Extract audio segments
    print("\n=== Step 3: Extracting audio segments ===")
    audio_dir = os.path.join(output_dir, "audio")
    segments = extract_audio_segments(video_path, transitions, duration, audio_dir)
    print(f"Extracted {len(segments)} audio segments")

    # Step 4: Build result manifest
    result = {
        "video_path": video_path,
        "duration": round(duration, 3),
        "num_slides": len(transitions) + 1,
        "transitions": [round(t, 3) for t in transitions],
        "frames": frames,
        "segments": segments,
        "diffs": [(round(t, 3), round(d, 4)) for t, d in diffs],
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")

    return result


if __name__ == "__main__":
    import time
    video = sys.argv[1] if len(sys.argv) > 1 else "/root/gurbani_kirtan/test_video.mp4"
    output = sys.argv[2] if len(sys.argv) > 2 else "/root/gurbani_kirtan/processed"

    t0 = time.time()
    result = process_video(video, output)
    elapsed = time.time() - t0

    duration = result["duration"]
    ratio = duration / elapsed if elapsed > 0 else 0
    print(f"\n=== Performance ===")
    print(f"Video duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"Processing time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Speed ratio: {ratio:.1f}x realtime")
    print(f"Estimated time for 300h of kirtan: {300*3600/ratio/3600:.1f} hours")

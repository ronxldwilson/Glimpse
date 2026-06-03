"""Video analysis pipeline — extract keyframes, detect objects, build timeline."""

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .segment import segment, merge_nearby_regions, extract_region_image
from .encoder import CLIPEncoder
from .vocab import load_cached, encode_and_cache


@dataclass
class FrameResult:
    frame_idx: int
    timestamp: float
    objects: list[dict]


@dataclass
class VideoManifest:
    video_path: str
    duration: float
    total_frames: int
    keyframes_analyzed: int
    objects: dict[str, dict]  # label -> {frames, first_seen, last_seen, peak_score, path}
    timeline: list[FrameResult]
    processing_time_ms: float


def _log(msg: str):
    print(msg, flush=True)


def extract_keyframes(
    video_path: str,
    mode: str = "scene",
    scene_threshold: float = 0.3,
    fixed_fps: float = 1.0,
    max_frames: int = 200,
) -> list[tuple[float, np.ndarray]]:
    if mode == "scene":
        return _extract_scene_changes(video_path, scene_threshold, max_frames)
    else:
        return _extract_fixed_fps(video_path, fixed_fps, max_frames)


def _extract_scene_changes(
    video_path: str, threshold: float, max_frames: int
) -> list[tuple[float, np.ndarray]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"select=gt(scene\\,{threshold}),showinfo",
            "-vsync", "vfr",
            "-frame_pts", "1",
            "-q:v", "2",
            os.path.join(tmpdir, "frame_%04d.jpg"),
            "-y", "-hide_banner", "-loglevel", "info",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        stderr = result.stderr

        timestamps = []
        for line in stderr.split("\n"):
            if "pts_time:" in line:
                try:
                    pts = float(line.split("pts_time:")[1].split()[0])
                    timestamps.append(pts)
                except (ValueError, IndexError):
                    pass

        frames = []
        frame_files = sorted(Path(tmpdir).glob("frame_*.jpg"))

        if not frame_files:
            _log("  Scene detection found no frames, falling back to fixed fps")
            return _extract_fixed_fps(video_path, 1.0, max_frames)

        for i, fpath in enumerate(frame_files[:max_frames]):
            img = cv2.imread(str(fpath))
            if img is None:
                continue
            ts = timestamps[i] if i < len(timestamps) else i
            frames.append((ts, img))

        # Always include first frame
        if frames and frames[0][0] > 0.5:
            cap = cv2.VideoCapture(video_path)
            ret, first = cap.read()
            cap.release()
            if ret:
                frames.insert(0, (0.0, first))

        return frames


def _extract_fixed_fps(
    video_path: str, fps: float, max_frames: int
) -> list[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0:
        cap.release()
        return []

    step = max(1, int(video_fps / fps))
    frames = []

    for frame_idx in range(0, total, step):
        if len(frames) >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img = cap.read()
        if not ret:
            break
        ts = frame_idx / video_fps
        frames.append((ts, img))

    cap.release()
    return frames


def _frame_similarity(a: np.ndarray, b: np.ndarray) -> float:
    ha = cv2.calcHist([a], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hb = cv2.calcHist([b], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    return float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))


def analyze_video(
    video_path: str,
    mode: str = "scene",
    scene_threshold: float = 0.3,
    fixed_fps: float = 1.0,
    max_frames: int = 200,
    top_k: int = 5,
    score_threshold: float = 0.22,
    skip_similarity: float = 0.95,
) -> VideoManifest:
    t_total = time.perf_counter()

    # Get video info
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    cap.release()
    _log(f"Video: {duration:.1f}s, {total_frames} frames @ {video_fps:.0f}fps")

    # Extract keyframes
    _log(f"Extracting keyframes (mode={mode})...")
    t0 = time.perf_counter()
    keyframes = extract_keyframes(video_path, mode, scene_threshold, fixed_fps, max_frames)
    _log(f"  {len(keyframes)} keyframes in {(time.perf_counter()-t0)*1000:.0f}ms")

    if not keyframes:
        return VideoManifest(
            video_path=video_path, duration=duration, total_frames=total_frames,
            keyframes_analyzed=0, objects={}, timeline=[], processing_time_ms=0,
        )

    # Load CLIP + vocabulary
    _log("Loading CLIP model...")
    encoder = CLIPEncoder()

    cached = load_cached()
    if cached:
        vocab_labels, vocab_embeddings, vocab_paths = cached
        _log(f"  Vocabulary: {len(vocab_labels)} labels from cache")
    else:
        _log("  No cached vocabulary, encoding...")
        vocab_labels, vocab_embeddings = encode_and_cache(encoder)
        vocab_paths = {}

    # Process keyframes
    _log("Analyzing keyframes...")
    timeline = []
    prev_frame = None
    skipped = 0

    for idx, (timestamp, frame) in enumerate(keyframes):
        # Skip if too similar to previous frame
        if prev_frame is not None and _frame_similarity(frame, prev_frame) > skip_similarity:
            skipped += 1
            continue
        prev_frame = frame

        # Segment
        regions = segment(frame)
        if not regions:
            continue
        regions = merge_nearby_regions(frame, regions)

        # Encode regions
        region_images = [extract_region_image(frame, r) for r in regions]
        image_embeds = encoder.encode_images(region_images)

        # Match against vocabulary
        similarity = image_embeds @ vocab_embeddings.T

        frame_objects = []
        seen_labels = set()
        for ri in range(len(regions)):
            scores = similarity[ri]
            top_indices = np.argsort(scores)[::-1][:top_k]

            for ti in top_indices:
                score = float(scores[ti])
                if score < score_threshold:
                    continue
                label = vocab_labels[ti]
                if label in seen_labels:
                    continue
                seen_labels.add(label)

                path = vocab_paths.get(label, [label])
                frame_objects.append({
                    "label": label,
                    "score": round(score, 3),
                    "path": path,
                    "region": ri,
                    "bbox": regions[ri].bbox,
                })

        timeline.append(FrameResult(
            frame_idx=idx, timestamp=timestamp, objects=frame_objects
        ))

        analyzed = idx + 1 - skipped
        _log(f"  [{analyzed}/{len(keyframes)}] t={timestamp:.1f}s: "
             f"{len(regions)} regions, {len(frame_objects)} objects")

    # Build object manifest
    objects: dict[str, dict] = {}
    for fr in timeline:
        for obj in fr.objects:
            label = obj["label"]
            if label not in objects:
                objects[label] = {
                    "label": label,
                    "path": obj["path"],
                    "frames": [],
                    "timestamps": [],
                    "peak_score": 0.0,
                    "first_seen": fr.timestamp,
                    "last_seen": fr.timestamp,
                }
            entry = objects[label]
            entry["frames"].append(fr.frame_idx)
            entry["timestamps"].append(round(fr.timestamp, 2))
            entry["peak_score"] = max(entry["peak_score"], obj["score"])
            entry["last_seen"] = fr.timestamp

    # Sort by number of appearances
    objects = dict(sorted(objects.items(), key=lambda x: len(x[1]["frames"]), reverse=True))

    total_ms = (time.perf_counter() - t_total) * 1000
    _log(f"\nDone: {len(objects)} unique objects found across {len(timeline)} frames in {total_ms/1000:.1f}s")
    if skipped:
        _log(f"  ({skipped} similar frames skipped)")

    return VideoManifest(
        video_path=video_path,
        duration=duration,
        total_frames=total_frames,
        keyframes_analyzed=len(timeline),
        objects=objects,
        timeline=timeline,
        processing_time_ms=total_ms,
    )

"""Video analysis pipeline — extract keyframes, detect objects via MoE, build timeline."""

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .moe import detect_moe, reset_tracker, MoeDetection


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
    objects: dict[str, dict]
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


def analyze_video(
    video_path: str,
    mode: str = "scene",
    scene_threshold: float = 0.3,
    fixed_fps: float = 1.0,
    max_frames: int = 200,
) -> VideoManifest:
    t_total = time.perf_counter()

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    cap.release()
    _log(f"Video: {duration:.1f}s, {total_frames} frames @ {video_fps:.0f}fps")

    _log(f"Extracting keyframes (mode={mode})...")
    t0 = time.perf_counter()
    keyframes = extract_keyframes(video_path, mode, scene_threshold, fixed_fps, max_frames)
    _log(f"  {len(keyframes)} keyframes in {(time.perf_counter()-t0)*1000:.0f}ms")

    if not keyframes:
        return VideoManifest(
            video_path=video_path, duration=duration, total_frames=total_frames,
            keyframes_analyzed=0, objects={}, timeline=[], processing_time_ms=0,
        )

    _log("Loading models (MoE: YOLO-World + CLIP + scene + tracking)...")
    reset_tracker()
    # Warmup — first call loads all models
    t0 = time.perf_counter()
    detect_moe(keyframes[0][1])
    _log(f"  Models ready in {(time.perf_counter()-t0)*1000:.0f}ms")

    _log("Analyzing keyframes...")
    reset_tracker()
    timeline = []

    for idx, (timestamp, frame) in enumerate(keyframes):
        t0 = time.perf_counter()
        detections = detect_moe(frame)
        ms = (time.perf_counter() - t0) * 1000

        frame_objects = []
        for d in detections:
            frame_objects.append({
                "label": d.label,
                "score": round(d.confidence, 3),
                "source": d.source,
                "bbox": d.bbox,
                "scene_boosted": d.scene_boosted,
            })

        timeline.append(FrameResult(
            frame_idx=idx, timestamp=timestamp, objects=frame_objects
        ))

        labels_str = ", ".join(d.label for d in detections[:5])
        _log(f"  [{idx+1}/{len(keyframes)}] t={timestamp:.1f}s ({ms:.0f}ms): {labels_str}")

    # Build object manifest
    objects: dict[str, dict] = {}
    for fr in timeline:
        for obj in fr.objects:
            label = obj["label"]
            if label not in objects:
                objects[label] = {
                    "label": label,
                    "frames": [],
                    "timestamps": [],
                    "peak_score": 0.0,
                    "sources": set(),
                    "first_seen": fr.timestamp,
                    "last_seen": fr.timestamp,
                }
            entry = objects[label]
            entry["frames"].append(fr.frame_idx)
            entry["timestamps"].append(round(fr.timestamp, 2))
            entry["peak_score"] = max(entry["peak_score"], obj["score"])
            entry["sources"].add(obj["source"])
            entry["last_seen"] = fr.timestamp

    # Convert sets to lists for JSON
    for entry in objects.values():
        entry["sources"] = sorted(entry["sources"])

    objects = dict(sorted(objects.items(), key=lambda x: len(x[1]["frames"]), reverse=True))

    total_ms = (time.perf_counter() - t_total) * 1000
    _log(f"\nDone: {len(objects)} unique objects across {len(timeline)} frames in {total_ms/1000:.1f}s")

    return VideoManifest(
        video_path=video_path,
        duration=duration,
        total_frames=total_frames,
        keyframes_analyzed=len(timeline),
        objects=objects,
        timeline=timeline,
        processing_time_ms=total_ms,
    )

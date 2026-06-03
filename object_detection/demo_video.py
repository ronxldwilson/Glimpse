"""HTML report for video object analysis."""

import argparse
import base64
import sys
import os

import cv2
import numpy as np

from .video import analyze_video, VideoManifest


def image_to_data_uri(img: np.ndarray, quality: int = 70) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def generate_html(manifest: VideoManifest, keyframes: list, output_path: str):
    name = os.path.basename(manifest.video_path)
    s = manifest

    # Build object cards
    obj_html = ""
    for label, info in list(s.objects.items())[:50]:
        n_frames = len(info["frames"])
        pct = n_frames / max(s.keyframes_analyzed, 1) * 100
        bar_pct = min(pct, 100)
        time_range = f"{info['first_seen']:.1f}s - {info['last_seen']:.1f}s"
        sources = ", ".join(info.get("sources", []))

        obj_html += f"""<div class="obj-card">
  <div class="obj-name">{label}</div>
  <div class="obj-stats">
    <span class="obj-frames">{n_frames} frame{"s" if n_frames != 1 else ""}</span>
    <span class="obj-time">{time_range}</span>
    <span class="obj-score">{info['peak_score']:.3f}</span>
  </div>
  <div class="obj-source">{sources}</div>
  <div class="obj-bar"><div class="obj-bar-fill" style="width:{bar_pct:.0f}%"></div></div>
</div>"""

    # Build timeline frames
    timeline_html = ""
    for i, fr in enumerate(s.timeline):
        if i < len(keyframes):
            thumb_uri = image_to_data_uri(keyframes[i], quality=50)
        else:
            thumb_uri = ""

        obj_labels = [o["label"] for o in fr.objects[:5]]
        labels_str = ", ".join(obj_labels) if obj_labels else "no objects"
        n_obj = len(fr.objects)

        timeline_html += f"""<div class="tl-frame" onclick="showFrame({i})">
  <img src="{thumb_uri}" loading="lazy">
  <div class="tl-info">
    <div class="tl-time">{fr.timestamp:.1f}s</div>
    <div class="tl-objects">{n_obj} objects</div>
    <div class="tl-labels">{labels_str}</div>
  </div>
</div>"""

    # Build frame detail panels (hidden, shown on click)
    frames_json_items = []
    for i, fr in enumerate(s.timeline):
        objs = [{"label": o["label"], "score": o["score"],
                 "source": o.get("source", ""),
                 "path": o["label"]}
                for o in fr.objects[:15]]
        frames_json_items.append({
            "timestamp": round(fr.timestamp, 2),
            "objects": objs,
        })
    frames_json = str(frames_json_items).replace("'", '"')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Video Analysis — {name}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f0f0f; color: #e0e0e0; }}

  .layout {{ display: flex; min-height: 100vh; }}

  .sidebar {{ width: 320px; background: #141414; border-right: 1px solid #2a2a2a; flex-shrink: 0; position: sticky; top: 0; height: 100vh; overflow-y: auto; padding: 16px; }}
  .sidebar h1 {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 2px; }}
  .sidebar h2 {{ font-size: 0.75rem; color: #666; margin-bottom: 16px; }}

  .stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 20px; }}
  .stat {{ background: #1a1a1a; border: 1px solid #333; border-radius: 6px; padding: 8px 10px; }}
  .stat-value {{ font-size: 1.1rem; font-weight: 600; color: #4fc3f7; }}
  .stat-label {{ font-size: 0.6rem; color: #888; margin-top: 1px; }}

  .section-title {{ font-size: 0.7rem; text-transform: uppercase; color: #555; letter-spacing: 0.05em; margin: 16px 0 8px; }}

  .obj-card {{ background: #1a1a1a; border: 1px solid #333; border-radius: 6px; padding: 8px 10px; margin-bottom: 5px; }}
  .obj-name {{ font-weight: 600; font-size: 0.85rem; color: #4fc3f7; }}
  .obj-path {{ font-size: 0.65rem; color: #666; margin-bottom: 4px; }}
  .obj-stats {{ display: flex; gap: 8px; font-size: 0.7rem; color: #888; margin-bottom: 4px; }}
  .obj-frames {{ color: #aaa; }}
  .obj-score {{ color: #4fc3f7; }}
  .obj-source {{ font-size: 0.6rem; color: #555; margin-bottom: 3px; }}
  .obj-bar {{ height: 3px; background: #333; border-radius: 2px; overflow: hidden; }}
  .obj-bar-fill {{ height: 100%; background: linear-gradient(90deg, #4fc3f7, #29b6f6); border-radius: 2px; }}

  .main {{ flex: 1; padding: 24px; overflow-y: auto; }}

  .tl-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }}
  .tl-frame {{ background: #1a1a1a; border: 1px solid #333; border-radius: 8px; overflow: hidden; cursor: pointer; transition: border-color 0.15s; }}
  .tl-frame:hover {{ border-color: #4fc3f7; }}
  .tl-frame.active {{ border-color: #4fc3f7; box-shadow: 0 0 0 1px #4fc3f7; }}
  .tl-frame img {{ width: 100%; aspect-ratio: 16/9; object-fit: cover; }}
  .tl-info {{ padding: 8px 10px; }}
  .tl-time {{ font-weight: 600; font-size: 0.85rem; color: #4fc3f7; }}
  .tl-objects {{ font-size: 0.7rem; color: #888; }}
  .tl-labels {{ font-size: 0.7rem; color: #aaa; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}

  .frame-detail {{ background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 16px; margin-bottom: 20px; display: none; }}
  .frame-detail.active {{ display: block; }}
  .frame-detail h3 {{ font-size: 0.9rem; margin-bottom: 10px; color: #4fc3f7; }}
  .frame-det {{ font-size: 0.8rem; padding: 4px 0; border-bottom: 1px solid #222; display: flex; justify-content: space-between; }}
  .frame-det:last-child {{ border: none; }}
  .frame-det-label {{ color: #e0e0e0; }}
  .frame-det-score {{ color: #4fc3f7; }}
  .frame-det-path {{ color: #666; font-size: 0.7rem; }}
</style>
</head>
<body>
<div class="layout">

<div class="sidebar">
  <h1>{name}</h1>
  <h2>Video Object Analysis</h2>

  <div class="stats">
    <div class="stat"><div class="stat-value">{s.duration:.1f}s</div><div class="stat-label">Duration</div></div>
    <div class="stat"><div class="stat-value">{s.keyframes_analyzed}</div><div class="stat-label">Frames Analyzed</div></div>
    <div class="stat"><div class="stat-value">{len(s.objects)}</div><div class="stat-label">Objects Found</div></div>
    <div class="stat"><div class="stat-value">{s.processing_time_ms/1000:.1f}s</div><div class="stat-label">Processing Time</div></div>
  </div>

  <div class="section-title">Objects Found</div>
  {obj_html}
</div>

<div class="main">
  <div class="frame-detail" id="frame-detail">
    <h3 id="detail-title">Click a frame to see details</h3>
    <div id="detail-objects"></div>
  </div>

  <div class="section-title" style="margin-bottom:12px;">Timeline ({s.keyframes_analyzed} keyframes)</div>
  <div class="tl-grid">
    {timeline_html}
  </div>
</div>

</div>

<script>
const frameData = {frames_json};

function showFrame(idx) {{
  document.querySelectorAll('.tl-frame').forEach((f, i) => f.classList.toggle('active', i === idx));

  const detail = document.getElementById('frame-detail');
  const title = document.getElementById('detail-title');
  const objs = document.getElementById('detail-objects');
  detail.classList.add('active');

  const f = frameData[idx];
  title.textContent = 'Frame at ' + f.timestamp + 's — ' + f.objects.length + ' objects';

  objs.innerHTML = f.objects.map(o =>
    '<div class="frame-det">' +
      '<div><span class="frame-det-label">' + o.label + '</span> ' +
      '<span class="frame-det-path">' + (o.source || '') + '</span></div>' +
      '<span class="frame-det-score">' + o.score.toFixed(3) + '</span>' +
    '</div>'
  ).join('') || '<div class="frame-det" style="color:#666">No objects detected</div>';
}}
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Saved report to {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Analyze video for objects")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output", default="video_report.html", help="Output HTML path")
    parser.add_argument("--mode", choices=["scene", "fixed"], default="scene",
                        help="Frame extraction mode")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second (fixed mode)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Scene change threshold")
    parser.add_argument("--max-frames", type=int, default=200, help="Max frames to analyze")
    args = parser.parse_args()

    manifest = analyze_video(
        args.video,
        mode=args.mode,
        scene_threshold=args.threshold,
        fixed_fps=args.fps,
        max_frames=args.max_frames,
    )

    # Re-extract frames for thumbnails
    from .video import extract_keyframes
    keyframes_raw = extract_keyframes(
        args.video, args.mode, args.threshold, args.fps, args.max_frames
    )
    keyframe_images = [img for _, img in keyframes_raw]

    generate_html(manifest, keyframe_images, args.output)


if __name__ == "__main__":
    main()

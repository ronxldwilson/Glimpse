import argparse
import base64
import json
import os
import time
import sys

import cv2
import numpy as np

from .segment import segment, merge_nearby_regions, extract_region_image
from .encoder import CLIPEncoder
from .detector import ObjectDetector, _nms_by_label


def image_to_data_uri(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def draw_segmentation_overlay(image: np.ndarray, regions) -> np.ndarray:
    vis = image.copy()
    for i, region in enumerate(regions):
        color = tuple(int(c) for c in np.random.RandomState(i * 7 + 3).randint(80, 255, 3))
        contours, _ = cv2.findContours(region.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
        x, y, w, h = region.bbox
        cv2.putText(vis, f"R{i}", (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return vis


def draw_detections_overlay(image: np.ndarray, detections) -> np.ndarray:
    vis = image.copy()
    for d in detections:
        x, y, w, h = d.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 2)
        label = f"{d.label} ({d.score:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x, y - th - 8), (x + tw + 4, y), (0, 200, 0), -1)
        cv2.putText(vis, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vis


def process_image(image_path: str, labels: list[str], encoder: CLIPEncoder, text_embeds: np.ndarray):
    image = cv2.imread(image_path)
    if image is None:
        print(f"  Skipping {image_path}: cannot read")
        return None

    name = os.path.basename(image_path)
    print(f"  [{name}] Segmenting...")
    t0 = time.perf_counter()
    raw_regions = segment(image)
    seg_ms = (time.perf_counter() - t0) * 1000

    regions = merge_nearby_regions(image, raw_regions)
    print(f"  [{name}] {len(raw_regions)} regions + {len(regions) - len(raw_regions)} merged in {seg_ms:.0f}ms")

    print(f"  [{name}] Encoding {len(regions)} regions...")
    t0 = time.perf_counter()
    region_images = [extract_region_image(image, r) for r in regions]
    image_embeds = encoder.encode_images(region_images)
    encode_ms = (time.perf_counter() - t0) * 1000

    similarity = image_embeds @ text_embeds.T

    from .detector import Detection
    all_detections = []
    region_data = []
    for i, region in enumerate(regions):
        scores = similarity[i]
        top_indices = np.argsort(scores)[::-1][:5]
        top_labels = [(labels[idx], float(scores[idx])) for idx in top_indices]

        region_dets = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= 0.15:
                det = Detection(label=labels[idx], score=score, bbox=region.bbox, region_index=i)
                all_detections.append(det)
                region_dets.append(det)

        region_data.append({
            "index": i,
            "bbox": region.bbox,
            "area": region.area,
            "image_uri": image_to_data_uri(region_images[i]),
            "top_labels": top_labels,
            "detections": [{"label": d.label, "score": d.score} for d in region_dets],
        })

    final_detections = _nms_by_label(all_detections, iou_threshold=0.5)

    return {
        "name": name,
        "original_uri": image_to_data_uri(image),
        "seg_uri": image_to_data_uri(draw_segmentation_overlay(image, regions)),
        "det_uri": image_to_data_uri(draw_detections_overlay(image, final_detections)),
        "regions": region_data,
        "detections": [
            {"label": d.label, "score": round(d.score, 3), "bbox": d.bbox, "region": d.region_index}
            for d in final_detections
        ],
        "stats": {
            "image_size": f"{image.shape[1]}x{image.shape[0]}",
            "num_regions": len(regions),
            "segmentation_ms": round(seg_ms),
            "encoding_ms": round(encode_ms),
            "num_detections": len(final_detections),
        },
    }


def generate_html(image_paths: list[str], labels: list[str], output_path: str):
    print("Loading CLIP model...")
    t0 = time.perf_counter()
    encoder = CLIPEncoder()
    print(f"  Loaded in {(time.perf_counter() - t0) * 1000:.0f}ms")

    print("Encoding labels...")
    text_embeds = encoder.encode_labels(labels)

    results = []
    for path in image_paths:
        r = process_image(path, labels, encoder, text_embeds)
        if r:
            results.append(r)

    if not results:
        print("No images processed.")
        sys.exit(1)

    html = _build_html(results, labels)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Saved demo to {output_path}")


def _build_html(results: list[dict], labels: list[str]):
    nav_items = ""
    pages = ""
    for idx, r in enumerate(results):
        active = "active" if idx == 0 else ""
        nav_items += f'<div class="nav-item {active}" onclick="switchImage({idx})">{r["name"]}</div>\n'

        det_html = "".join(_det_card_html(d) for d in r["detections"]) or '<p style="color:#666;">No detections above threshold.</p>'
        region_html = "".join(_region_card_html(reg) for reg in r["regions"])
        s = r["stats"]

        pages += f"""
<div class="page {active}" id="page-{idx}">
  <div class="stats">
    <div class="stat"><div class="stat-value">{s['image_size']}</div><div class="stat-label">Image Size</div></div>
    <div class="stat"><div class="stat-value">{s['num_regions']}</div><div class="stat-label">Regions</div></div>
    <div class="stat"><div class="stat-value">{s['segmentation_ms']}ms</div><div class="stat-label">Segmentation</div></div>
    <div class="stat"><div class="stat-value">{s['encoding_ms']}ms</div><div class="stat-label">CLIP Encoding</div></div>
    <div class="stat"><div class="stat-value">{s['num_detections']}</div><div class="stat-label">Detections</div></div>
  </div>

  <div class="tabs">
    <div class="tab active" onclick="switchTab(this, {idx}, 0)">Original</div>
    <div class="tab" onclick="switchTab(this, {idx}, 1)">Segments</div>
    <div class="tab" onclick="switchTab(this, {idx}, 2)">Detections</div>
  </div>
  <div class="view active" id="view-{idx}-0"><img src="{r['original_uri']}"></div>
  <div class="view" id="view-{idx}-1"><img src="{r['seg_uri']}"></div>
  <div class="view" id="view-{idx}-2"><img src="{r['det_uri']}"></div>

  <div class="detections">
    <h2>Detections</h2>
    {det_html}
  </div>

  <div class="regions">
    <h2>All Regions (on white background)</h2>
    <div class="region-grid">
      {region_html}
    </div>
  </div>
</div>
"""

    labels_html = "".join(f'<span class="label-chip">{l}</span>' for l in labels)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Object Detection — FastSAM + CLIP</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f0f0f; color: #e0e0e0; }}

  .layout {{ display: flex; min-height: 100vh; }}

  .sidebar {{ width: 220px; background: #141414; border-right: 1px solid #2a2a2a; padding: 16px 0; flex-shrink: 0; position: sticky; top: 0; height: 100vh; overflow-y: auto; }}
  .sidebar h1 {{ font-size: 1rem; font-weight: 600; padding: 0 16px 4px; }}
  .sidebar h2 {{ font-size: 0.75rem; font-weight: 400; color: #666; padding: 0 16px 12px; }}
  .nav-item {{ padding: 10px 16px; cursor: pointer; font-size: 0.85rem; border-left: 3px solid transparent; transition: all 0.15s; }}
  .nav-item:hover {{ background: #1a1a1a; }}
  .nav-item.active {{ background: #1a2a3a; border-left-color: #4fc3f7; color: #4fc3f7; }}

  .sidebar .labels-section {{ padding: 12px 16px; border-top: 1px solid #2a2a2a; margin-top: 12px; }}
  .sidebar .labels-section h3 {{ font-size: 0.7rem; text-transform: uppercase; color: #555; margin-bottom: 8px; letter-spacing: 0.05em; }}

  .main {{ flex: 1; padding: 24px; overflow-y: auto; }}

  .stats {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }}
  .stat {{ background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 10px 14px; }}
  .stat-value {{ font-size: 1.2rem; font-weight: 600; color: #4fc3f7; }}
  .stat-label {{ font-size: 0.7rem; color: #888; margin-top: 2px; }}

  h2 {{ font-size: 1rem; font-weight: 500; margin-bottom: 12px; color: #aaa; }}

  .tabs {{ display: flex; gap: 4px; margin-bottom: 12px; }}
  .tab {{ padding: 7px 14px; background: #1a1a1a; border: 1px solid #333; border-radius: 6px 6px 0 0; cursor: pointer; font-size: 0.8rem; }}
  .tab.active {{ background: #2a2a2a; border-bottom-color: #2a2a2a; color: #4fc3f7; }}
  .view {{ display: none; background: #2a2a2a; border-radius: 0 8px 8px 8px; padding: 12px; margin-bottom: 20px; }}
  .view.active {{ display: block; }}
  .view img {{ max-width: 100%; border-radius: 4px; }}

  .page {{ display: none; }}
  .page.active {{ display: block; }}

  .detections {{ margin-bottom: 24px; }}
  .det-card {{ background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 10px 14px; margin-bottom: 6px; display: flex; align-items: center; gap: 10px; }}
  .det-label {{ font-weight: 600; font-size: 0.9rem; min-width: 110px; }}
  .det-score {{ font-size: 0.85rem; color: #4fc3f7; min-width: 50px; }}
  .det-bar {{ flex: 1; height: 5px; background: #333; border-radius: 3px; overflow: hidden; }}
  .det-bar-fill {{ height: 100%; background: linear-gradient(90deg, #4fc3f7, #29b6f6); border-radius: 3px; }}
  .det-meta {{ font-size: 0.7rem; color: #666; min-width: 80px; text-align: right; }}

  .region-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; }}
  .region-card {{ background: #1a1a1a; border: 1px solid #333; border-radius: 8px; overflow: hidden; }}
  .region-card img {{ width: 100%; aspect-ratio: 1; object-fit: contain; background: #fff; }}
  .region-info {{ padding: 8px 10px; }}
  .region-title {{ font-weight: 600; font-size: 0.8rem; margin-bottom: 4px; }}
  .region-scores {{ font-size: 0.7rem; color: #888; }}
  .region-scores span {{ display: block; margin-bottom: 1px; }}
  .region-scores .match {{ color: #4fc3f7; font-weight: 500; }}

  .labels-list {{ display: flex; gap: 5px; flex-wrap: wrap; }}
  .label-chip {{ background: #1a3a4a; color: #4fc3f7; padding: 3px 8px; border-radius: 10px; font-size: 0.7rem; }}
</style>
</head>
<body>
<div class="layout">

<div class="sidebar">
  <h1>FastSAM + CLIP</h1>
  <h2>{len(results)} image{"s" if len(results) != 1 else ""}</h2>
  {nav_items}
  <div class="labels-section">
    <h3>Search Labels</h3>
    <div class="labels-list">{labels_html}</div>
  </div>
</div>

<div class="main">
  {pages}
</div>

</div>

<script>
function switchImage(idx) {{
  document.querySelectorAll('.nav-item').forEach((n, i) => n.classList.toggle('active', i === idx));
  document.querySelectorAll('.page').forEach((p, i) => p.classList.toggle('active', i === idx));
}}

function switchTab(el, pageIdx, viewIdx) {{
  const page = document.getElementById('page-' + pageIdx);
  page.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  for (let i = 0; i < 3; i++) {{
    const v = document.getElementById('view-' + pageIdx + '-' + i);
    if (v) v.classList.toggle('active', i === viewIdx);
  }}
}}
</script>
</body>
</html>"""


def _det_card_html(d):
    pct = min(d['score'] * 100 / 0.4, 100)
    return f"""<div class="det-card">
  <div class="det-label">{d['label']}</div>
  <div class="det-score">{d['score']:.3f}</div>
  <div class="det-bar"><div class="det-bar-fill" style="width:{pct:.0f}%"></div></div>
  <div class="det-meta">R{d['region']} ({d['bbox'][0]},{d['bbox'][1]},{d['bbox'][2]},{d['bbox'][3]})</div>
</div>"""


def _region_card_html(r):
    scores_html = ""
    for label, score in r['top_labels'][:5]:
        cls = "match" if score >= 0.15 else ""
        scores_html += f'<span class="{cls}">{label}: {score:.3f}</span>'
    return f"""<div class="region-card">
  <img src="{r['image_uri']}">
  <div class="region-info">
    <div class="region-title">R{r['index']} — {r['area']}px</div>
    <div class="region-scores">{scores_html}</div>
  </div>
</div>"""


def main():
    parser = argparse.ArgumentParser(description="Generate HTML demo for object detection")
    parser.add_argument("images", nargs="+", help="Path(s) to image files")
    parser.add_argument("--labels", nargs="+", required=True, help="Labels to search for")
    parser.add_argument("--output", default="demo.html", help="Output HTML path")
    args = parser.parse_args()
    generate_html(args.images, args.labels, args.output)


if __name__ == "__main__":
    main()

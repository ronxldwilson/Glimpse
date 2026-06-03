"""HTML demo for hierarchical object discovery — no labels needed."""

import argparse
import base64
import os
import time
import sys

import cv2
import numpy as np

from .segment import segment, merge_nearby_regions, extract_region_image
from .encoder import CLIPEncoder
from .taxonomy import build_taxonomy, encode_taxonomy
from .discover import discover_objects, Discovery


def image_to_data_uri(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def draw_discovery_overlay(image: np.ndarray, discoveries: list[Discovery]) -> np.ndarray:
    vis = image.copy()
    colors = [
        (78, 195, 247), (129, 199, 132), (255, 183, 77), (240, 98, 146),
        (149, 117, 205), (77, 208, 225), (255, 138, 101), (174, 213, 129),
        (255, 213, 79), (144, 164, 174),
    ]
    for i, d in enumerate(discoveries):
        color = colors[i % len(colors)]
        x, y, w, h = d.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        label = f"{' > '.join(d.path)} ({d.confidence:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(vis, (x, y - th - 8), (x + tw + 4, y), color, -1)
        cv2.putText(vis, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return vis


def process_image(image_path: str, encoder: CLIPEncoder, taxonomy, text_embeds_time: float):
    image = cv2.imread(image_path)
    if image is None:
        return None

    name = os.path.basename(image_path)

    t0 = time.perf_counter()
    raw_regions = segment(image)
    regions = merge_nearby_regions(image, raw_regions)
    seg_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    region_images = [extract_region_image(image, r) for r in regions]
    image_embeds = encoder.encode_images(region_images)
    encode_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    discoveries = discover_objects(image_embeds, regions, taxonomy, top_per_region=1)
    discover_ms = (time.perf_counter() - t0) * 1000

    print(f"  [{name}] {len(regions)} regions, {len(discoveries)} discoveries in "
          f"{seg_ms:.0f}+{encode_ms:.0f}+{discover_ms:.1f}ms")

    region_data = []
    disc_by_region = {d.region_index: d for d in discoveries}
    for i, region in enumerate(regions):
        d = disc_by_region.get(i)
        region_data.append({
            "index": i,
            "area": region.area,
            "image_uri": image_to_data_uri(region_images[i]),
            "discovery": {
                "path": d.path,
                "scores": [round(s, 3) for s in d.scores],
                "label": d.label,
                "confidence": round(d.confidence, 3),
            } if d else None,
        })

    return {
        "name": name,
        "original_uri": image_to_data_uri(image),
        "seg_uri": image_to_data_uri(draw_discovery_overlay(image, discoveries)),
        "regions": region_data,
        "discoveries": [
            {
                "path": d.path,
                "scores": [round(s, 3) for s in d.scores],
                "label": d.label,
                "confidence": round(d.confidence, 3),
                "region": d.region_index,
                "bbox": d.bbox,
            }
            for d in discoveries
        ],
        "stats": {
            "image_size": f"{image.shape[1]}x{image.shape[0]}",
            "num_regions": len(regions),
            "segmentation_ms": round(seg_ms),
            "encoding_ms": round(encode_ms),
            "discover_ms": round(discover_ms, 1),
            "num_discoveries": len(discoveries),
        },
    }


def generate_html(image_paths: list[str], output_path: str):
    print("Building taxonomy...")
    taxonomy = build_taxonomy()
    print(f"  {taxonomy.count()} nodes, depth {taxonomy.depth()}")

    print("Loading CLIP + encoding taxonomy...")
    t0 = time.perf_counter()
    encoder = CLIPEncoder()
    encode_taxonomy(taxonomy, encoder)
    tax_ms = (time.perf_counter() - t0) * 1000
    print(f"  Done in {tax_ms:.0f}ms (one-time cost)")

    results = []
    for path in image_paths:
        r = process_image(path, encoder, taxonomy, tax_ms)
        if r:
            results.append(r)

    if not results:
        print("No images processed.")
        sys.exit(1)

    html = _build_html(results, taxonomy.count())
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Saved to {output_path}")


def _build_html(results: list[dict], taxonomy_size: int):
    nav_items = ""
    pages = ""

    for idx, r in enumerate(results):
        active = "active" if idx == 0 else ""
        nav_items += f'<div class="nav-item {active}" onclick="switchImage({idx})">{r["name"]}</div>\n'

        s = r["stats"]
        disc_html = ""
        for d in r["discoveries"]:
            path_html = " &rsaquo; ".join(
                f'<span class="path-step">{step}</span>' for step in d["path"]
            )
            scores_html = " &rsaquo; ".join(f'{s:.3f}' for s in d["scores"])
            pct = min(d["confidence"] * 100 / 0.4, 100)
            disc_html += f"""<div class="disc-card">
  <div class="disc-path">{path_html}</div>
  <div class="disc-score">{d['confidence']:.3f}</div>
  <div class="disc-bar"><div class="disc-bar-fill" style="width:{pct:.0f}%"></div></div>
  <div class="disc-meta">R{d['region']}</div>
</div>"""

        region_html = ""
        for reg in r["regions"]:
            d = reg["discovery"]
            if d:
                path_str = " &rsaquo; ".join(d["path"])
                label_html = f'<div class="region-discovery">{path_str} <span>{d["confidence"]:.3f}</span></div>'
            else:
                label_html = '<div class="region-discovery dim">unclassified</div>'
            region_html += f"""<div class="region-card">
  <img src="{reg['image_uri']}">
  <div class="region-info">
    <div class="region-title">R{reg['index']} — {reg['area']}px</div>
    {label_html}
  </div>
</div>"""

        pages += f"""
<div class="page {active}" id="page-{idx}">
  <div class="stats">
    <div class="stat"><div class="stat-value">{s['image_size']}</div><div class="stat-label">Image Size</div></div>
    <div class="stat"><div class="stat-value">{s['num_regions']}</div><div class="stat-label">Regions</div></div>
    <div class="stat"><div class="stat-value">{s['segmentation_ms']}ms</div><div class="stat-label">Segmentation</div></div>
    <div class="stat"><div class="stat-value">{s['encoding_ms']}ms</div><div class="stat-label">CLIP Encoding</div></div>
    <div class="stat"><div class="stat-value">{s['discover_ms']}ms</div><div class="stat-label">Tree Walk</div></div>
    <div class="stat"><div class="stat-value">{s['num_discoveries']}</div><div class="stat-label">Discovered</div></div>
  </div>

  <div class="tabs">
    <div class="tab active" onclick="switchTab(this, {idx}, 0)">Original</div>
    <div class="tab" onclick="switchTab(this, {idx}, 1)">Discoveries</div>
  </div>
  <div class="view active" id="view-{idx}-0"><img src="{r['original_uri']}"></div>
  <div class="view" id="view-{idx}-1"><img src="{r['seg_uri']}"></div>

  <div class="detections">
    <h2>Discovered Objects</h2>
    {disc_html or '<p class="dim">No objects discovered.</p>'}
  </div>

  <div class="regions">
    <h2>All Regions</h2>
    <div class="region-grid">{region_html}</div>
  </div>
</div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Object Discovery — FastSAM + CLIP + WordNet</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f0f0f; color: #e0e0e0; }}
  .layout {{ display: flex; min-height: 100vh; }}
  .sidebar {{ width: 220px; background: #141414; border-right: 1px solid #2a2a2a; padding: 16px 0; flex-shrink: 0; position: sticky; top: 0; height: 100vh; overflow-y: auto; }}
  .sidebar h1 {{ font-size: 1rem; font-weight: 600; padding: 0 16px 2px; }}
  .sidebar h2 {{ font-size: 0.7rem; font-weight: 400; color: #666; padding: 0 16px 4px; }}
  .sidebar .tax-info {{ font-size: 0.65rem; color: #444; padding: 0 16px 12px; }}
  .nav-item {{ padding: 10px 16px; cursor: pointer; font-size: 0.85rem; border-left: 3px solid transparent; transition: all 0.15s; }}
  .nav-item:hover {{ background: #1a1a1a; }}
  .nav-item.active {{ background: #1a2a3a; border-left-color: #4fc3f7; color: #4fc3f7; }}
  .main {{ flex: 1; padding: 24px; overflow-y: auto; }}
  .stats {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }}
  .stat {{ background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 10px 14px; }}
  .stat-value {{ font-size: 1.1rem; font-weight: 600; color: #4fc3f7; }}
  .stat-label {{ font-size: 0.65rem; color: #888; margin-top: 2px; }}
  h2 {{ font-size: 1rem; font-weight: 500; margin-bottom: 12px; color: #aaa; }}
  .tabs {{ display: flex; gap: 4px; margin-bottom: 12px; }}
  .tab {{ padding: 7px 14px; background: #1a1a1a; border: 1px solid #333; border-radius: 6px 6px 0 0; cursor: pointer; font-size: 0.8rem; }}
  .tab.active {{ background: #2a2a2a; border-bottom-color: #2a2a2a; color: #4fc3f7; }}
  .view {{ display: none; background: #2a2a2a; border-radius: 0 8px 8px 8px; padding: 12px; margin-bottom: 20px; }}
  .view.active {{ display: block; }}
  .view img {{ max-width: 100%; border-radius: 4px; }}
  .page {{ display: none; }}
  .page.active {{ display: block; }}
  .disc-card {{ background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 10px 14px; margin-bottom: 6px; display: flex; align-items: center; gap: 10px; }}
  .disc-path {{ flex: 1; font-size: 0.85rem; }}
  .path-step {{ color: #e0e0e0; }}
  .disc-path .path-step:last-child {{ color: #4fc3f7; font-weight: 600; }}
  .disc-score {{ font-size: 0.85rem; color: #4fc3f7; font-weight: 500; min-width: 45px; }}
  .disc-bar {{ width: 80px; height: 5px; background: #333; border-radius: 3px; overflow: hidden; }}
  .disc-bar-fill {{ height: 100%; background: linear-gradient(90deg, #4fc3f7, #29b6f6); border-radius: 3px; }}
  .disc-meta {{ font-size: 0.7rem; color: #666; min-width: 30px; text-align: right; }}
  .region-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; }}
  .region-card {{ background: #1a1a1a; border: 1px solid #333; border-radius: 8px; overflow: hidden; }}
  .region-card img {{ width: 100%; aspect-ratio: 1; object-fit: contain; background: #fff; }}
  .region-info {{ padding: 8px 10px; }}
  .region-title {{ font-weight: 600; font-size: 0.8rem; margin-bottom: 4px; }}
  .region-discovery {{ font-size: 0.75rem; color: #4fc3f7; }}
  .region-discovery span {{ color: #888; margin-left: 4px; }}
  .dim {{ color: #555; }}
</style>
</head>
<body>
<div class="layout">
<div class="sidebar">
  <h1>Object Discovery</h1>
  <h2>FastSAM + CLIP + WordNet</h2>
  <div class="tax-info">{taxonomy_size} concepts in taxonomy</div>
  {nav_items}
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
  for (let i = 0; i < 2; i++) {{
    const v = document.getElementById('view-' + pageIdx + '-' + i);
    if (v) v.classList.toggle('active', i === viewIdx);
  }}
}}
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Discover objects in images — no labels needed")
    parser.add_argument("images", nargs="+", help="Path(s) to image files")
    parser.add_argument("--output", default="discover.html", help="Output HTML path")
    args = parser.parse_args()
    generate_html(args.images, args.output)


if __name__ == "__main__":
    main()

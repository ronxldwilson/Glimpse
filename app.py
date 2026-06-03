import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os

st.set_page_config(page_title="Glimpse", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0f0f0f; }
    section[data-testid="stSidebar"] { background-color: #141414; min-width: 320px; }
    .block-container { padding: 1rem 1.5rem; }
    h1, h2, h3 { margin-bottom: 0.3rem !important; }

    .s { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 6px; padding: 8px 12px; margin-bottom: 6px; }
    .sv { font-size: 1.2rem; font-weight: 700; color: #4fc3f7; line-height: 1.2; }
    .sl { font-size: 0.6rem; color: #666; text-transform: uppercase; letter-spacing: 0.04em; }

    .pill { display: inline-block; background: #1a2a3a; color: #4fc3f7; padding: 2px 10px;
            border-radius: 12px; margin: 2px; font-size: 0.78rem; font-weight: 500; }
    .pill-dim { background: #1a1a1a; color: #888; }

    .ocard { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 6px;
             padding: 6px 10px; margin-bottom: 4px; }
    .oname { font-weight: 600; font-size: 0.82rem; color: #4fc3f7; }
    .ometa { font-size: 0.65rem; color: #666; }
    .obar { height: 2px; background: #333; border-radius: 1px; margin-top: 3px; }
    .obar-fill { height: 2px; background: #4fc3f7; border-radius: 1px; }

    .det-row { display: flex; justify-content: space-between; align-items: center;
               padding: 3px 8px; font-size: 0.78rem; border-bottom: 1px solid #1a1a1a; }
    .det-row:last-child { border: none; }
    .det-label { font-weight: 500; color: #e0e0e0; }
    .det-src { font-size: 0.6rem; color: #555; margin-left: 4px; }
    .det-score { color: #4fc3f7; font-weight: 500; }

    .frame-thumb { cursor: pointer; border: 2px solid transparent; border-radius: 6px;
                   transition: border-color 0.15s; }
    .frame-thumb:hover { border-color: #4fc3f7; }

    .section-hdr { font-size: 0.65rem; text-transform: uppercase; color: #555;
                   letter-spacing: 0.05em; margin: 12px 0 6px; }
</style>
""", unsafe_allow_html=True)


COLORS = [
    (78, 195, 247), (129, 199, 132), (255, 183, 77), (240, 98, 146),
    (149, 117, 205), (77, 208, 225), (255, 138, 101), (174, 213, 129),
    (255, 213, 79), (144, 164, 174),
]


def draw_boxes(image, detections):
    vis = image.copy()
    for i, d in enumerate(detections):
        bbox = d.get("bbox")
        if not bbox:
            continue
        x, y, w, h = bbox
        c = COLORS[i % len(COLORS)]
        bgr = (c[2], c[1], c[0])
        cv2.rectangle(vis, (x, y), (x + w, y + h), bgr, 2)
        label = f"{d['label']} {d['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(vis, (x, y - th - 6), (x + tw + 4, y), bgr, -1)
        cv2.putText(vis, label, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return vis


@st.cache_resource
def load_models():
    from object_detection.moe import detect_moe
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    detect_moe(dummy, use_scene=False, use_tracking=False)
    return True


def stat_box(value, label):
    return f'<div class="s"><div class="sv">{value}</div><div class="sl">{label}</div></div>'


def main():
    with st.spinner("Loading models..."):
        load_models()

    st.sidebar.markdown("# Glimpse")
    st.sidebar.caption("YOLO-World + CLIP + Scene + Tracking")
    st.sidebar.markdown("---")

    mode = st.sidebar.radio("Mode", ["Video", "Image"], label_visibility="collapsed")

    if mode == "Video":
        video_mode()
    else:
        image_mode()


def image_mode():
    uploaded = st.sidebar.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])
    if not uploaded:
        st.markdown("### Upload an image to analyze")
        return

    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Cannot read image")
        return

    from object_detection.moe import detect_moe
    t0 = time.perf_counter()
    detections = detect_moe(image)
    ms = (time.perf_counter() - t0) * 1000

    # Sidebar: stats + object list
    h, w = image.shape[:2]
    st.sidebar.markdown(stat_box(f"{w}x{h}", "Image Size"), unsafe_allow_html=True)
    st.sidebar.markdown(stat_box(f"{ms:.0f}ms", "Processing"), unsafe_allow_html=True)
    st.sidebar.markdown(stat_box(len(detections), "Objects"), unsafe_allow_html=True)

    st.sidebar.markdown('<div class="section-hdr">Detections</div>', unsafe_allow_html=True)
    for d in detections:
        st.sidebar.markdown(
            f'<div class="ocard"><span class="oname">{d.label}</span> '
            f'<span class="ometa">{d.confidence:.3f} | {d.source}</span></div>',
            unsafe_allow_html=True
        )

    # Main: side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Original")
    with col2:
        annotated = draw_boxes(image, [{"label": d.label, "score": d.confidence, "bbox": d.bbox} for d in detections])
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True,
                 caption=f"{len(detections)} objects — {ms:.0f}ms")


def video_mode():
    uploaded = st.sidebar.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv", "webm"])

    max_frames = st.sidebar.slider("Max frames", 5, 100, 30)
    extract_mode = st.sidebar.radio("Extraction", ["Scene detection", "Fixed 1fps"], horizontal=True)

    if not uploaded:
        st.markdown("### Upload a video to analyze")
        return

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    c1, c2, c3, c4 = st.sidebar.columns(4)
    c1.markdown(stat_box(f"{duration:.0f}s", "Dur"), unsafe_allow_html=True)
    c2.markdown(stat_box(total_frames, "Frm"), unsafe_allow_html=True)
    c3.markdown(stat_box(f"{fps:.0f}", "FPS"), unsafe_allow_html=True)
    c4.markdown(stat_box("--", "Obj"), unsafe_allow_html=True)

    if "results" not in st.session_state or st.sidebar.button("Re-analyze", type="primary"):
        _run_analysis(video_path, extract_mode, max_frames)

    if "results" not in st.session_state:
        if st.sidebar.button("Analyze", type="primary"):
            _run_analysis(video_path, extract_mode, max_frames)
        else:
            try:
                os.unlink(video_path)
            except OSError:
                pass
            return

    results = st.session_state["results"]
    all_objects = st.session_state["all_objects"]
    total_time = st.session_state["total_time"]

    # Sidebar: summary + object list
    st.sidebar.markdown("---")
    c1, c2 = st.sidebar.columns(2)
    c1.markdown(stat_box(len(all_objects), "Objects Found"), unsafe_allow_html=True)
    c2.markdown(stat_box(f"{total_time:.1f}s", "Processed In"), unsafe_allow_html=True)

    st.sidebar.markdown('<div class="section-hdr">Objects Found</div>', unsafe_allow_html=True)
    sorted_obj = sorted(all_objects.items(), key=lambda x: x[1]["count"], reverse=True)
    for label, info in sorted_obj:
        pct = info["count"] / len(results) * 100
        st.sidebar.markdown(
            f'<div class="ocard">'
            f'<div class="oname">{label}</div>'
            f'<div class="ometa">{info["count"]}x | {info["first"]:.1f}s-{info["last"]:.1f}s | peak {info["peak"]:.2f}</div>'
            f'<div class="obar"><div class="obar-fill" style="width:{min(pct,100):.0f}%"></div></div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Object pills
    pills = " ".join(f'<span class="pill">{l} ({info["count"]})</span>' for l, info in sorted_obj)
    st.markdown(pills, unsafe_allow_html=True)

    # Search bar
    all_labels = sorted(all_objects.keys())
    search = st.text_input("Search for an object", placeholder="e.g. car, knife, man...", label_visibility="collapsed")

    search_term = search.strip().lower() if search else ""

    if search_term:
        # Find matching frames
        matching_indices = []
        for i, r in enumerate(results):
            for d in r["detections"]:
                if search_term in d["label"].lower():
                    matching_indices.append(i)
                    break

        if not matching_indices:
            st.warning(f'No frames contain "{search}"')
        else:
            st.markdown(
                f'<div class="section-hdr">'
                f'"{search}" found in {len(matching_indices)} / {len(results)} frames'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Show matching frames in a grid
            n_cols = min(4, len(matching_indices))
            for row_start in range(0, len(matching_indices), n_cols):
                cols = st.columns(n_cols)
                for j in range(n_cols):
                    idx = row_start + j
                    if idx >= len(matching_indices):
                        break
                    fi = matching_indices[idx]
                    r = results[fi]
                    # Highlight only matching detections
                    matched = [d for d in r["detections"] if search_term in d["label"].lower()]
                    highlighted = draw_boxes(r["frame"], matched)

                    with cols[j]:
                        st.image(cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB), use_container_width=True)
                        match_labels = ", ".join(f"{d['label']} ({d['score']:.2f})" for d in matched)
                        st.caption(f"**t={r['timestamp']:.1f}s** — {match_labels}")

            st.markdown("---")

    # Frame browser
    st.markdown('<div class="section-hdr">Frame Browser</div>', unsafe_allow_html=True)
    if results:
        frame_idx = st.slider(
            "Frame", 0, len(results) - 1, 0,
            format="Frame %d",
            label_visibility="collapsed",
        )

        r = results[frame_idx]

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(r["frame"], cv2.COLOR_BGR2RGB),
                     use_container_width=True,
                     caption=f"t={r['timestamp']:.1f}s — Original")
        with col2:
            st.image(cv2.cvtColor(r["annotated"], cv2.COLOR_BGR2RGB),
                     use_container_width=True,
                     caption=f"t={r['timestamp']:.1f}s — {len(r['detections'])} detections")

        # Detection table
        if r["detections"]:
            det_html = ""
            for d in r["detections"]:
                det_html += (
                    f'<div class="det-row">'
                    f'<div><span class="det-label">{d["label"]}</span>'
                    f'<span class="det-src">{d["source"]}</span></div>'
                    f'<span class="det-score">{d["score"]:.3f}</span>'
                    f'</div>'
                )
            st.markdown(f'<div style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:6px;padding:4px 0;">{det_html}</div>', unsafe_allow_html=True)

    # Timeline grid
    st.markdown('<div class="section-hdr">Timeline</div>', unsafe_allow_html=True)
    n_cols = min(6, len(results))
    for row_start in range(0, len(results), n_cols):
        cols = st.columns(n_cols)
        for j in range(n_cols):
            idx = row_start + j
            if idx >= len(results):
                break
            r = results[idx]
            with cols[j]:
                thumb = cv2.resize(r["annotated"], (320, 180))
                labels = ", ".join(d["label"] for d in r["detections"][:3])
                st.image(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.caption(f"**{r['timestamp']:.1f}s** — {labels}")

    try:
        os.unlink(video_path)
    except OSError:
        pass


def _run_analysis(video_path, extract_mode, max_frames):
    from object_detection.video import extract_keyframes
    from object_detection.moe import detect_moe, reset_tracker

    with st.spinner("Extracting keyframes..."):
        if extract_mode == "Scene detection":
            keyframes = extract_keyframes(video_path, mode="scene", max_frames=max_frames)
        else:
            keyframes = extract_keyframes(video_path, mode="fixed", fixed_fps=1.0, max_frames=max_frames)

    reset_tracker()
    progress = st.progress(0, text="Analyzing...")
    results = []
    all_objects = {}
    t_start = time.perf_counter()

    for i, (ts, frame) in enumerate(keyframes):
        detections = detect_moe(frame)

        frame_dets = []
        for d in detections:
            det = {"label": d.label, "score": d.confidence, "bbox": d.bbox, "source": d.source}
            frame_dets.append(det)

            if d.label not in all_objects:
                all_objects[d.label] = {"count": 0, "peak": 0, "first": ts, "last": ts}
            all_objects[d.label]["count"] += 1
            all_objects[d.label]["peak"] = max(all_objects[d.label]["peak"], d.confidence)
            all_objects[d.label]["last"] = ts

        annotated = draw_boxes(frame, frame_dets)
        results.append({"timestamp": ts, "frame": frame, "annotated": annotated, "detections": frame_dets})

        labels = ", ".join(d.label for d in detections[:4])
        progress.progress((i + 1) / len(keyframes), text=f"[{i+1}/{len(keyframes)}] t={ts:.1f}s — {labels}")

    total_time = time.perf_counter() - t_start
    progress.empty()

    st.session_state["results"] = results
    st.session_state["all_objects"] = all_objects
    st.session_state["total_time"] = total_time


if __name__ == "__main__":
    main()

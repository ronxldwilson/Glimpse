# Glimpse

Zero-shot object discovery for images and video. No labels, no training, no cloud APIs.

Glimpse uses a **Mixture of Experts** (MoE) approach — combining YOLO-World, CLIP, scene classification, and temporal tracking — to detect and identify objects in any image or video. It produces structured manifests with bounding boxes, confidence scores, and object timelines.

## What it does

Give Glimpse any video. It tells you what's in it, where, and when.

```
$ analyze-video cooking.mp4 --output report.html

Video: 30.0s, 750 frames @ 25fps
Extracting keyframes (mode=scene)...
  11 keyframes in 150ms
Analyzing keyframes...
  [6/11] t=16.3s: cutting board, counter, man, woman, sink
  [8/11] t=22.7s: man, saucepan, window, pan, wok
  [9/11] t=23.4s: table, lemon, apple, apple, onion

Done: 22 unique objects across 11 frames in 15.7s
```

The HTML report includes an interactive timeline — click any keyframe to see the full-res image with **bounding boxes** drawn around every detected object.

## How it works

```
Image/Video
    |
    v
[ffmpeg scene detection] ── extract keyframes where content changes
    |
    v
[MoE Detection] ── four experts work together per frame:
    |
    |── Expert 1: YOLO-World ──── fast object detection with bboxes (~63ms)
    |── Expert 2: FastSAM+CLIP ── segment → isolate → classify rare objects (~800ms)
    |── Expert 3: Scene context ── "kitchen" boosts cookware scores (+60ms)
    |── Expert 4: Temporal track ─ carry detections across similar frames (+3ms)
    |
    v
[Merge + Filter] ── YOLO names common objects, CLIP catches what YOLO misses,
                     scene context boosts relevant labels, confidence filtering
    |
    v
Structured output: object labels, bounding boxes, confidence, source, timestamps
```

### Mixture of Experts strategy

Each expert has a strength:

| Expert | Speed | Strength | Weakness |
|--------|-------|----------|----------|
| **YOLO-World** | 63ms | Precise bboxes, high confidence on common objects | Limited vocabulary |
| **CLIP taxonomy** | 800ms | 1,060 concepts, catches rare objects (onion, wok) | Lower confidence, some noise |
| **Scene classifier** | 60ms | Contextual reweighting (kitchen → boost cookware) | Indirect |
| **Temporal tracker** | 3ms | Carries detections across similar frames | Only helps video |

When both YOLO and CLIP detect the same region, YOLO's label wins if confident (it's more precise). CLIP's label wins when YOLO misses the object entirely. Scene context adds a +0.05 confidence boost to labels that match the detected scene type.

## Installation

```bash
git clone https://github.com/ronxldwilson/Glimpse.git
cd Glimpse
uv venv && uv pip install -e ".[dev]"
```

Download model weights:
```bash
mkdir -p models
curl -L https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt -o models/FastSAM-s.pt
curl -L https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-worldv2.pt -o models/yolov8s-worldv2.pt
```

Download WordNet data:
```bash
uv run python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

The CLIP ViT-B-16 weights and taxonomy embeddings are downloaded/cached automatically on first run.

### Docker

```bash
docker compose build
```

Analyze a video:
```bash
docker compose run glimpse object_detection.demo_video /data/video.mp4 --output /data/report.html
```

Discover objects in images:
```bash
docker compose run glimpse object_detection.demo_discover /data/image.jpg --output /data/report.html
```

Place your files in the `data/` directory — it's mounted into the container.

## Usage

### Analyze a video (MoE pipeline)

```bash
analyze-video video.mp4 --output report.html
```

Options:
```
--mode scene|fixed    Frame extraction mode (default: scene detection)
--fps 1.0             Frames per second (fixed mode)
--threshold 0.3       Scene change sensitivity
--max-frames 200      Maximum frames to analyze
```

The HTML report includes:
- Sidebar with all detected objects, sorted by frequency
- Interactive timeline of keyframes
- Click any frame to see full-res image with **bounding boxes**
- Per-detection confidence scores and source tags (yolo/clip/both)

### Discover objects in images

```bash
discover image1.jpg image2.jpg --output report.html
```

Uses the CLIP taxonomy for hierarchical discovery — no labels needed.

### Detect specific objects

```bash
detect-demo image.jpg --labels laptop headphones phone desk --output demo.html
```

## Performance

All benchmarks on Apple M-series CPU, no GPU.

| Component | Time |
|-----------|------|
| YOLO-World detection | ~63ms/frame |
| FastSAM segmentation | ~160ms/frame |
| CLIP ViT-B-16 encoding | ~50ms/region |
| Scene classification | ~60ms/frame |
| Temporal tracking | ~3ms/frame |
| **MoE total per frame** | **~800ms** |

| Video | Keyframes | Processing Time |
|-------|-----------|-----------------|
| 30s cooking video | 11 | **16s** |
| 2 min street video | 50 | **90s** |

Taxonomy encoding is a one-time cost (~25s, cached to disk). Model loading is ~6s per session.

## Architecture

```
object_detection/
  moe.py            -- Mixture of Experts: YOLO + CLIP + scene + tracking
  segment.py        -- FastSAM segmentation, region merging
  encoder.py        -- CLIP ViT-B-16 with prompt templates
  taxonomy.py       -- WordNet taxonomy (1,060 nodes, auto-expanded)
  discover.py       -- Hierarchical tree-walk discovery
  video.py          -- Video pipeline (ffmpeg keyframes + MoE per frame)
  detector.py       -- Simple label-based detection
  demo.py           -- Multi-image HTML report
  demo_discover.py  -- Discovery HTML report
  demo_video.py     -- Video HTML report with bbox overlays
  vocab.py          -- Large vocabulary builder (optional)
  cli.py            -- CLI entry points
```

## How the MoE taxonomy works

Glimpse uses a curated taxonomy built from WordNet, organized into 18 visual categories with auto-expanded leaves (~1,060 total concepts):

```
object
├── person (man, woman, child, baby)
├── electronics (laptop, smartphone, headphones, camera, ...)
├── furniture (chair, table, desk, sofa, bed, shelf, ...)
├── food
│   ├── fruit (apple, banana, orange, ...)
│   ├── vegetable (onion, tomato, carrot, ...)
│   └── ...
├── kitchen item
│   ├── cookware (pot, pan, wok, saucepan, ...)
│   ├── utensil (knife, spatula, ladle, ...)
│   └── appliance (oven, microwave, stove, ...)
├── vehicle (car, truck, bus, bicycle, ...)
├── clothing (shirt, jacket, hat, shoes, ...)
└── ... (18 categories total)
```

At runtime, identifying an object is a **hierarchical tree walk**: score against top-level categories, descend into the best match, repeat. This is more accurate than flat vocabulary matching because each branching point only has 5-15 choices.

## License

MIT

# Glimpse

Zero-shot object discovery for images and video. No labels, no training, no cloud APIs.

Glimpse segments an image into objects using [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM), identifies each object by walking a 27,000-concept vocabulary built from [WordNet](https://wordnet.princeton.edu/) using [CLIP](https://github.com/mlfoundations/open_clip) similarity, and produces structured manifests of everything it finds.

## What it does

Give Glimpse any image or video. It tells you what's in it.

```
$ discover photo.jpg
electronics > headphones  [0.281 > 0.331]
electronics > smartphone  [0.254 > 0.303]
accessory > watch > smartwatch  [0.250 > 0.264 > 0.281]
food > vegetable > bell pepper  [0.249 > 0.288 > 0.328]
person > woman  [0.271 > 0.265]
kitchen item > cookware > pot  [0.228 > 0.236 > 0.251]
```

No labels provided. The system discovers objects by navigating a taxonomy tree top-down — broad category first, then narrowing to the specific object. Each step is a CLIP similarity lookup against pre-encoded embeddings.

## How it works

```
Image/Video
    |
    v
[FastSAM] -- YOLO-based segmentation, ~160ms/image
    |
    v
[Isolate regions on white background] -- cleaner CLIP signal
    |
    v
[CLIP encode each region] -- batch encode, ~15ms/region
    |
    v
[Match against 27K WordNet vocabulary] -- single matrix multiply, <1ms
    |
    v
Structured output: object labels, taxonomy paths, bounding boxes, timestamps
```

### Key ideas

- **Segment first, classify second** — FastSAM finds object boundaries without knowing what they are, then CLIP identifies each isolated region. This beats whole-image CLIP which dilutes the embedding when multiple objects are present.

- **White background isolation** — each segment is placed on a white background before CLIP encoding. This aligns with CLIP's training distribution (product photos, stock images) and measurably improves scores.

- **Prompt-engineered CLIP** — instead of encoding bare labels ("laptop"), each concept is encoded as the average of 5 prompt templates ("a photo of a laptop", "a close-up photo of a laptop", etc.) for better score separation.

- **Hierarchical discovery** — instead of flat top-k matching against 27K labels, the system can walk a curated taxonomy tree (electronics > audio > headphones) using beam search at each level.

- **Pre-encoded vocabulary** — the 27K-label vocabulary is encoded once and cached to disk (~55MB). At runtime, matching is just a matrix multiply — near instant.

## Installation

```bash
git clone https://github.com/ronxldwilson/Glimpse.git
cd Glimpse
uv venv && uv pip install -e ".[dev]"
```

Download the FastSAM weights:
```bash
mkdir -p models
curl -L https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt -o models/FastSAM-s.pt
```

Download WordNet data:
```bash
uv run python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

Build the vocabulary (one-time, ~2 min):
```bash
uv run python encode_vocab.py
```

### Docker

```bash
docker compose build
```

Build the vocabulary (one-time):
```bash
docker compose run glimpse object_detection.vocab
```

Discover objects:
```bash
docker compose run glimpse object_detection.demo_discover /data/image.jpg --output /data/report.html
```

Analyze video:
```bash
docker compose run glimpse object_detection.demo_video /data/video.mp4 --output /data/report.html
```

Place your images/videos in the `data/` directory — it's mounted into the container.

## Usage

### Discover objects in images (no labels needed)

```bash
discover image1.jpg image2.jpg --output report.html
```

Generates an HTML report with:
- Segmentation overlay
- Per-region identification with taxonomy paths
- Confidence scores at each level

### Detect specific objects

```bash
detect-demo image.jpg --labels laptop headphones phone desk --output demo.html
```

### Analyze a video

```bash
analyze-video video.mp4 --output report.html
```

Options:
```
--mode scene|fixed    Frame extraction mode (default: scene detection)
--fps 1.0             Frames per second (fixed mode)
--threshold 0.3       Scene change sensitivity
--max-frames 200      Maximum frames to analyze
--score 0.22          Minimum detection score
```

The video pipeline:
1. Extracts keyframes using ffmpeg scene detection (only frames where something changes)
2. Skips visually similar frames automatically
3. Segments and identifies objects in each keyframe
4. Produces a manifest: which objects appear, when, and for how long

## Performance

All benchmarks on Apple M-series CPU, no GPU.

| Step | Time |
|------|------|
| FastSAM segmentation | ~160ms/image |
| CLIP region encoding | ~15ms/region |
| Vocabulary matching (27K labels) | <1ms |
| **Typical image (15 regions)** | **~400ms** |

| Video (1fps sampling) | Time |
|---|---|
| 30s video | ~12s |
| 1 min video | ~24s |
| 5 min video | ~2 min |

Vocabulary encoding is a one-time cost (~2 min). Model loading is a one-time cost per session (~2.5s).

## Architecture

```
object_detection/
  segment.py        -- FastSAM segmentation, region merging
  encoder.py        -- CLIP encoding with prompt templates
  detector.py       -- Label-based detection (user provides labels)
  taxonomy.py       -- WordNet taxonomy builder (335 curated nodes)
  vocab.py          -- 27K vocabulary builder with incremental encoding
  discover.py       -- Hierarchical tree-walk discovery
  video.py          -- Video analysis pipeline (ffmpeg + per-frame analysis)
  demo.py           -- Multi-image HTML report generator
  demo_discover.py  -- Discovery HTML report generator
  demo_video.py     -- Video HTML report generator
  cli.py            -- CLI entry points
```

## How the vocabulary works

On first run, Glimpse builds a vocabulary from WordNet:

1. Traverses WordNet's noun hierarchy under visual anchors (artifact, animal, food, clothing, vehicle, etc.)
2. Filters for visually recognizable concepts (~27K nouns)
3. Encodes each with CLIP using 5 prompt templates and averages the embeddings
4. Saves to disk (`models/vocab_embeddings.npy`, ~55MB)

At runtime, identifying an object is a single matrix multiply: region embedding (512-dim) dot product against all 27K vocabulary embeddings. The top-k results are the identification.

The vocabulary also stores WordNet hypernym paths, so every identification comes with a full taxonomy: `entity > physical entity > object > artifact > instrumentality > device > electronic equipment > headphones`.

## License

MIT

"""Build and cache a large visual vocabulary from WordNet, pre-encoded with CLIP."""

import json
import sys
import time
import numpy as np
from pathlib import Path

from nltk.corpus import wordnet as wn


VISUAL_ANCHORS = [
    "artifact.n.01", "animal.n.01", "plant.n.02", "food.n.01",
    "person.n.01", "body_part.n.01", "geological_formation.n.01",
    "natural_object.n.01", "clothing.n.01", "furniture.n.01",
    "vehicle.n.01", "building.n.01", "container.n.01",
    "device.n.01", "tool.n.01", "fabric.n.01",
    "electronic_equipment.n.01", "musical_instrument.n.01",
    "game_equipment.n.01",
]

CACHE_DIR = Path(__file__).parent.parent / "models"


def _log(msg: str):
    print(msg, flush=True)


def build_visual_vocab() -> list[dict]:
    vocab = {}

    for anchor_id in VISUAL_ANCHORS:
        try:
            anchor = wn.synset(anchor_id)
        except Exception:
            continue

        for desc in anchor.closure(lambda s: s.hyponyms()):
            lemma = desc.lemmas()[0].name().replace("_", " ")
            if len(lemma) < 3 or len(lemma) > 30:
                continue
            if any(c.isupper() for c in lemma[1:]):
                continue
            if lemma in vocab:
                continue

            hypernym_path = []
            for path in desc.hypernym_paths():
                if len(path) > len(hypernym_path):
                    hypernym_path = path
            path_names = [s.lemmas()[0].name().replace("_", " ") for s in hypernym_path]

            vocab[lemma] = {
                "name": lemma,
                "synset": desc.name(),
                "path": path_names,
            }

    return sorted(vocab.values(), key=lambda v: v["name"])


def encode_and_cache(encoder, force: bool = False) -> tuple[list[str], np.ndarray]:
    labels_path = CACHE_DIR / "vocab_labels.json"
    embeds_path = CACHE_DIR / "vocab_embeddings.npy"
    partial_path = CACHE_DIR / "vocab_embeddings_partial.npy"
    progress_path = CACHE_DIR / "vocab_progress.json"

    if not force and labels_path.exists() and embeds_path.exists():
        _log("Loading cached vocabulary...")
        with open(labels_path) as f:
            labels = json.load(f)
        embeddings = np.load(embeds_path)
        _log(f"  {len(labels)} labels loaded from cache")
        return labels, embeddings

    _log("Building visual vocabulary from WordNet...")
    vocab = build_visual_vocab()
    labels = [v["name"] for v in vocab]
    _log(f"  {len(labels)} visual nouns found")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w") as f:
        json.dump(labels, f)

    # Check for partial progress
    start_idx = 0
    all_embeddings = []
    if not force and partial_path.exists() and progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)
        start_idx = progress.get("done", 0)
        if start_idx > 0 and start_idx <= len(labels):
            partial = np.load(partial_path)
            all_embeddings.append(partial)
            _log(f"  Resuming from {start_idx}/{len(labels)} (loaded partial cache)")

    batch_size = 200
    t_start = time.perf_counter()

    for i in range(start_idx, len(labels), batch_size):
        batch = labels[i : i + batch_size]
        embeds = encoder.encode_labels(batch)
        all_embeddings.append(embeds)

        done = min(i + batch_size, len(labels))
        elapsed = time.perf_counter() - t_start
        rate = (done - start_idx) / elapsed if elapsed > 0 else 0
        remaining = (len(labels) - done) / rate if rate > 0 else 0
        _log(f"  {done}/{len(labels)} ({done*100//len(labels)}%) - {rate:.0f} labels/s - ~{remaining:.0f}s remaining")

        # Save partial progress every 1000 labels
        if done % 1000 < batch_size:
            partial_embeds = np.vstack(all_embeddings)
            np.save(partial_path, partial_embeds)
            with open(progress_path, "w") as f:
                json.dump({"done": done, "total": len(labels)}, f)

    embeddings = np.vstack(all_embeddings)

    np.save(embeds_path, embeddings)
    _log(f"  Saved {embeds_path.name} ({embeddings.nbytes / 1e6:.1f}MB)")

    paths_data = {v["name"]: v["path"] for v in vocab}
    with open(CACHE_DIR / "vocab_paths.json", "w") as f:
        json.dump(paths_data, f)

    # Clean up partial files
    partial_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)

    total_time = time.perf_counter() - t_start
    _log(f"  Done in {total_time:.0f}s")

    return labels, embeddings


def load_cached() -> tuple[list[str], np.ndarray, dict] | None:
    labels_path = CACHE_DIR / "vocab_labels.json"
    embeds_path = CACHE_DIR / "vocab_embeddings.npy"
    paths_path = CACHE_DIR / "vocab_paths.json"

    if not labels_path.exists() or not embeds_path.exists():
        return None

    with open(labels_path) as f:
        labels = json.load(f)
    embeddings = np.load(embeds_path)

    paths = {}
    if paths_path.exists():
        with open(paths_path) as f:
            paths = json.load(f)

    return labels, embeddings, paths

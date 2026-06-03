"""Build object-only vocabulary (no abstract concepts, no people traits, no chemicals)."""

import json
import time
import numpy as np
from pathlib import Path
from nltk.corpus import wordnet as wn
from object_detection.encoder import CLIPEncoder

OBJECT_ANCHORS = [
    "artifact.n.01", "food.n.02", "animal.n.01",
    "plant.n.02", "natural_object.n.01",
]
EXCLUDE = {
    "person.n.01", "chemical.n.01", "compound.n.02", "substance.n.01",
    "drug.n.01", "microorganism.n.01", "virus.n.01",
    "body_substance.n.01", "color.n.01", "waste.n.01",
    "mixture.n.01", "alloy.n.01",
}


def is_excluded(s):
    for p in s.hypernym_paths():
        for a in p:
            if a.name() in EXCLUDE:
                return True
    return False


def collect(syn, depth=0, max_depth=3):
    r = []
    lemma = syn.lemmas()[0].name().replace("_", " ")
    if len(lemma) < 3 or len(lemma) > 25:
        return r
    if any(c.isupper() for c in lemma[1:]):
        return r
    if is_excluded(syn):
        return r
    r.append({"name": lemma, "synset": syn.name()})
    if depth < max_depth:
        for h in syn.hyponyms():
            r.extend(collect(h, depth + 1, max_depth))
    return r


print("Building object-only vocabulary...", flush=True)
vocab = {}
for aid in OBJECT_ANCHORS:
    try:
        anchor = wn.synset(aid)
    except Exception:
        continue
    for item in collect(anchor, 0, 3):
        vocab[item["name"]] = item

vocab = sorted(vocab.values(), key=lambda v: v["name"])
labels = [v["name"] for v in vocab]
print(f"  {len(labels)} objects found", flush=True)

print("Loading CLIP...", flush=True)
encoder = CLIPEncoder()

print("Encoding...", flush=True)
batch_size = 200
all_embeds = []
t0 = time.perf_counter()
for i in range(0, len(labels), batch_size):
    batch = labels[i : i + batch_size]
    embeds = encoder.encode_labels(batch)
    all_embeds.append(embeds)
    done = min(i + batch_size, len(labels))
    elapsed = time.perf_counter() - t0
    rate = done / elapsed if elapsed > 0 else 0
    remaining = (len(labels) - done) / rate if rate > 0 else 0
    print(f"  {done}/{len(labels)} ({done*100//len(labels)}%) ~{remaining:.0f}s left", flush=True)

embeddings = np.vstack(all_embeds)

models = Path("models")
models.mkdir(exist_ok=True)
np.save(models / "vocab_embeddings.npy", embeddings)
with open(models / "vocab_labels.json", "w") as f:
    json.dump(labels, f)

paths = {}
for v in vocab:
    s = wn.synset(v["synset"])
    best = max(s.hypernym_paths(), key=len)
    paths[v["name"]] = [x.lemmas()[0].name().replace("_", " ") for x in best]
with open(models / "vocab_paths.json", "w") as f:
    json.dump(paths, f)

print(f"Done: {len(labels)} objects, {embeddings.nbytes/1e6:.1f}MB", flush=True)

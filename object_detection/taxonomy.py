"""Hierarchical object taxonomy built from WordNet, pre-encoded with CLIP."""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from nltk.corpus import wordnet as wn


@dataclass
class TaxNode:
    name: str
    synset: str | None = None
    children: list["TaxNode"] = field(default_factory=list)
    embedding: np.ndarray | None = None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def count(self) -> int:
        return 1 + sum(c.count() for c in self.children)

    def all_names(self) -> list[str]:
        names = [self.name]
        for c in self.children:
            names.extend(c.all_names())
        return names


# Curated top-level with visually meaningful WordNet expansions
TAXONOMY_SPEC = {
    "person": {
        "_synset": "person.n.01",
        "_expand": False,
        "_children": {
            "man": {},
            "woman": {},
            "child": {},
            "baby": {},
        },
    },
    "animal": {
        "_synset": "animal.n.01",
        "_expand": True,
        "_max_children": 12,
        "_only": ["dog", "cat", "bird", "fish", "horse", "cow", "sheep",
                  "elephant", "bear", "deer", "rabbit", "insect"],
    },
    "vehicle": {
        "_synset": "vehicle.n.01",
        "_expand": True,
        "_max_children": 12,
    },
    "electronics": {
        "_synset": "electronic_equipment.n.01",
        "_expand": True,
        "_max_children": 15,
        "_extra": ["laptop", "smartphone", "tablet", "headphones", "smartwatch",
                   "camera", "speaker", "monitor", "keyboard", "mouse",
                   "television", "microphone", "charger", "cable", "remote control"],
    },
    "furniture": {
        "_synset": "furniture.n.01",
        "_expand": True,
        "_max_children": 15,
        "_extra": ["chair", "table", "desk", "sofa", "bed", "shelf",
                   "cabinet", "dresser", "bookshelf", "bench", "stool"],
    },
    "clothing": {
        "_synset": "clothing.n.01",
        "_expand": True,
        "_max_children": 15,
        "_extra": ["shirt", "pants", "dress", "jacket", "coat", "hat",
                   "shoes", "boots", "scarf", "gloves", "tie", "belt"],
    },
    "food": {
        "_synset": "food.n.01",
        "_expand": False,
        "_children": {
            "fruit": {"_extra": ["apple", "banana", "orange", "grape", "strawberry",
                                  "watermelon", "lemon", "pineapple", "mango", "peach"]},
            "vegetable": {"_extra": ["tomato", "carrot", "broccoli", "onion", "potato",
                                      "bell pepper", "cucumber", "lettuce", "corn", "mushroom"]},
            "meat": {"_extra": ["steak", "chicken", "fish fillet", "sausage", "bacon"]},
            "bread": {"_extra": ["loaf", "baguette", "croissant", "muffin", "bagel"]},
            "drink": {"_extra": ["coffee", "tea", "juice", "soda", "water bottle",
                                  "wine", "beer"]},
            "snack": {"_extra": ["chips", "cookie", "candy", "chocolate", "popcorn"]},
        },
    },
    "kitchen item": {
        "_expand": False,
        "_children": {
            "cookware": {"_extra": ["pot", "pan", "frying pan", "wok", "saucepan",
                                     "baking sheet", "casserole dish"]},
            "utensil": {"_extra": ["spatula", "ladle", "whisk", "tongs", "knife",
                                    "spoon", "fork", "cutting board", "rolling pin"]},
            "appliance": {"_extra": ["oven", "microwave", "toaster", "blender",
                                      "coffee maker", "refrigerator", "dishwasher",
                                      "stove", "mixer"]},
            "tableware": {"_extra": ["plate", "bowl", "cup", "mug", "glass",
                                      "wine glass", "pitcher", "napkin"]},
        },
    },
    "tool": {
        "_synset": "tool.n.01",
        "_expand": True,
        "_max_children": 12,
        "_extra": ["hammer", "screwdriver", "wrench", "pliers", "drill",
                   "saw", "tape measure", "level", "paintbrush"],
    },
    "container": {
        "_expand": False,
        "_children": {
            "box": {},
            "bag": {"_extra": ["backpack", "handbag", "tote bag", "suitcase",
                                "shopping bag", "duffel bag"]},
            "bottle": {"_extra": ["water bottle", "wine bottle", "jar", "vase",
                                   "thermos", "flask"]},
            "basket": {},
            "bucket": {},
            "barrel": {},
        },
    },
    "sport equipment": {
        "_expand": False,
        "_children": {
            "ball": {"_extra": ["soccer ball", "basketball", "tennis ball",
                                 "baseball", "football", "volleyball", "golf ball"]},
            "racket": {"_extra": ["tennis racket", "badminton racket"]},
            "bat": {"_extra": ["baseball bat", "cricket bat"]},
            "helmet": {},
            "skateboard": {},
            "surfboard": {},
            "ski": {},
            "dumbbell": {},
            "yoga mat": {},
        },
    },
    "plant": {
        "_expand": False,
        "_children": {
            "tree": {"_extra": ["oak", "pine", "palm tree", "willow", "maple"]},
            "flower": {"_extra": ["rose", "sunflower", "tulip", "daisy", "lily",
                                   "orchid", "lavender"]},
            "bush": {},
            "grass": {},
            "houseplant": {"_extra": ["cactus", "fern", "succulent", "bonsai"]},
        },
    },
    "building": {
        "_synset": "building.n.01",
        "_expand": True,
        "_max_children": 10,
        "_extra": ["house", "skyscraper", "church", "bridge", "tower", "barn",
                   "warehouse", "garage", "shed"],
    },
    "accessory": {
        "_expand": False,
        "_children": {
            "watch": {"_extra": ["wristwatch", "smartwatch", "pocket watch"]},
            "glasses": {"_extra": ["sunglasses", "eyeglasses", "reading glasses"]},
            "jewelry": {"_extra": ["ring", "necklace", "bracelet", "earring"]},
            "wallet": {},
            "umbrella": {},
            "key": {},
        },
    },
    "writing implement": {
        "_expand": False,
        "_children": {
            "pen": {},
            "pencil": {},
            "marker": {},
            "crayon": {},
        },
    },
    "fixture": {
        "_expand": False,
        "_children": {
            "window": {},
            "door": {},
            "wall": {},
            "floor": {},
            "ceiling": {},
            "stairs": {},
            "pillar": {},
        },
    },
    "decoration": {
        "_expand": False,
        "_children": {
            "painting": {},
            "poster": {},
            "photograph": {},
            "sculpture": {},
            "candle": {},
            "clock": {},
            "mirror": {},
            "rug": {},
            "curtain": {},
        },
    },
    "document": {
        "_expand": False,
        "_children": {
            "book": {},
            "newspaper": {},
            "magazine": {},
            "letter": {},
            "notebook": {},
            "card": {},
        },
    },
}


def build_taxonomy(expand: bool = True) -> TaxNode:
    root = TaxNode(name="object")

    for name, spec in TAXONOMY_SPEC.items():
        child = _build_from_spec(name, spec)
        root.children.append(child)

    if expand:
        auto_expand_leaves(root, max_depth=1, max_children=5)

    return root


def _build_from_spec(name: str, spec: dict) -> TaxNode:
    node = TaxNode(name=name, synset=spec.get("_synset"))

    if spec.get("_expand") and node.synset:
        try:
            syn = wn.synset(node.synset)
            _expand_from_wordnet(node, syn, spec)
        except Exception:
            pass

    if "_children" in spec:
        for child_name, child_spec in spec["_children"].items():
            child = _build_from_spec(child_name, child_spec)
            node.children.append(child)

    if "_extra" in spec:
        existing = {c.name for c in node.children}
        for extra_name in spec["_extra"]:
            if extra_name not in existing:
                node.children.append(TaxNode(name=extra_name))

    return node


def _expand_from_wordnet(node: TaxNode, synset, spec: dict):
    max_children = spec.get("_max_children", 10)
    only_set = set(spec.get("_only", []))

    hyps = synset.hyponyms()
    candidates = []
    for h in hyps:
        name = h.lemmas()[0].name().replace("_", " ")
        if only_set and name not in only_set:
            continue
        n_desc = len(list(h.closure(lambda s: s.hyponyms())))
        candidates.append((name, h.name(), n_desc))

    candidates.sort(key=lambda x: x[2], reverse=True)
    existing = {c.name for c in node.children}

    for name, syn_id, _ in candidates[:max_children]:
        if name not in existing:
            node.children.append(TaxNode(name=name, synset=syn_id))


SKIP_EXPANSIONS = {
    "person", "man", "woman", "child", "baby",
    "wall", "floor", "ceiling", "door", "window", "stairs", "pillar",
}


def auto_expand_leaves(root: TaxNode, max_depth: int = 2, max_children: int = 10):
    _expand_node(root, 0, max_depth, max_children)
    return root


def _expand_node(node: TaxNode, depth: int, max_depth: int, max_children: int):
    for child in node.children:
        _expand_node(child, depth, max_depth, max_children)

    if not node.is_leaf():
        return
    if node.name in SKIP_EXPANSIONS:
        return
    if not node.synset:
        synsets = wn.synsets(node.name.replace(" ", "_"), pos=wn.NOUN)
        if not synsets:
            return
        node.synset = synsets[0].name()

    try:
        syn = wn.synset(node.synset)
    except Exception:
        return

    _add_hyponyms(node, syn, 0, max_depth, max_children)


def _add_hyponyms(node: TaxNode, synset, depth: int, max_depth: int, max_children: int):
    if depth >= max_depth:
        return

    hyps = synset.hyponyms()
    if not hyps:
        return

    candidates = []
    for h in hyps:
        name = h.lemmas()[0].name().replace("_", " ")
        if len(name) < 3 or len(name) > 25:
            continue
        if any(c.isupper() for c in name[1:]):
            continue
        n_desc = len(h.hyponyms())
        candidates.append((name, h, n_desc))

    candidates.sort(key=lambda x: x[2], reverse=True)
    existing = {c.name for c in node.children}

    added = 0
    for name, h_syn, _ in candidates:
        if added >= max_children:
            break
        if name in existing or name == node.name:
            continue

        child = TaxNode(name=name, synset=h_syn.name())
        _add_hyponyms(child, h_syn, depth + 1, max_depth, max_children)
        node.children.append(child)
        added += 1


CACHE_DIR = Path(__file__).parent.parent / "models"


def encode_taxonomy(root: TaxNode, encoder) -> None:
    cache_path = CACHE_DIR / "taxonomy_embeddings.npy"
    all_names = root.all_names()

    if cache_path.exists():
        all_embeddings = np.load(cache_path)
        if len(all_embeddings) == len(all_names):
            _assign_embeddings(root, all_embeddings)
            return

    batch_size = 200
    chunks = []
    for i in range(0, len(all_names), batch_size):
        chunks.append(encoder.encode_labels(all_names[i:i + batch_size]))
    all_embeddings = np.vstack(chunks)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, all_embeddings)

    _assign_embeddings(root, all_embeddings)


def _assign_embeddings(root: TaxNode, all_embeddings: np.ndarray) -> None:
    idx = [0]

    def _assign(node):
        node.embedding = all_embeddings[idx[0]]
        idx[0] += 1
        for c in node.children:
            _assign(c)

    _assign(root)


def print_taxonomy(node: TaxNode, indent: int = 0):
    prefix = "  " * indent
    n_children = len(node.children)
    suffix = f" ({n_children})" if n_children > 0 else ""
    print(f"{prefix}{node.name}{suffix}")
    for c in node.children:
        print_taxonomy(c, indent + 1)

"""Mixture of Experts — YOLO-World + CLIP + scene context + temporal tracking.

Expert 1 (YOLO-World): fast, precise boxes, strong on common objects
Expert 2 (CLIP taxonomy): broader vocabulary, hierarchical, catches rare objects
Expert 3 (Scene classifier): whole-frame context, reweights detections
Expert 4 (Temporal tracker): carries detections across similar frames
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from pathlib import Path

from ultralytics import YOLOWorld

from .segment import segment, merge_nearby_regions, extract_region_image
from .encoder import CLIPEncoder
from .taxonomy import build_taxonomy, encode_taxonomy, TaxNode
from .discover import discover_objects, Discovery


MOE_CLASSES = [
    "person", "man", "woman", "child",
    "knife", "cutting board", "spoon", "fork", "spatula", "ladle", "whisk",
    "pan", "pot", "frying pan", "wok", "saucepan",
    "plate", "bowl", "cup", "mug", "glass", "bottle", "jar",
    "stove", "oven", "microwave", "toaster", "refrigerator", "sink",
    "table", "counter", "chair", "stool", "shelf", "cabinet",
    "onion", "tomato", "carrot", "apple", "banana", "orange", "lemon",
    "pepper", "mushroom", "potato", "garlic", "lettuce", "broccoli",
    "meat", "chicken", "fish", "steak", "sausage", "egg",
    "bread", "cheese", "butter", "rice", "pasta",
    "salt shaker", "pepper grinder", "napkin", "towel",
    "bag", "box", "basket", "bucket",
    "laptop", "phone", "keyboard", "mouse", "monitor", "headphones",
    "book", "pen", "pencil", "notebook",
    "clock", "lamp", "candle", "mirror", "painting", "photograph",
    "car", "bicycle", "motorcycle", "bus", "truck",
    "dog", "cat", "bird", "horse",
    "tree", "flower", "plant",
    "door", "window", "wall", "floor", "stairs",
    "hat", "shirt", "jacket", "shoes", "glasses", "watch", "ring",
    "umbrella", "backpack", "handbag", "suitcase",
    "ball", "racket", "skateboard", "surfboard",
    "scissors", "hammer", "screwdriver",
]

SCENES = [
    "kitchen", "office", "living room", "bedroom", "bathroom",
    "restaurant", "street", "park", "gym", "store",
    "garage", "garden", "classroom", "workshop", "studio",
    "warehouse", "factory", "hospital", "library", "bar",
]

# Scene -> labels that get a confidence boost in that scene
SCENE_BOOSTS = {
    "kitchen": {"knife", "cutting board", "spoon", "fork", "spatula", "ladle",
                "whisk", "pan", "pot", "frying pan", "wok", "saucepan",
                "plate", "bowl", "cup", "mug", "stove", "oven", "microwave",
                "toaster", "refrigerator", "sink", "onion", "tomato", "carrot",
                "apple", "garlic", "pepper", "mushroom", "potato", "lettuce",
                "broccoli", "meat", "chicken", "egg", "bread", "cheese",
                "counter", "salt shaker", "pepper grinder", "napkin", "towel"},
    "restaurant": {"plate", "bowl", "cup", "mug", "glass", "bottle", "fork",
                   "spoon", "knife", "napkin", "table", "chair", "stool", "menu"},
    "office": {"laptop", "phone", "keyboard", "mouse", "monitor", "headphones",
               "book", "pen", "pencil", "notebook", "chair", "desk", "lamp"},
    "living room": {"chair", "stool", "lamp", "clock", "mirror", "painting",
                    "photograph", "candle", "shelf", "cabinet", "book"},
    "street": {"car", "bicycle", "motorcycle", "bus", "truck", "person",
               "man", "woman", "dog", "tree"},
    "gym": {"ball", "racket", "skateboard", "surfboard", "shoes", "towel"},
    "garage": {"car", "motorcycle", "hammer", "screwdriver", "scissors", "box"},
    "workshop": {"hammer", "screwdriver", "scissors", "box"},
    "garden": {"tree", "flower", "plant", "dog", "cat", "bird"},
}


@dataclass
class MoeDetection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    source: str
    yolo_conf: float | None = None
    clip_conf: float | None = None
    clip_path: list[str] | None = None
    scene_boosted: bool = False


_yolo_model: YOLOWorld | None = None
_clip_encoder: CLIPEncoder | None = None
_taxonomy: TaxNode | None = None
_scene_embeds: np.ndarray | None = None


def _get_yolo() -> YOLOWorld:
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    weights = Path(__file__).parent.parent / "models" / "yolov8s-worldv2.pt"
    _yolo_model = YOLOWorld(str(weights))
    _yolo_model.set_classes(MOE_CLASSES)
    return _yolo_model


def _get_clip_and_taxonomy() -> tuple[CLIPEncoder, TaxNode]:
    global _clip_encoder, _taxonomy, _scene_embeds
    if _clip_encoder is not None and _taxonomy is not None:
        return _clip_encoder, _taxonomy
    _clip_encoder = CLIPEncoder()
    _taxonomy = build_taxonomy()
    encode_taxonomy(_taxonomy, _clip_encoder)
    _scene_embeds = _clip_encoder.encode_labels(SCENES)
    return _clip_encoder, _taxonomy


def classify_scene(encoder: CLIPEncoder, image: np.ndarray) -> tuple[str, float]:
    global _scene_embeds
    if _scene_embeds is None:
        _scene_embeds = encoder.encode_labels(SCENES)
    frame_embed = encoder.encode_images([image])[0]
    scores = frame_embed @ _scene_embeds.T
    best = int(np.argmax(scores))
    return SCENES[best], float(scores[best])


class TemporalTracker:
    def __init__(self):
        self.prev_dets: list[MoeDetection] = []
        self.prev_frame: np.ndarray | None = None

    def track(self, frame: np.ndarray, current_dets: list[MoeDetection]) -> list[MoeDetection]:
        if self.prev_frame is None:
            self.prev_frame = frame
            self.prev_dets = current_dets
            return current_dets

        sim = _frame_similarity(frame, self.prev_frame)

        if sim > 0.85:
            carried = self._carry_forward(current_dets)
            current_dets = current_dets + carried

        self.prev_frame = frame
        self.prev_dets = current_dets
        return current_dets

    def _carry_forward(self, current_dets: list[MoeDetection]) -> list[MoeDetection]:
        current_labels = {d.label for d in current_dets}
        carried = []
        for prev in self.prev_dets:
            if prev.label in current_labels:
                continue
            if prev.confidence < 0.30:
                continue
            for curr in current_dets:
                if _bbox_iou(prev.bbox, curr.bbox) > 0.3:
                    break
            else:
                carried.append(MoeDetection(
                    label=prev.label,
                    confidence=prev.confidence * 0.9,
                    bbox=prev.bbox,
                    source=prev.source + "+track",
                    yolo_conf=prev.yolo_conf,
                    clip_conf=prev.clip_conf,
                    clip_path=prev.clip_path,
                ))
        return carried


_tracker = TemporalTracker()


def detect_moe(
    image: np.ndarray,
    yolo_conf: float = 0.20,
    clip_min_score: float = 0.22,
    use_scene: bool = True,
    use_tracking: bool = True,
) -> list[MoeDetection]:
    # Expert 1: YOLO-World
    yolo = _get_yolo()
    yolo_results = yolo(image, verbose=False, conf=yolo_conf)
    boxes = yolo_results[0].boxes

    yolo_dets = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].int().tolist()
        yolo_dets.append({
            "label": MOE_CLASSES[int(boxes.cls[i])],
            "conf": float(boxes.conf[i]),
            "bbox": (x1, y1, x2 - x1, y2 - y1),
        })

    # Expert 2: FastSAM + CLIP taxonomy
    encoder, taxonomy = _get_clip_and_taxonomy()

    regions = segment(image)
    regions = merge_nearby_regions(image, regions)

    clip_discoveries = []
    if regions:
        region_images = [extract_region_image(image, r) for r in regions]
        image_embeds = encoder.encode_images(region_images)
        clip_discoveries = discover_objects(
            image_embeds, regions, taxonomy,
            top_per_region=1, min_score=clip_min_score,
        )

    # Expert 3: Scene classification
    scene = None
    scene_boost_set = set()
    if use_scene:
        scene, scene_conf = classify_scene(encoder, image)
        scene_boost_set = SCENE_BOOSTS.get(scene, set())

    # Merge YOLO + CLIP
    merged = _merge_experts(yolo_dets, clip_discoveries, scene_boost_set)

    # Filter
    merged = _filter_confident(merged)

    # Expert 4: Temporal tracking
    if use_tracking:
        merged = _tracker.track(image, merged)

    merged.sort(key=lambda d: d.confidence, reverse=True)
    return merged


def _merge_experts(
    yolo_dets: list[dict],
    clip_discoveries: list[Discovery],
    scene_boost_set: set[str],
) -> list[MoeDetection]:
    merged: list[MoeDetection] = []
    used_yolo = set()

    boost = 0.05

    for ci, disc in enumerate(clip_discoveries):
        best_yolo_idx = None
        best_iou = 0.0
        for yi, yd in enumerate(yolo_dets):
            if yi in used_yolo:
                continue
            iou = _bbox_iou(disc.bbox, yd["bbox"])
            if iou > best_iou and iou > 0.2:
                best_iou = iou
                best_yolo_idx = yi

        if best_yolo_idx is not None:
            yd = yolo_dets[best_yolo_idx]
            used_yolo.add(best_yolo_idx)

            if yd["conf"] >= 0.25:
                label = yd["label"]
                conf = yd["conf"]
            else:
                label = disc.label
                conf = disc.confidence

            boosted = label in scene_boost_set
            if boosted:
                conf += boost

            merged.append(MoeDetection(
                label=label, confidence=conf, bbox=yd["bbox"],
                source="both", yolo_conf=yd["conf"], clip_conf=disc.confidence,
                clip_path=disc.path, scene_boosted=boosted,
            ))
        else:
            label = disc.label
            conf = disc.confidence
            boosted = label in scene_boost_set
            if boosted:
                conf += boost

            merged.append(MoeDetection(
                label=label, confidence=conf, bbox=disc.bbox,
                source="clip", clip_conf=disc.confidence,
                clip_path=disc.path, scene_boosted=boosted,
            ))

    for yi, yd in enumerate(yolo_dets):
        if yi not in used_yolo:
            label = yd["label"]
            conf = yd["conf"]
            boosted = label in scene_boost_set
            if boosted:
                conf += boost

            merged.append(MoeDetection(
                label=label, confidence=conf, bbox=yd["bbox"],
                source="yolo", yolo_conf=yd["conf"], scene_boosted=boosted,
            ))

    return merged


def _filter_confident(
    detections: list[MoeDetection],
    both_min: float = 0.25,
    yolo_only_min: float = 0.25,
    clip_only_min: float = 0.30,
) -> list[MoeDetection]:
    moe_set = set(MOE_CLASSES)
    filtered = []
    for d in detections:
        if d.source == "both":
            if d.yolo_conf and d.yolo_conf >= 0.25:
                filtered.append(d)
            elif d.confidence >= both_min:
                filtered.append(d)
        elif d.source == "yolo":
            if d.yolo_conf >= yolo_only_min:
                filtered.append(d)
        elif d.source == "clip":
            if d.clip_conf >= clip_only_min and d.label in moe_set:
                filtered.append(d)
    return filtered


def reset_tracker():
    global _tracker
    _tracker = TemporalTracker()


def _frame_similarity(a: np.ndarray, b: np.ndarray) -> float:
    ha = cv2.calcHist([a], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hb = cv2.calcHist([b], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    return float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))


def _bbox_iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

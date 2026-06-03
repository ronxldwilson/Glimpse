import numpy as np
from dataclasses import dataclass

from .segment import slic_segment, extract_region_image, Region
from .encoder import CLIPEncoder


@dataclass
class Detection:
    label: str
    score: float
    bbox: tuple[int, int, int, int]
    region_index: int


class ObjectDetector:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        num_superpixels: int = 200,
        compactness: float = 10.0,
        min_region_ratio: float = 0.01,
        merge_threshold: float = 20.0,
    ):
        self.encoder = CLIPEncoder(model_name, pretrained)
        self.num_superpixels = num_superpixels
        self.compactness = compactness
        self.min_region_ratio = min_region_ratio
        self.merge_threshold = merge_threshold

        self._vocab_embeddings: np.ndarray | None = None
        self._vocab_labels: list[str] = []

    def set_vocabulary(self, labels: list[str]):
        self._vocab_labels = labels
        self._vocab_embeddings = self.encoder.encode_texts(labels)

    def detect(
        self,
        image: np.ndarray,
        labels: list[str] | None = None,
        top_k: int = 5,
        threshold: float = 0.15,
    ) -> list[Detection]:
        if labels is not None:
            text_embeds = self.encoder.encode_texts(labels)
            vocab = labels
        elif self._vocab_embeddings is not None:
            text_embeds = self._vocab_embeddings
            vocab = self._vocab_labels
        else:
            raise ValueError("No vocabulary set. Call set_vocabulary() or pass labels.")

        regions = slic_segment(
            image,
            self.num_superpixels,
            self.compactness,
            self.min_region_ratio,
            self.merge_threshold,
        )

        if not regions:
            return []

        region_images = [extract_region_image(image, r) for r in regions]
        image_embeds = self.encoder.encode_images(region_images)

        similarity = image_embeds @ text_embeds.T

        detections = []
        for i, region in enumerate(regions):
            scores = similarity[i]
            top_indices = np.argsort(scores)[::-1][:top_k]

            for idx in top_indices:
                score = float(scores[idx])
                if score >= threshold:
                    detections.append(Detection(
                        label=vocab[idx],
                        score=score,
                        bbox=region.bbox,
                        region_index=i,
                    ))

        detections.sort(key=lambda d: d.score, reverse=True)

        return _nms_by_label(detections, iou_threshold=0.5)


def _nms_by_label(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    by_label: dict[str, list[Detection]] = {}
    for d in detections:
        by_label.setdefault(d.label, []).append(d)

    result = []
    for label_dets in by_label.values():
        label_dets.sort(key=lambda d: d.score, reverse=True)
        kept = []
        for d in label_dets:
            if not any(_iou(d.bbox, k.bbox) > iou_threshold for k in kept):
                kept.append(d)
        result.extend(kept)

    result.sort(key=lambda d: d.score, reverse=True)
    return result


def _iou(a: tuple, b: tuple) -> float:
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

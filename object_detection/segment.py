import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from ultralytics import FastSAM


@dataclass
class Region:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]  # x, y, w, h
    area: int
    crop: np.ndarray


_model: FastSAM | None = None


def _get_model() -> FastSAM:
    global _model
    if _model is not None:
        return _model

    weights = Path(__file__).parent.parent / "models" / "FastSAM-s.pt"
    _model = FastSAM(str(weights))
    return _model


def segment(
    image: np.ndarray,
    min_region_ratio: float = 0.005,
    conf: float = 0.4,
    iou: float = 0.9,
) -> list[Region]:
    h, w = image.shape[:2]
    min_area = int(h * w * min_region_ratio)

    model = _get_model()
    results = model(image, conf=conf, iou=iou, imgsz=640, verbose=False)

    masks_data = results[0].masks
    if masks_data is None:
        return []

    raw_masks = masks_data.data.cpu().numpy()
    boxes = results[0].boxes.xyxy.cpu().numpy()

    regions = []
    for i in range(len(raw_masks)):
        mask_small = raw_masks[i]
        mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.uint8)

        area = int(mask.sum())
        if area < min_area:
            continue

        x1, y1, x2, y2 = boxes[i].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue

        crop = image[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2]
        crop[crop_mask == 0] = 255

        regions.append(Region(mask=mask, bbox=(x1, y1, bw, bh), area=area, crop=crop))

    regions.sort(key=lambda r: r.area, reverse=True)
    return regions


def merge_nearby_regions(
    image: np.ndarray,
    regions: list[Region],
    iou_threshold: float = 0.05,
    distance_ratio: float = 0.3,
) -> list[Region]:
    if len(regions) <= 1:
        return regions

    merged_extra = []
    used = set()

    for i in range(len(regions)):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, len(regions)):
            if j in used:
                continue
            if _masks_nearby(regions[i], regions[j], iou_threshold, distance_ratio):
                group.append(j)
                used.add(j)

        if len(group) > 1:
            combined_mask = np.zeros_like(regions[group[0]].mask)
            for idx in group:
                combined_mask = np.bitwise_or(combined_mask, regions[idx].mask)

            coords = cv2.findNonZero(combined_mask)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                area = int(combined_mask.sum())
                crop = image[y : y + h, x : x + w].copy()
                crop_mask = combined_mask[y : y + h, x : x + w]
                crop[crop_mask == 0] = 255
                merged_extra.append(Region(
                    mask=combined_mask, bbox=(x, y, w, h), area=area, crop=crop,
                ))

    all_regions = regions + merged_extra
    all_regions.sort(key=lambda r: r.area, reverse=True)
    return all_regions


def _masks_nearby(a: Region, b: Region, iou_thresh: float, dist_ratio: float) -> bool:
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    if x2 > x1 and y2 > y1:
        inter = (x2 - x1) * (y2 - y1)
        union = aw * ah + bw * bh - inter
        if union > 0 and inter / union >= iou_thresh:
            return True

    acx, acy = ax + aw / 2, ay + ah / 2
    bcx, bcy = bx + bw / 2, by + bh / 2
    dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
    max_dim = max(aw, ah, bw, bh)
    return dist < max_dim * dist_ratio


def extract_region_image(image: np.ndarray, region: Region) -> np.ndarray:
    x, y, w, h = region.bbox
    crop = image[y : y + h, x : x + w].copy()
    mask_crop = region.mask[y : y + h, x : x + w]
    crop[mask_crop == 0] = 255
    return crop

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class Region:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]  # x, y, w, h
    area: int
    crop: np.ndarray


def slic_segment(
    image: np.ndarray,
    num_superpixels: int = 200,
    compactness: float = 10.0,
    min_region_ratio: float = 0.01,
    merge_threshold: float = 20.0,
) -> list[Region]:
    h, w = image.shape[:2]
    min_area = int(h * w * min_region_ratio)

    slic = cv2.ximgproc.createSuperpixelSLIC(
        image, cv2.ximgproc.SLIC, num_superpixels, compactness
    )
    slic.iterate(10)
    slic.enforceLabelConnectivity(25)
    labels = slic.getLabels()
    n_labels = slic.getNumberOfSuperpixels()

    merged_labels = _merge_similar(image, labels, n_labels, merge_threshold)

    regions = []
    for label_id in np.unique(merged_labels):
        mask = (merged_labels == label_id).astype(np.uint8)
        area = int(mask.sum())
        if area < min_area:
            continue

        coords = cv2.findNonZero(mask)
        if coords is None:
            continue
        x, y, rw, rh = cv2.boundingRect(coords)

        crop = image[y : y + rh, x : x + rw].copy()
        crop_mask = mask[y : y + rh, x : x + rw]
        crop[crop_mask == 0] = 0

        regions.append(Region(mask=mask, bbox=(x, y, rw, rh), area=area, crop=crop))

    regions.sort(key=lambda r: r.area, reverse=True)
    return regions


def _merge_similar(
    image: np.ndarray,
    labels: np.ndarray,
    n_labels: int,
    threshold: float,
) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    means = np.zeros((n_labels, 3), dtype=np.float32)
    for i in range(n_labels):
        mask = labels == i
        if mask.any():
            means[i] = lab[mask].mean(axis=0)

    parent = list(range(n_labels))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    adjacency = set()
    for y in range(labels.shape[0] - 1):
        for x in range(labels.shape[1] - 1):
            c = labels[y, x]
            r = labels[y, x + 1]
            d = labels[y + 1, x]
            if c != r:
                adjacency.add((min(c, r), max(c, r)))
            if c != d:
                adjacency.add((min(c, d), max(c, d)))

    for a, b in adjacency:
        dist = np.linalg.norm(means[a] - means[b])
        if dist < threshold:
            union(a, b)

    merged = labels.copy()
    for i in range(n_labels):
        merged[labels == i] = find(i)

    return merged


def extract_region_image(image: np.ndarray, region: Region) -> np.ndarray:
    x, y, w, h = region.bbox
    crop = image[y : y + h, x : x + w].copy()

    mask_crop = region.mask[y : y + h, x : x + w]
    bg = np.median(crop[mask_crop == 1], axis=0).astype(np.uint8) if mask_crop.any() else np.array([128, 128, 128], dtype=np.uint8)
    crop[mask_crop == 0] = bg

    return crop

"""Hierarchical object discovery — walks the taxonomy tree per segment,
using whole-image context to improve accuracy."""

import numpy as np
from dataclasses import dataclass

from .taxonomy import TaxNode


@dataclass
class Discovery:
    path: list[str]
    scores: list[float]
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    region_index: int


def discover_objects(
    image_embeddings: np.ndarray,
    regions: list,
    taxonomy: TaxNode,
    whole_image_embedding: np.ndarray | None = None,
    context_weight: float = 0.3,
    beam_width: int = 2,
    min_score: float = 0.20,
    top_per_region: int = 1,
) -> list[Discovery]:
    all_discoveries = []

    for i, region in enumerate(regions):
        embed = image_embeddings[i]

        if whole_image_embedding is not None:
            embed = _blend(embed, whole_image_embedding, context_weight)

        paths = _walk_tree(embed, taxonomy, beam_width, min_score)

        region_dets = []
        for path, scores in paths:
            region_dets.append(Discovery(
                path=path,
                scores=scores,
                label=path[-1],
                confidence=scores[-1],
                bbox=region.bbox,
                region_index=i,
            ))

        region_dets.sort(key=lambda d: d.confidence, reverse=True)
        seen_labels = set()
        for d in region_dets:
            if d.label not in seen_labels and len(seen_labels) < top_per_region:
                all_discoveries.append(d)
                seen_labels.add(d.label)

    all_discoveries.sort(key=lambda d: d.confidence, reverse=True)
    return all_discoveries


def _blend(segment_embed: np.ndarray, context_embed: np.ndarray, weight: float) -> np.ndarray:
    blended = (1 - weight) * segment_embed + weight * context_embed
    blended = blended / np.linalg.norm(blended)
    return blended


def _walk_tree(
    embedding: np.ndarray,
    node: TaxNode,
    beam_width: int,
    min_score: float,
) -> list[tuple[list[str], list[float]]]:
    if node.is_leaf() or not node.children:
        return []

    child_embeddings = np.array([c.embedding for c in node.children])
    scores = embedding @ child_embeddings.T

    top_indices = np.argsort(scores)[::-1][:beam_width]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score < min_score:
            continue

        child = node.children[idx]
        path = [child.name]
        score_path = [score]

        if not child.is_leaf() and child.children:
            sub_results = _walk_tree(embedding, child, beam_width, min_score)
            if sub_results:
                best_sub = max(sub_results, key=lambda r: r[1][-1])
                results.append(
                    (path + best_sub[0], score_path + best_sub[1])
                )
            else:
                results.append((path, score_path))
        else:
            results.append((path, score_path))

    return results

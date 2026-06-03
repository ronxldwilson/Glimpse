import argparse
import sys
import time

import cv2
import numpy as np

from .detector import ObjectDetector
from .segment import slic_segment


def main():
    parser = argparse.ArgumentParser(description="SLIC + CLIP object detection")
    sub = parser.add_subparsers(dest="command")

    detect_p = sub.add_parser("detect", help="Detect objects in an image")
    detect_p.add_argument("image", help="Path to image file")
    detect_p.add_argument("--labels", nargs="+", required=True, help="Labels to search for")
    detect_p.add_argument("--top-k", type=int, default=3)
    detect_p.add_argument("--threshold", type=float, default=0.15)
    detect_p.add_argument("--superpixels", type=int, default=200)
    detect_p.add_argument("--output", help="Save annotated image to this path")

    seg_p = sub.add_parser("segment", help="Visualize segmentation only")
    seg_p.add_argument("image", help="Path to image file")
    seg_p.add_argument("--superpixels", type=int, default=200)
    seg_p.add_argument("--output", required=True, help="Save segmentation visualization")

    args = parser.parse_args()

    if args.command == "segment":
        _run_segment(args)
    elif args.command == "detect":
        _run_detect(args)
    else:
        parser.print_help()
        sys.exit(1)


def _run_segment(args):
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: cannot read image {args.image}")
        sys.exit(1)

    t0 = time.perf_counter()
    regions = slic_segment(image, num_superpixels=args.superpixels)
    elapsed = time.perf_counter() - t0

    print(f"Segmented into {len(regions)} regions in {elapsed*1000:.1f}ms")

    vis = image.copy()
    colors = [
        tuple(int(c) for c in np.random.RandomState(i).randint(50, 255, 3))
        for i in range(len(regions))
    ]
    for i, region in enumerate(regions):
        contours, _ = cv2.findContours(region.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, colors[i], 2)
        x, y, w, h = region.bbox
        cv2.putText(vis, f"R{i}", (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)

    cv2.imwrite(args.output, vis)
    print(f"Saved segmentation to {args.output}")

    for i, r in enumerate(regions):
        print(f"  R{i}: bbox=({r.bbox[0]},{r.bbox[1]},{r.bbox[2]},{r.bbox[3]}) area={r.area}px")


def _run_detect(args):
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: cannot read image {args.image}")
        sys.exit(1)

    print("Loading CLIP model...")
    t0 = time.perf_counter()
    detector = ObjectDetector(num_superpixels=args.superpixels)
    model_time = time.perf_counter() - t0
    print(f"Model loaded in {model_time:.1f}s")

    print(f"Detecting: {', '.join(args.labels)}")
    t0 = time.perf_counter()
    detections = detector.detect(
        image,
        labels=args.labels,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    detect_time = time.perf_counter() - t0

    print(f"\nFound {len(detections)} detections in {detect_time*1000:.1f}ms:")
    for d in detections:
        print(f"  {d.label}: {d.score:.3f} at bbox=({d.bbox[0]},{d.bbox[1]},{d.bbox[2]},{d.bbox[3]})")

    if args.output:
        vis = image.copy()
        for d in detections:
            x, y, w, h = d.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{d.label} {d.score:.2f}"
            cv2.putText(vis, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(args.output, vis)
        print(f"Saved annotated image to {args.output}")


if __name__ == "__main__":
    main()

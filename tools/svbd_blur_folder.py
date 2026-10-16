#!/usr/bin/env python3
"""Run the Spatially-Varying Blur Detector (Golestaneh & Karam, CVPR 2017) on a folder of images.

Package: `pip install blur_detector`  (module `blur_detector`, class `BlurDetector`)

What the detector returns
    A blur MAP the size of the input image, values in [0, 1], normalised by the map's OWN maximum
    (so max == 1 for every image). Higher = more high-frequency content (sharper), lower = blurrier.
    It separates sharp and blurry REGIONS inside one image; the package gives no whole-image blur
    score and no threshold. The statistics reported here (mean / median / percentiles / fraction of
    pixels below a level) are summaries of that map and are NOT calibrated absolute blur scores.
    Always inspect the saved heat-maps (--save-maps) and calibrate on labelled images before trusting
    any threshold.

Notes
    * The package is pure Python/NumPy loops and slow on big images -> images are resized so the
      longest side is --max-side (default 640). Resizing changes the apparent blur scale.
    * show_progress is forced off (the package otherwise clears the terminal on every image).
    * A fresh BlurDetector is built per image (the class cannot be reused: it refuses to rebuild its DCT
      matrices).
    * Spurious "divide by zero / overflow in matmul" RuntimeWarnings can appear with NumPy on macOS
      (Accelerate BLAS); they are silenced here.

Usage:
    python svbd_blur_folder.py /path/to/folder --csv out.csv
    python svbd_blur_folder.py /path/to/folder --recursive --save-maps maps/ --workers 4
"""

import argparse
import csv
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

import blur_detector

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LEVELS = (0.1, 0.2, 0.3)  # fractions of map pixels below these map values are reported


class _Detector(blur_detector.BlurDetector):
    """BlurDetector that also keeps the paper's median-entropy score computed internally."""

    def computeScore(self, weighted_local_entropy, T_max):
        self.paper_score = super().computeScore(weighted_local_entropy, T_max)
        return self.paper_score


def run_one(job):
    path, opts = job
    warnings.filterwarnings("ignore")
    row = {"file": str(path)}
    try:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("cannot read image")
        h, w = img.shape
        row["orig_w"], row["orig_h"] = w, h
        if opts["max_side"] and max(h, w) > opts["max_side"]:
            s = opts["max_side"] / max(h, w)
            img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        row["run_w"], row["run_h"] = img.shape[1], img.shape[0]

        det = _Detector(downsampling_factor=opts["downsampling_factor"], num_scales=opts["num_scales"],
                        scale_start=opts["scale_start"], show_progress=False)
        m = det.detectBlur(img)
        if not np.isfinite(m).all():
            raise ValueError("blur map contains NaN/inf (flat or degenerate image?)")

        row.update(map_mean=float(m.mean()), map_median=float(np.median(m)),
                   map_p10=float(np.percentile(m, 10)), map_p90=float(np.percentile(m, 90)),
                   paper_score=float(getattr(det, "paper_score", float("nan"))))
        for lv in LEVELS:
            row[f"frac_below_{lv}"] = float((m < lv).mean())

        if opts["save_maps"]:
            out = Path(opts["save_maps"])
            out.mkdir(parents=True, exist_ok=True)
            heat = cv2.applyColorMap((np.clip(m, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
            side = np.hstack([cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), heat])
            cv2.imwrite(str(out / f"{Path(path).stem}_map.png"), side)
    except Exception as e:
        row["error"] = str(e)
    return row


def collect_images(target: Path, recursive: bool):
    if target.is_file():
        return [target]
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in target.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="image file or folder of images")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--max-side", type=int, default=640, help="resize longest side to this (0 = no resize)")
    ap.add_argument("--downsampling-factor", type=int, default=4)
    ap.add_argument("--num-scales", type=int, default=4)
    ap.add_argument("--scale-start", type=int, default=3, help="package default 3 (README example uses 2)")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes")
    ap.add_argument("--save-maps", type=Path, default=None, help="folder for image|heat-map side-by-side PNGs")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    files = collect_images(args.path, args.recursive)
    if not files:
        sys.exit(f"No images found in {args.path}")
    opts = dict(max_side=args.max_side, downsampling_factor=args.downsampling_factor,
                num_scales=args.num_scales, scale_start=args.scale_start,
                save_maps=str(args.save_maps) if args.save_maps else None)
    jobs = [(f, opts) for f in files]

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(run_one, jobs))
    else:
        rows = [run_one(j) for j in jobs]

    print(f"{'file':<40} {'run WxH':>11} {'mean':>6} {'median':>6} {'p10':>6} {'p90':>6} "
          + " ".join(f"{'<' + str(l):>6}" for l in LEVELS))
    for r in rows:
        name = Path(r["file"]).name[:40]
        if "error" in r:
            print(f"{name:<40} ERROR: {r['error']}")
            continue
        print(f"{name:<40} {str(r['run_w']) + 'x' + str(r['run_h']):>11} {r['map_mean']:>6.3f} "
              f"{r['map_median']:>6.3f} {r['map_p10']:>6.3f} {r['map_p90']:>6.3f} "
              + " ".join(f"{r[f'frac_below_{l}']:>6.2f}" for l in LEVELS))

    if args.csv:
        fields = ["file", "orig_w", "orig_h", "run_w", "run_h", "map_mean", "map_median", "map_p10",
                  "map_p90", *[f"frac_below_{l}" for l in LEVELS], "paper_score", "error"]
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved {len(rows)} rows to {args.csv}")
    print("\nNote: the map is normalised by its own maximum; these numbers are not calibrated blur scores.")


if __name__ == "__main__":
    main()

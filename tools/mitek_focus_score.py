#!/usr/bin/env python3
"""Image Focus score following Mitek US 10,275,673 B2 ("Image Focus IQA Test").

Patent text implemented here:

    Image Focus Score = (Maximum Video Gradient)
                        / [(Gray Level Dynamic Range) * (Pixel Pitch)]

    - Video Gradient          = |gray level of pixel i - gray level of pixel i+1|
                                (adjacent pixels; maximum taken "over the entire image")
    - Gray Level Dynamic Range = (average of the N lightest pixels)
                                 - (average of the N darkest pixels), N = 64
    - Pixel Pitch             = 1 / image resolution (in dpi)
    - The result is multiplied by 10 "to bring the value into the 0-1000 range".
    - Pass if score >= threshold (threshold depends on the device; the patent gives no value).

What the patent does NOT specify (so these are explicit, adjustable choices, not patent facts):
    * Gradient direction. Default: horizontal (pixel i and i+1 along a row). --direction vertical|both.
    * Colour -> gray conversion. Default: Pillow "L" (ITU-R 601 luma).
    * dpi. Read from the image metadata if present, otherwise --dpi (default 100).
      With dpi=100 a perfectly sharp edge with gradient == dynamic range scores 1000.
    * No clipping above 1000 is described for this test, so none is applied.
    * The exact equation is in an external reference (FSTC "Image Defect Metrics") that
      this script could not consult.

Because the score scales linearly with dpi, `ratio` (= max_gradient / dynamic_range, no dpi)
is also reported so images with unknown or differing dpi can be compared.

Usage:
    python mitek_focus_score.py /path/to/folder
    python mitek_focus_score.py /path/to/folder --recursive --csv out.csv --threshold 300
    python mitek_focus_score.py image.jpg --dpi 150 --direction both
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
N_PIXELS = 64  # patent: "the value of N is set to 64"


def load_gray(path: Path):
    """Return (gray uint8 array, dpi from metadata or None)."""
    with Image.open(path) as im:
        dpi = im.info.get("dpi")
        im = ImageOps.exif_transpose(im)
        gray = np.asarray(im.convert("L"), dtype=np.uint8)
    meta_dpi = None
    if dpi:
        try:
            meta_dpi = float(dpi[0])
            if meta_dpi <= 1:  # some files store 0/1 as "unknown"
                meta_dpi = None
        except (TypeError, ValueError, IndexError):
            meta_dpi = None
    return gray, meta_dpi


def max_video_gradient(gray: np.ndarray, direction: str) -> int:
    g = gray.astype(np.int16)
    values = []
    if direction in ("horizontal", "both") and g.shape[1] > 1:
        values.append(np.abs(np.diff(g, axis=1)).max())
    if direction in ("vertical", "both") and g.shape[0] > 1:
        values.append(np.abs(np.diff(g, axis=0)).max())
    return int(max(values)) if values else 0


def gray_level_dynamic_range(gray: np.ndarray, n: int = N_PIXELS) -> float:
    flat = gray.ravel().astype(np.float64)
    n = min(n, flat.size)
    darkest = np.partition(flat, n - 1)[:n].mean()
    lightest = np.partition(flat, flat.size - n)[flat.size - n:].mean()
    return float(lightest - darkest)


def focus_score(gray: np.ndarray, dpi: float, direction: str = "horizontal"):
    max_grad = max_video_gradient(gray, direction)
    dyn = gray_level_dynamic_range(gray)
    pixel_pitch = 1.0 / dpi
    if dyn <= 0:  # flat image: score undefined
        return {"max_gradient": max_grad, "dynamic_range": dyn, "pixel_pitch": pixel_pitch,
                "ratio": float("nan"), "score": float("nan")}
    ratio = max_grad / dyn
    score = ratio / pixel_pitch * 10.0
    return {"max_gradient": max_grad, "dynamic_range": dyn, "pixel_pitch": pixel_pitch,
            "ratio": ratio, "score": score}


def collect_images(target: Path, recursive: bool):
    if target.is_file():
        return [target]
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in target.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="image file or folder of images")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--dpi", type=float, default=100.0,
                    help="dpi used when the image has no dpi metadata (default 100; NOT from the patent)")
    ap.add_argument("--force-dpi", action="store_true", help="ignore image metadata and always use --dpi")
    ap.add_argument("--direction", choices=["horizontal", "vertical", "both"], default="horizontal")
    ap.add_argument("--threshold", type=float, default=None,
                    help="optional pass/fail threshold on the score (patent gives no value)")
    ap.add_argument("--csv", type=Path, default=None, help="write results to this CSV file")
    args = ap.parse_args()

    files = collect_images(args.path, args.recursive)
    if not files:
        sys.exit(f"No images found in {args.path}")

    rows = []
    for f in files:
        try:
            gray, meta_dpi = load_gray(f)
        except Exception as e:  # unreadable / corrupt image
            rows.append({"file": str(f), "error": str(e)})
            continue
        dpi = args.dpi if (args.force_dpi or meta_dpi is None) else meta_dpi
        res = focus_score(gray, dpi, args.direction)
        row = {"file": str(f), "width": gray.shape[1], "height": gray.shape[0],
               "dpi": dpi, "dpi_source": "arg" if (args.force_dpi or meta_dpi is None) else "metadata",
               **res}
        if args.threshold is not None:
            row["passed"] = bool(res["score"] >= args.threshold)
        rows.append(row)

    header = f"{'file':<48} {'WxH':>11} {'dpi':>6} {'maxGrad':>7} {'dynRng':>7} {'ratio':>6} {'score':>8}"
    if args.threshold is not None:
        header += "  pass"
    print(header)
    for r in rows:
        name = Path(r["file"]).name
        if "error" in r:
            print(f"{name:<48} ERROR: {r['error']}")
            continue
        line = (f"{name[:48]:<48} {str(r['width']) + 'x' + str(r['height']):>11} {r['dpi']:>6.0f} "
                f"{r['max_gradient']:>7d} {r['dynamic_range']:>7.1f} {r['ratio']:>6.3f} {r['score']:>8.1f}")
        if "passed" in r:
            line += f"  {'PASS' if r['passed'] else 'FAIL'}"
        print(line)

    if args.csv:
        fields = ["file", "width", "height", "dpi", "dpi_source", "max_gradient", "dynamic_range",
                  "pixel_pitch", "ratio", "score", "passed", "error"]
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()

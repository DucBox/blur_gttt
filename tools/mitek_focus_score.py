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

Resolution (dpi)
    Default (--dpi-mode card): the real physical resolution of a card image that is cropped
    tightly to the card. The card is ISO/IEC 7810 ID-1, 85.60 x 53.98 mm:
        dpi = long side in pixels / (85.60 mm / 25.4)   (and short side / 53.98 mm for a check)
    This is only valid when the image is a tight crop of the card (no background border);
    the aspect ratio is checked and a warning is printed if it is far from 1.586.
    Image-file metadata dpi (JFIF/EXIF) is NOT trusted by default: it is a camera/software
    default (e.g. 72/300) and is often inconsistent inside the same file. Use
    --dpi-mode metadata to read it anyway, or --dpi-mode fixed --dpi N for a constant.

What the patent does NOT specify (so these are explicit, adjustable choices, not patent facts):
    * Gradient direction. Default: horizontal (pixel i and i+1 along a row). --direction vertical|both.
    * Colour -> gray conversion. Default: Pillow "L" (ITU-R 601 luma).
    * Which dpi the "image resolution" refers to (see above).
    * No clipping above 1000 is described for this test, so none is applied. With a real card
      dpi (~250-350) a perfectly sharp image scores well above 1000, so the patent's
      "in focus ~ 1000, out of focus 50-100" scale cannot be assumed to hold unchanged.
    * The exact equation is in an external reference (FSTC "Image Defect Metrics") that
      this script could not consult.

`ratio` (= max_gradient / dynamic_range, dpi-independent) is also reported.

Usage:
    python mitek_focus_score.py /path/to/folder
    python mitek_focus_score.py /path/to/folder --recursive --csv out.csv --threshold 1000
    python mitek_focus_score.py image.jpg --dpi-mode fixed --dpi 100 --direction both
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
N_PIXELS = 64  # patent: "the value of N is set to 64"

MM_PER_INCH = 25.4
CARD_LONG_MM = 85.60   # ISO/IEC 7810 ID-1
CARD_SHORT_MM = 53.98
CARD_ASPECT = CARD_LONG_MM / CARD_SHORT_MM  # 1.586
ASPECT_TOLERANCE = 0.05  # relative deviation allowed before warning


def load_gray(path: Path):
    """Return (gray uint8 array, dpi from file metadata or None)."""
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


def card_dpi(height: int, width: int):
    """Physical dpi of a tightly cropped ID-1 card image.

    Returns (dpi_from_long_side, dpi_from_short_side, aspect_ratio, aspect_ok)."""
    long_px, short_px = max(height, width), min(height, width)
    dpi_long = long_px / (CARD_LONG_MM / MM_PER_INCH)
    dpi_short = short_px / (CARD_SHORT_MM / MM_PER_INCH)
    aspect = long_px / short_px
    aspect_ok = abs(aspect - CARD_ASPECT) / CARD_ASPECT <= ASPECT_TOLERANCE
    return dpi_long, dpi_short, aspect, aspect_ok


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


def resolve_dpi(mode: str, fixed_dpi: float, meta_dpi, gray: np.ndarray):
    """Return (dpi, source, aspect, note)."""
    h, w = gray.shape
    dpi_long, dpi_short, aspect, aspect_ok = card_dpi(h, w)
    note = ""
    if mode == "card":
        dpi, source = dpi_long, "card_size"
        if not aspect_ok:
            note = f"aspect {aspect:.3f} != {CARD_ASPECT:.3f}: not a tight card crop, dpi unreliable"
    elif mode == "metadata":
        if meta_dpi is None:
            dpi, source, note = fixed_dpi, "fixed(no metadata)", "no dpi in file, used --dpi"
        else:
            dpi, source = meta_dpi, "metadata"
    else:
        dpi, source = fixed_dpi, "fixed"
    return dpi, source, aspect, note


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="image file or folder of images")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--dpi-mode", choices=["card", "metadata", "fixed"], default="card",
                    help="card: derive real dpi from ID-1 card size (default); metadata: file dpi; fixed: --dpi")
    ap.add_argument("--dpi", type=float, default=100.0, help="dpi for --dpi-mode fixed (or metadata fallback)")
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
        dpi, source, aspect, note = resolve_dpi(args.dpi_mode, args.dpi, meta_dpi, gray)
        res = focus_score(gray, dpi, args.direction)
        row = {"file": str(f), "width": gray.shape[1], "height": gray.shape[0], "aspect": aspect,
               "dpi": dpi, "dpi_source": source, "note": note, **res}
        if args.threshold is not None:
            row["passed"] = bool(res["score"] >= args.threshold)
        rows.append(row)

    header = f"{'file':<40} {'WxH':>11} {'dpi':>6} {'maxGrad':>7} {'dynRng':>7} {'ratio':>6} {'score':>8}"
    if args.threshold is not None:
        header += "  pass"
    print(header)
    for r in rows:
        name = Path(r["file"]).name
        if "error" in r:
            print(f"{name:<40} ERROR: {r['error']}")
            continue
        line = (f"{name[:40]:<40} {str(r['width']) + 'x' + str(r['height']):>11} {r['dpi']:>6.0f} "
                f"{r['max_gradient']:>7d} {r['dynamic_range']:>7.1f} {r['ratio']:>6.3f} {r['score']:>8.1f}")
        if "passed" in r:
            line += f"  {'PASS' if r['passed'] else 'FAIL'}"
        if r["note"]:
            line += f"   [!] {r['note']}"
        print(line)

    if args.csv:
        fields = ["file", "width", "height", "aspect", "dpi", "dpi_source", "max_gradient", "dynamic_range",
                  "pixel_pitch", "ratio", "score", "passed", "note", "error"]
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()

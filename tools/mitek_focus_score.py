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

Real resolution (dpi) when the card does not fill the image
    The card (ISO/IEC 7810 ID-1, 85.60 x 53.98 mm) is first LOCATED in the image, then
        dpi = card long side in pixels / (85.60 mm / 25.4)
    Two ways to locate the card:
      * --boxes file.csv   : card boxes from your own card detector (most accurate).
                             Columns: file,x1,y1,x2,y2   (file = image file name)
      * automatic (default): OpenCV quadrilateral search (edges -> 4-corner contour whose
                             aspect ratio is ~1.586). Classical, can fail on busy
                             backgrounds or low-contrast card edges; failures are reported
                             as "card not found" (score = NaN), never silently guessed.
    File-metadata dpi (JFIF/EXIF) is NOT used by default (camera default, often inconsistent).

Measurement region
    --measure-region card  (default): score is computed INSIDE the located card (perspective
        corrected, inset by --inset so the card border is excluded). Reason: the maximum gradient
        is otherwise dominated by the sharp card/background edge or background objects and no
        longer says anything about text sharpness. NOTE: this differs from the patent, which
        measures "over the entire image".
    --measure-region image : whole image, as in the patent (dpi still from the located card).

What the patent does NOT specify (explicit, adjustable choices, not patent facts):
    * Gradient direction. Default: horizontal (pixel i and i+1 along a row). --direction vertical|both.
    * Colour -> gray conversion: Pillow "L" (ITU-R 601 luma).
    * Which resolution "image resolution (in dpi)" refers to (see above).
    * No clipping above 1000 is described for this test, so none is applied. With a real card
      dpi (~250-350) a perfectly sharp image scores well above 1000, so the patent's
      "in focus ~ 1000, out of focus 50-100" scale cannot be assumed to hold unchanged.
    * The exact equation is in an external reference (FSTC "Image Defect Metrics") that
      this script could not consult.

`ratio` (= max_gradient / dynamic_range, dpi-independent) is also reported.

Usage:
    python mitek_focus_score.py /path/to/folder --csv out.csv
    python mitek_focus_score.py /path/to/folder --boxes card_boxes.csv --csv out.csv
    python mitek_focus_score.py image.jpg --dpi-mode fixed --dpi 100 --measure-region image
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
N_PIXELS = 64  # patent: "the value of N is set to 64"

MM_PER_INCH = 25.4
CARD_LONG_MM = 85.60   # ISO/IEC 7810 ID-1
CARD_SHORT_MM = 53.98
CARD_ASPECT = CARD_LONG_MM / CARD_SHORT_MM  # 1.586
ASPECT_TOLERANCE = 0.12  # relative aspect deviation accepted for an auto-detected card


# ----------------------------------------------------------------------------- loading
def load_image(path: Path):
    """Return (gray uint8 array, dpi from file metadata or None)."""
    with Image.open(path) as im:
        dpi = im.info.get("dpi")
        im = ImageOps.exif_transpose(im)
        gray = np.asarray(im.convert("L"), dtype=np.uint8)
    meta_dpi = None
    if dpi:
        try:
            meta_dpi = float(dpi[0])
            if meta_dpi <= 1:
                meta_dpi = None
        except (TypeError, ValueError, IndexError):
            meta_dpi = None
    return gray, meta_dpi


# ----------------------------------------------------------------------------- card location
def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
                    dtype=np.float32)


def quad_dims(q: np.ndarray):
    """(long_px, short_px) of an ordered quad: mean of opposite edges."""
    tl, tr, br, bl = q
    horiz = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    vert = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return max(horiz, vert), min(horiz, vert)


def find_card_auto(gray: np.ndarray):
    """Classical card search. Returns ordered 4x2 corners (original pixel coords) or None."""
    h, w = gray.shape
    scale = 800.0 / max(h, w) if max(h, w) > 800 else 1.0
    small = cv2.resize(gray, (int(w * scale), int(h * scale))) if scale != 1.0 else gray
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    med = float(np.median(blur))
    edges = cv2.Canny(blur, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_area = small.shape[0] * small.shape[1]
    best, best_area = None, 0.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.08 * img_area:
            continue
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.03 * peri, True)
        if len(approx) == 4:
            quad = order_corners(approx)
        else:  # rounded corners often give >4 vertices: fall back to the min-area rectangle
            quad = order_corners(cv2.boxPoints(cv2.minAreaRect(hull)))
        long_px, short_px = quad_dims(quad)
        if short_px <= 0:
            continue
        aspect = long_px / short_px
        if abs(aspect - CARD_ASPECT) / CARD_ASPECT > ASPECT_TOLERANCE:
            continue
        quad_area = cv2.contourArea(quad)
        if quad_area > best_area:
            best, best_area = quad, quad_area
    if best is None:
        return None
    return best / scale


def quad_from_box(x1, y1, x2, y2) -> np.ndarray:
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def load_boxes(path: Path):
    boxes = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                boxes[Path(row["file"]).name] = quad_from_box(
                    float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
            except (KeyError, ValueError):
                continue
    return boxes


def warp_card(gray: np.ndarray, quad: np.ndarray, inset: float) -> np.ndarray:
    """Perspective-correct the card to an upright rectangle at its native pixel density,
    then drop an `inset` fraction on each side to exclude the card border."""
    long_px, short_px = quad_dims(quad)
    tl, tr, br, bl = quad
    horizontal = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) >= \
                 (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))
    out_w, out_h = (long_px, short_px) if horizontal else (short_px, long_px)
    out_w, out_h = max(int(round(out_w)), 2), max(int(round(out_h)), 2)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(gray, m, (out_w, out_h), flags=cv2.INTER_LINEAR)
    dx, dy = int(out_w * inset), int(out_h * inset)
    return warped[dy:out_h - dy, dx:out_w - dx] if dx * 2 < out_w and dy * 2 < out_h else warped


# ----------------------------------------------------------------------------- the patent metric
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
    if dyn <= 0:
        return {"max_gradient": max_grad, "dynamic_range": dyn, "pixel_pitch": pixel_pitch,
                "ratio": float("nan"), "score": float("nan")}
    ratio = max_grad / dyn
    return {"max_gradient": max_grad, "dynamic_range": dyn, "pixel_pitch": pixel_pitch,
            "ratio": ratio, "score": ratio / pixel_pitch * 10.0}


# ----------------------------------------------------------------------------- driver
def collect_images(target: Path, recursive: bool):
    if target.is_file():
        return [target]
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in target.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def process(path: Path, args, boxes):
    gray, meta_dpi = load_image(path)
    h, w = gray.shape
    row = {"file": str(path), "width": w, "height": h, "card_method": "", "note": ""}
    region = gray
    dpi = None

    if args.dpi_mode == "card":
        quad = boxes.get(path.name)
        if quad is not None:
            row["card_method"] = "boxes"
        else:
            quad = find_card_auto(gray)
            row["card_method"] = "auto" if quad is not None else ""
        if quad is None:
            row.update(dpi=float("nan"), dpi_source="card_size", score=float("nan"), ratio=float("nan"),
                       note="card not found (give --boxes or use --dpi-mode fixed)")
            return row
        long_px, short_px = quad_dims(quad)
        dpi = long_px / (CARD_LONG_MM / MM_PER_INCH)
        dpi_short = short_px / (CARD_SHORT_MM / MM_PER_INCH)
        row["card_long_px"], row["card_short_px"] = round(long_px, 1), round(short_px, 1)
        if abs(dpi - dpi_short) / dpi > 0.10:
            row["note"] = f"long/short side dpi differ ({dpi:.0f} vs {dpi_short:.0f}): check card box"
        row["dpi_source"] = "card_size"
        if args.measure_region == "card":
            region = warp_card(gray, quad, args.inset)
    elif args.dpi_mode == "metadata" and meta_dpi is not None:
        dpi, row["dpi_source"] = meta_dpi, "metadata"
    else:
        dpi, row["dpi_source"] = args.dpi, "fixed"
        if args.dpi_mode == "metadata":
            row["note"] = "no dpi in file, used --dpi"

    row["dpi"] = dpi
    row["region"] = "card" if region is not gray else "image"
    row.update(focus_score(region, dpi, args.direction))
    if args.threshold is not None:
        row["passed"] = bool(row["score"] >= args.threshold) if not math.isnan(row["score"]) else ""
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="image file or folder of images")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--boxes", type=Path, default=None,
                    help="CSV with card boxes (file,x1,y1,x2,y2) from your card detector")
    ap.add_argument("--dpi-mode", choices=["card", "metadata", "fixed"], default="card",
                    help="card: real dpi from the located card (default); metadata: file dpi; fixed: --dpi")
    ap.add_argument("--dpi", type=float, default=100.0, help="dpi for --dpi-mode fixed (or metadata fallback)")
    ap.add_argument("--measure-region", choices=["card", "image"], default="card")
    ap.add_argument("--inset", type=float, default=0.02,
                    help="fraction of the card width/height removed on each side before measuring")
    ap.add_argument("--direction", choices=["horizontal", "vertical", "both"], default="horizontal")
    ap.add_argument("--threshold", type=float, default=None,
                    help="optional pass/fail threshold on the score (patent gives no value)")
    ap.add_argument("--csv", type=Path, default=None, help="write results to this CSV file")
    args = ap.parse_args()

    files = collect_images(args.path, args.recursive)
    if not files:
        sys.exit(f"No images found in {args.path}")
    boxes = load_boxes(args.boxes) if args.boxes else {}

    rows = []
    for f in files:
        try:
            rows.append(process(f, args, boxes))
        except Exception as e:
            rows.append({"file": str(f), "error": str(e)})

    header = (f"{'file':<34} {'WxH':>11} {'card':>6} {'dpi':>6} {'maxGrad':>7} {'dynRng':>7} "
              f"{'ratio':>6} {'score':>8}")
    if args.threshold is not None:
        header += "  pass"
    print(header)
    for r in rows:
        name = Path(r["file"]).name
        if "error" in r:
            print(f"{name[:34]:<34} ERROR: {r['error']}")
            continue
        if math.isnan(r.get("score", float("nan"))):
            print(f"{name[:34]:<34} {str(r['width']) + 'x' + str(r['height']):>11} {r['card_method'] or '-':>6} "
                  f"{'-':>6} {'-':>7} {'-':>7} {'-':>6} {'NaN':>8}   [!] {r['note']}")
            continue
        line = (f"{name[:34]:<34} {str(r['width']) + 'x' + str(r['height']):>11} {r['card_method'] or '-':>6} "
                f"{r['dpi']:>6.0f} {r['max_gradient']:>7d} {r['dynamic_range']:>7.1f} {r['ratio']:>6.3f} "
                f"{r['score']:>8.1f}")
        if "passed" in r and r["passed"] != "":
            line += f"  {'PASS' if r['passed'] else 'FAIL'}"
        if r["note"]:
            line += f"   [!] {r['note']}"
        print(line)

    if args.csv:
        fields = ["file", "width", "height", "card_method", "card_long_px", "card_short_px", "dpi", "dpi_source",
                  "region", "max_gradient", "dynamic_range", "pixel_pitch", "ratio", "score", "passed",
                  "note", "error"]
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()

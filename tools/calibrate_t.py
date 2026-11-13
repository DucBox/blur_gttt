#!/usr/bin/env python3
"""Step 3 of the blur-labelling workflow: confirm fields without ground truth with OCR confidence + a VLM.

A field is auto-confirmed "OCR correct" only when BOTH hold:
    confidence >= T   and   the VLM read the same text as the OCR (after normalisation).
Otherwise it goes to a human. The machine never concludes "wrong".

T is calibrated on the fields that DO have ground truth (fields.csv from gt_compare.py), separately per
field group, as the smallest T whose precision (accepted cells that are truly correct / accepted cells)
reaches --target with at least --min-accepted accepted cells:
    short   : id, birthday, expiry, issue_date, sex, nationality
    name    : name (only cells checked against the accented QR name)
    address : address, birthplace, hometown, feature
mrz1..3 (checksum) and issue_by (fixed text) always have ground truth and are not part of step 3.

Inputs:
    fields.csv  from gt_compare.py
    vlm.jsonl   one line per image: {"image": "pair_x/<img>.jpg", "fields": {...}, "template_ok": true, ...}

Outputs (--out-dir):
    calibration.csv  per group: T grid with accepted cells, precision, coverage
    thresholds.csv   chosen T per group (empty when no T reaches the target)
    b3_fields.csv    every field needing step 3 with its decision: auto_correct / human
    b3_images.csv    need_b3 images -> ok (all auto-confirmed) or need_human_b3 (+ the fields to check)

Usage:
    python calibrate_t.py --fields step2/fields.csv --images step2/images.csv --vlm vlm.jsonl --out-dir step3/
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from gt_compare import norm, norm_sex

GROUPS = {
    "id": "short", "birthday": "short", "expiry": "short", "issue_date": "short",
    "sex": "short", "nationality": "short",
    "name": "name",
    "address": "address", "birthplace": "address", "hometown": "address", "feature": "address",
}
STEP3_STATUS = ("no_gt", "correct_noaccent")


def same(field, a, b):
    if field == "sex":
        return norm_sex(a) == norm_sex(b)
    return norm(a) == norm(b)


def load_vlm(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("template_ok", True):
                out[d["image"]] = d.get("fields") or {}
    return out


def calibrate(cells, target, min_accepted):
    """cells: list of (conf, vlm_match, truly_correct). Return (chosen T or None, curve rows)."""
    acc = sorted((c for c in cells if c[1]), key=lambda c: -c[0])  # VLM-matching cells, high conf first
    curve, ok, chosen = [], 0, None
    for i, (conf, _, correct) in enumerate(acc, 1):
        ok += correct
        last_of_value = i == len(acc) or acc[i][0] != conf
        if not last_of_value:
            continue
        prec = ok / i
        curve.append((conf, i, ok, prec))
        if prec >= target and i >= min_accepted:
            chosen = conf  # keeps moving down while the target still holds
    return chosen, curve


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fields", required=True, type=Path, help="fields.csv from gt_compare.py")
    ap.add_argument("--images", required=True, type=Path, help="images.csv from gt_compare.py")
    ap.add_argument("--vlm", required=True, type=Path, help="VLM output, one json per line")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--conf", choices=("conf", "min_char_conf"), default="conf",
                    help="confidence column used as the score (default conf)")
    ap.add_argument("--target", type=float, default=0.995, help="required precision (default 0.995)")
    ap.add_argument("--min-accepted", type=int, default=1000, help="min accepted cells for a T (default 1000)")
    a = ap.parse_args()

    vlm = load_vlm(a.vlm)
    with open(a.fields, encoding="utf-8-sig") as fh:
        fields = list(csv.DictReader(fh))

    def score(r):
        try:
            return float(r[a.conf])
        except ValueError:
            return 0.0

    def vlm_text(r):
        v = vlm.get(r["image"])
        return None if v is None else (v.get(r["field"]) or "")

    # ---- calibration on cells with real ground truth
    cells = defaultdict(list)
    no_vlm = 0
    for r in fields:
        g = GROUPS.get(r["field"])
        if not g or r["status"] not in ("correct", "wrong"):
            continue  # missing OCR, no_gt and accent-less names are not usable as truth
        v = vlm_text(r)
        if v is None:
            no_vlm += 1
            continue
        cells[g].append((score(r), bool(r["ocr"]) and same(r["field"], r["ocr"], v), r["status"] == "correct"))

    a.out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = {}
    with (a.out_dir / "calibration.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "T", "accepted", "accepted_correct", "precision", "coverage"])
        for g in ("short", "name", "address"):
            t, curve = calibrate(cells[g], a.target, a.min_accepted)
            thresholds[g] = t
            n = len(cells[g])
            for conf, acc, ok, prec in curve:
                w.writerow([g, "%.6f" % conf, acc, ok, "%.5f" % prec, "%.4f" % (acc / n)])
    with (a.out_dir / "thresholds.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "T", "target", "cells_with_gt"])
        for g, t in thresholds.items():
            w.writerow([g, "" if t is None else "%.6f" % t, a.target, len(cells[g])])

    # ---- apply to fields without ground truth
    b3, human = [], defaultdict(list)
    for r in fields:
        if r["status"] not in STEP3_STATUS:
            continue
        g = GROUPS.get(r["field"])
        t = thresholds.get(g)
        v = vlm_text(r)
        passed = (t is not None and v is not None and bool(r["ocr"])
                  and score(r) >= t and same(r["field"], r["ocr"], v))
        why = ("" if passed else "no_group" if not g else "no_T" if t is None else "no_vlm" if v is None
               else "vlm_differs" if not same(r["field"], r["ocr"], v) else "low_conf")
        b3.append({"image": r["image"], "document": r["document"], "field": r["field"], "group": g or "",
                   "ocr": r["ocr"], "vlm": "" if v is None else v, "score": score(r),
                   "T": "" if t is None else t, "decision": "auto_correct" if passed else "human", "reason": why})
        if not passed:
            human[r["image"]].append(r["field"])
    cols = ["image", "document", "field", "group", "ocr", "vlm", "score", "T", "decision", "reason"]
    with (a.out_dir / "b3_fields.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(b3)

    with open(a.images, encoding="utf-8-sig") as fh:
        need = [r for r in csv.DictReader(fh) if r["status"] == "need_b3"]
    with (a.out_dir / "b3_images.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "document", "status", "human_fields"])
        for r in need:
            hf = human.get(r["image"], [])
            w.writerow([r["image"], r["document"], "need_human_b3" if hf else "ok", ",".join(hf)])

    # ---- summary
    print("VLM lines: %d   GT cells without VLM: %d -> %s" % (len(vlm), no_vlm, a.out_dir), file=sys.stderr)
    for g in ("short", "name", "address"):
        t = thresholds[g]
        print("  %-8s cells=%-6d T=%s" % (g, len(cells[g]), "none (target not reached)" if t is None else "%.4f" % t),
              file=sys.stderr)
    dec = defaultdict(int)
    for r in b3:
        dec[(r["field"], r["decision"])] += 1
    for f in sorted({r["field"] for r in b3}):
        print("    %-12s auto=%-6d human=%d" % (f, dec[(f, "auto_correct")], dec[(f, "human")]), file=sys.stderr)
    n_ok = sum(1 for r in need if not human.get(r["image"]))
    print("  need_b3 images: %d -> ok %d, need_human_b3 %d" % (len(need), n_ok, len(need) - n_ok), file=sys.stderr)


if __name__ == "__main__":
    main()

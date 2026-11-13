#!/usr/bin/env python3
"""Steps 1 (end) and 2 of the blur-labelling workflow on pair folders.

Data layout (one folder per pair, 1_* = front, 2_* = back):
    <ocr-root>/pair_<id>/<img>.jpg.json
Only ground truth and comparison happen here; filtering by quality / spoof verdicts is a later step.

Step 1 end: the ground truth of a pair is built from gt.csv (gt_extract.py): MRZ of the back, QR of the
CC back / CCCD front. Group A (MRZ+QR agree), B (QR only), C (MRZ only), D (MRZ and QR disagree -> no ground truth),
E (nothing -> everything to a human). From here on every image is handled on its own.

Step 2: every field the OCR returned (plus the fields that card side must have) is compared with the
ground truth after normalisation:
    correct / wrong / missing (expected but OCR returned nothing) / no_gt /
    correct_noaccent (name only checked without accents: the accents still need step 3)
    id, birthday, sex, nationality, expiry, issue_date, name (QR) : exact match
    name (MRZ only)                                                : unaccented, prefix if MRZ truncated it
    address                                                        : CER <= --cer
    mrz1..3                                                        : correct iff the 4 check digits pass
    issue_by                                                       : fixed text printed on every card of that type

Image status:
    gt_conflict: group D, MRZ and QR disagree (pair dropped)
    need_human : group E (no ground truth at all)
    need_b4    : at least one field wrong/missing -> a human decides blur vs other cause
    need_b3    : every checked field correct, some fields have no ground truth -> step 3 (conf + VLM)
    ok         : every field correct against ground truth

Usage:
    python gt_compare.py --gt gt.csv --ocr-root .../vds_api_sample/ocr --out-dir step2/
Outputs fields.csv (one row per image x field) and images.csv (one row per image).
"""

import argparse
import csv
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# Fields each side must carry; a missing one counts as an OCR failure.
EXPECTED = {
    "CC_FRONT_CHIP": ["id", "name", "birthday", "sex", "nationality"],
    "CCCD_FRONT_CHIP": ["id", "name", "birthday", "sex", "nationality", "birthplace", "address", "expiry"],
    "CC_BACK_CHIP": ["address", "birthplace", "issue_date", "expiry", "issue_by", "mrz1", "mrz2", "mrz3"],
    "CCCD_BACK_CHIP": ["feature", "issue_date", "issue_by", "mrz1", "mrz2", "mrz3"],
}
NOT_FIELDS = {"qrcode"}
# issue_by is the same on every card of a type.
FIXED_GT = {
    ("CC_BACK_CHIP", "issue_by"): "BỘ CÔNG AN / MINISTRY OF PUBLIC SECURITY",
    ("CCCD_BACK_CHIP", "issue_by"): "CỤC TRƯỞNG CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI",
}

FIELD_COLS = ["pair_id", "image", "document", "group", "field", "ocr", "gt", "gt_source", "status", "cer",
              "conf", "min_char_conf"]
IMAGE_COLS = ["pair_id", "image", "document", "group", "status", "n_fields", "n_correct", "n_wrong", "n_no_gt", "wrong_fields", "no_gt_fields"]


# ---------------------------------------------------------------- normalisation

def norm(s):
    s = unicodedata.normalize("NFC", s or "").upper().replace("/", " / ")
    return " ".join(s.split())


def noaccent(s):
    s = unicodedata.normalize("NFD", (s or "").replace("Đ", "D").replace("đ", "d"))
    return " ".join("".join(c for c in s if unicodedata.category(c) != "Mn").upper().split())


def norm_sex(s):
    return {"NAM": "M", "M": "M", "NỮ": "F", "NU": "F", "F": "F"}.get(norm(s), norm(s))


def norm_nat(s):
    return {"VNM": "VIỆT NAM", "VIET NAM": "VIỆT NAM"}.get(norm(s), norm(s))


def norm_addr(s):
    return " ".join(norm(s).replace(",", ", ").split()).replace(" ,", ",")


def cer(ocr, gt):
    if not gt:
        return 1.0
    prev = list(range(len(gt) + 1))
    for i, a in enumerate(ocr, 1):
        cur = [i]
        for j, b in enumerate(gt, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1] / len(gt)


def mrz_name_truncated(name):
    """MRZ line 3 is SURNAME<<GIVEN<NAMES in 30 chars; longer names are cut."""
    t = name.split()
    return bool(t) and len(t[0]) + 2 + len("<".join(t[1:])) >= 30


# ---------------------------------------------------------------- ground truth per pair

def rel_of(path):
    p = Path(path)
    return Path(p.parts[-2], p.name.removesuffix(".json"))  # pair_x/<img>.jpg


def build_pair_gt(gt_csv):
    """Return {pair_id: gt dict} and the list of (pair_id, rel image) to process."""
    pairs, images = defaultdict(lambda: {"mrz": None, "qr": None}), []
    with open(gt_csv, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            rel = rel_of(r["ocr_json"])
            pid = rel.parts[0]
            images.append((pid, rel))
            p = pairs[pid]  # create the entry even when nothing is valid (group E)
            if r["mrz_valid"] == "1":
                p["mrz"] = {k[4:]: v for k, v in r.items() if k.startswith("mrz_")}
            if r["qr_ok"] == "1":
                p["qr"] = {k[3:]: v for k, v in r.items() if k.startswith("qr_")}
    for p in pairs.values():
        m, q = p["mrz"], p["qr"]
        if m and q:
            same = (m["id"] == q["id"] and m["birthday"] == q["birthday"]
                    and norm_sex(m["sex"]) == norm_sex(q["sex"]))
            p["group"] = "A" if same else "D"
        else:
            p["group"] = "B" if q else "C" if m else "E"
    return pairs, images


def gt_for(field, p, doc):
    """(gt value, source, comparison kind) or None."""
    if (doc, field) in FIXED_GT:
        return (FIXED_GT[(doc, field)], "fixed", "exact")
    if p["group"] == "D":
        return None
    m, q = p["mrz"], p["qr"]
    if field in ("mrz1", "mrz2", "mrz3"):
        return ("valid" if m else "invalid", "checksum", "mrz")
    if field == "name":
        if q:
            return (q["name"], "QR", "exact")
        if m:
            return (m["name"], "MRZ", "noaccent")
    if field in ("id", "birthday", "sex"):
        src = m or q
        if src:
            return (src[field], "MRZ" if m else "QR", field)
    if field in ("expiry", "nationality") and m:
        return (m[field], "MRZ", field)
    if field in ("issue_date", "address") and q:
        return (q[field], "QR", "cer" if field == "address" else "exact")
    return None


def compare(field, ocr, gt, kind, max_cer):
    """Return (status, cer)."""
    if kind == "mrz":
        return ("correct" if gt == "valid" else "wrong"), ""
    if not ocr:
        return "missing", ""
    if kind == "sex":
        return ("correct" if norm_sex(ocr) == norm_sex(gt) else "wrong"), ""
    if kind == "nationality":
        return ("correct" if norm_nat(ocr) == norm_nat(gt) else "wrong"), ""
    if kind == "noaccent":
        o, g = noaccent(ocr), noaccent(gt)
        ok = o == g or (mrz_name_truncated(o) and o.replace(" ", "").startswith(g.replace(" ", "")))
        return ("correct_noaccent" if ok else "wrong"), ""
    if kind == "cer":
        c = cer(norm_addr(ocr), norm_addr(gt))
        return ("correct" if c <= max_cer else "wrong"), "%.4f" % c
    return ("correct" if norm(ocr) == norm(gt) else "wrong"), ""


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- driver

def process(pid, rel, p, ocr_root, max_cer):
    info = ((load_json(ocr_root / (str(rel) + ".json")) or {}).get("information") or [{}])[0]
    doc = info.get("document", "")
    conf = info.get("confidence") or {}
    chars = info.get("alphanumeric_char_confidence") or {}
    fields = list(dict.fromkeys(EXPECTED.get(doc, []) + [k for k in conf if k not in NOT_FIELDS]))

    frows = []
    for f in fields:
        ocr = info.get(f, "") or ""
        g = gt_for(f, p, doc)
        if g is None:
            status, c, gt, src = ("missing" if not ocr else "no_gt"), "", "", ""
        else:
            gt, src, kind = g
            status, c = compare(f, ocr, gt, kind, max_cer)
        cc = chars.get(f) or []
        frows.append({"pair_id": pid, "image": str(rel), "document": doc, "group": p["group"], "field": f,
                      "ocr": ocr, "gt": gt, "gt_source": src, "status": status, "cer": c,
                      "conf": conf.get(f, ""), "min_char_conf": min(cc) if cc else ""})

    st = Counter(r["status"] for r in frows)
    wrong = [r["field"] for r in frows if r["status"] in ("wrong", "missing")]
    no_gt = [r["field"] for r in frows if r["status"] in ("no_gt", "correct_noaccent")]
    if p["group"] == "D":
        status = "gt_conflict"
    elif p["group"] == "E":
        status = "need_human"
    elif wrong:
        status = "need_b4"
    elif no_gt:
        status = "need_b3"
    else:
        status = "ok"

    irow = {"pair_id": pid, "image": str(rel), "document": doc, "group": p["group"], "status": status,
            "n_fields": len(frows), "n_correct": st["correct"], "n_wrong": len(wrong), "n_no_gt": len(no_gt),
            "wrong_fields": ",".join(wrong), "no_gt_fields": ",".join(no_gt)}
    return irow, frows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, type=Path, help="gt.csv from gt_extract.py")
    ap.add_argument("--ocr-root", required=True, type=Path, help="folder holding pair_<id>/ OCR json")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cer", type=float, default=0.05, help="max CER for address (default 0.05)")
    a = ap.parse_args()

    pairs, images = build_pair_gt(a.gt)
    try:
        from tqdm import tqdm
        images = tqdm(images, unit="img", desc="gt_compare")
    except ImportError:  # progress bar is optional
        pass
    irows, frows = [], []
    for pid, rel in images:
        i, f = process(pid, rel, pairs[pid], a.ocr_root, a.cer)
        irows.append(i)
        frows.extend(f)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, cols, rows in (("images.csv", IMAGE_COLS, irows), ("fields.csv", FIELD_COLS, frows)):
        with (a.out_dir / name).open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, cols)
            w.writeheader()
            w.writerows(rows)

    print("%d pairs (%s), %d images -> %s" % (
        len(pairs), " ".join("%s=%d" % kv for kv in sorted(Counter(p["group"] for p in pairs.values()).items())),
        len(irows), a.out_dir), file=sys.stderr)
    for doc in sorted({r["document"] for r in irows}):
        c = Counter(r["status"] for r in irows if r["document"] == doc)
        print("  %-16s n=%-6d %s" % (doc, sum(c.values()), "  ".join("%s=%d" % kv for kv in sorted(c.items()))),
              file=sys.stderr)
    print("  wrong/missing by field (gt available):", file=sys.stderr)
    by = defaultdict(Counter)
    for r in frows:
        by[(r["document"], r["field"])][r["status"]] += 1
    for (doc, f), c in sorted(by.items()):
        n = sum(v for k, v in c.items() if k != "no_gt")
        if n:
            print("    %-16s %-12s %d/%d" % (doc, f, c["wrong"] + c["missing"], n), file=sys.stderr)


if __name__ == "__main__":
    main()

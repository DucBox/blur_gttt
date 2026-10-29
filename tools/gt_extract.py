#!/usr/bin/env python3
"""Step 1 of the blur-labelling workflow: ground truth from MRZ (OCR json) and QR (image).

For every OCR json under --ocr-root (one json per jpg, `<name>.jpg.json`):
  * BACK documents: validate the MRZ (ICAO 9303 TD1, 4 check digits) and extract the fields.
  * Documents that carry the QR (CCCD_FRONT_CHIP, CC_BACK_CHIP): decode the QR from the jpg
    found under --img-root (same relative path, without the `.json` suffix) and split its fields.
  * CC_BACK_CHIP has both on one image, so QR vs MRZ (id / birthday / sex) is compared here.
    For CCCD the QR is on the front and the MRZ on the back: pair the two rows by session afterwards.

Usage:
    python gt_extract.py --ocr-root .../ocr_results/idcard-quality \
        --img-root .../external/idcard-quality --csv gt.csv
    python gt_extract.py --ocr-root ocr_output --csv gt.csv        # MRZ only, no images

Assumptions to confirm on real cards (marked ASSUMPTION below):
    * QR payload is `id|cmnd_id|name|ddmmyyyy|sex|address|ddmmyyyy(issue)`.
    * The 12-digit ID sits in the optional field of MRZ line 1 and its last 9 digits are the document number.
"""

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

QR_DOCS = {"CCCD_FRONT_CHIP", "CC_BACK_CHIP"}
CSV_FIELDS = [
    "ocr_json", "image", "document",
    "mrz_valid", "mrz_error", "mrz_id", "mrz_name", "mrz_birthday", "mrz_sex", "mrz_expiry", "mrz_nationality",
    "qr_ok", "qr_error", "qr_id", "qr_cmnd_id", "qr_name", "qr_birthday", "qr_sex", "qr_address", "qr_issue_date",
    "qr_vs_mrz",
]


# ---------------------------------------------------------------- MRZ

def _val(c):
    if c.isdigit():
        return int(c)
    if "A" <= c <= "Z":
        return ord(c) - 55
    return 0  # '<'


def check_digit(s):
    return sum(_val(c) * (7, 3, 1)[i % 3] for i, c in enumerate(s)) % 10


def _yymmdd(s, pivot_to_past):
    """YYMMDD -> dd/mm/yyyy. Birth dates after today's yy belong to the 1900s; expiry is always 20xx."""
    if len(s) != 6 or not s.isdigit():
        return ""
    yy, mm, dd = int(s[:2]), int(s[2:4]), int(s[4:])
    year = (1900 if pivot_to_past and yy > date.today().year % 100 else 2000) + yy
    try:
        date(year, mm, dd)
    except ValueError:
        return ""
    return "%02d/%02d/%04d" % (dd, mm, year)


def parse_mrz(l1, l2, l3):
    """Return (fields, error). fields is None unless every check digit passes."""
    l1, l2, l3 = (x.strip().upper().replace(" ", "") for x in (l1, l2, l3))
    if not (len(l1) == len(l2) == len(l3) == 30):
        return None, "length %d/%d/%d" % (len(l1), len(l2), len(l3))
    if not l1.startswith("ID"):
        return None, "line1 not ID"

    doc9, doc_chk = l1[5:14], l1[14]
    birth, birth_chk, sex = l2[0:6], l2[6], l2[7]
    exp, exp_chk, nat, comp_chk = l2[8:14], l2[14], l2[15:18], l2[29]
    composite = l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29]

    bad = []
    for name, data, chk in (("doc", doc9, doc_chk), ("birth", birth, birth_chk),
                            ("expiry", exp, exp_chk), ("composite", composite, comp_chk)):
        if not chk.isdigit() or check_digit(data) != int(chk):
            bad.append(name)
    if bad:
        return None, "check digit: " + ",".join(bad)

    id12 = l1[15:27]  # ASSUMPTION: 12-digit ID in the optional field
    if not (id12.isdigit() and id12[3:] == doc9):
        return None, "id12 %r does not match doc number %s" % (id12, doc9)

    surname, _, given = l3.strip("<").partition("<<")
    return {
        "id": id12,
        "name": " ".join((surname.replace("<", " ") + " " + given.replace("<", " ")).split()),
        "birthday": _yymmdd(birth, True),
        "sex": sex,
        "expiry": _yymmdd(exp, False),
        "nationality": nat,
    }, ""


# ---------------------------------------------------------------- QR

def decode_qr(img_path):
    """Return the QR text or None. Tries zxing-cpp, then OpenCV, then pyzbar; upscales on a miss."""
    import cv2  # noqa: PLC0415 - only needed when images are present
    img = cv2.imread(str(img_path))
    if img is None:
        raise IOError("cannot read image")
    variants = [img, cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)]
    try:
        import zxingcpp
        for v in variants:
            res = zxingcpp.read_barcodes(v)
            for r in res:
                if "QR" in str(r.format):
                    return r.text
    except ImportError:
        pass
    det = cv2.QRCodeDetector()
    for v in variants:
        text, _, _ = det.detectAndDecode(v)
        if text:
            return text
    try:
        from pyzbar import pyzbar
        for v in variants:
            for r in pyzbar.decode(v):
                return r.data.decode("utf-8", "replace")
    except ImportError:
        pass
    return None


def parse_qr(text):
    """ASSUMPTION: id|cmnd_id|name|ddmmyyyy|sex|address|ddmmyyyy(issue)."""
    p = text.split("|")
    if len(p) < 7:
        return None, "%d fields" % len(p)

    def d(s):
        return "%s/%s/%s" % (s[:2], s[2:4], s[4:]) if len(s) == 8 and s.isdigit() else s

    return {"id": p[0], "cmnd_id": p[1], "name": p[2], "birthday": d(p[3]), "sex": p[4],
            "address": p[5], "issue_date": d(p[6])}, ""


# ---------------------------------------------------------------- compare / driver

def norm_sex(s):
    return {"NAM": "M", "M": "M", "NỮ": "F", "NU": "F", "F": "F"}.get(s.strip().upper(), "")


def compare(mrz, qr):
    diff = [k for k in ("id", "birthday") if mrz[k] != qr[k]]
    if norm_sex(mrz["sex"]) != norm_sex(qr["sex"]):
        diff.append("sex")
    return "match" if not diff else "mismatch:" + ",".join(diff)


def process(json_path, ocr_root, img_root):
    row = dict.fromkeys(CSV_FIELDS, "")
    row["ocr_json"] = str(json_path)
    infos = json.loads(json_path.read_text(encoding="utf-8")).get("information") or []
    if not infos:
        row["mrz_error"] = "no information"
        return row
    info = infos[0]
    doc = info.get("document", "")
    row["document"] = doc

    mrz = None
    if doc.endswith("_BACK_CHIP") or info.get("mrz1"):
        mrz, err = parse_mrz(info.get("mrz1", ""), info.get("mrz2", ""), info.get("mrz3", ""))
        row["mrz_valid"] = int(mrz is not None)
        row["mrz_error"] = err
        if mrz:
            for k, v in mrz.items():
                row["mrz_" + k] = v

    qr = None
    if doc in QR_DOCS and img_root:
        rel = json_path.relative_to(ocr_root).with_suffix("")  # drops ".json", keeps ".jpg"
        img = img_root / rel
        row["image"] = str(img)
        try:
            text = decode_qr(img)
            if text is None:
                row["qr_ok"], row["qr_error"] = 0, "no QR decoded"
            else:
                qr, err = parse_qr(text)
                row["qr_ok"], row["qr_error"] = int(qr is not None), err
        except Exception as e:  # missing image / missing cv2: keep the MRZ result
            row["qr_ok"], row["qr_error"] = 0, "%s: %s" % (type(e).__name__, e)
        if qr:
            for k, v in qr.items():
                row["qr_" + k] = v
    if mrz and qr:
        row["qr_vs_mrz"] = compare(mrz, qr)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ocr-root", required=True, type=Path)
    ap.add_argument("--img-root", type=Path, help="omit to skip QR decoding")
    ap.add_argument("--csv", required=True, type=Path)
    a = ap.parse_args()

    files = sorted(a.ocr_root.rglob("*.json"))
    rows = [process(f, a.ocr_root, a.img_root) for f in files]
    with a.csv.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    backs = [r for r in rows if r["mrz_valid"] != ""]
    print("%d json, %d with MRZ, %d MRZ valid -> %s" % (
        len(rows), len(backs), sum(1 for r in backs if r["mrz_valid"] == 1), a.csv), file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Attach ground truth to FRONT images by identity (no session id needed).

Input : gt.csv from gt_extract.py (back-side MRZ / QR results) + the same OCR json folder (fronts).
Output: one row per front image with the matched person's ground truth and a match type.

Ground truth is a property of the person/card, so any back image of the same card gives the same
answer. A front is matched to a back of the SAME card family (CC with CC, CCCD with CCCD) by:
    exact_id : front OCR id == back id
    fuzzy    : id differs by <= 2 characters AND (same birthday OR same unaccented name),
               accepted only when exactly one person qualifies
    ambiguous: several people qualify  -> no ground truth, human if needed
    no_back  : nobody qualifies. In practice this is almost always a valid 12-digit id whose card simply
               has no back image in the data set (NOT a blur sign), so it gets no ground truth.

Every front also gets `id_check`: the 12-digit id encodes sex + century (4th digit) and birth year
(5th-6th digits), so it must agree with the OCR birthday / sex on the same image:
    ok | mismatch:<year,sex,...> | bad_id | incomplete
`mismatch` means at least one of id / birthday / sex is OCR-wrong; it is evidence, not ground truth.

Usage:
    python gt_pair.py --gt gt.csv --ocr-root .../ocr_results/idcard-quality --csv pairs.csv
"""

import argparse
import csv
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

OUT_FIELDS = [
    "ocr_json", "document", "match", "id_check", "gt_group", "n_backs", "back_ocr_json",
    "gt_id", "gt_name", "gt_name_noaccent", "gt_birthday", "gt_sex", "gt_expiry", "gt_nationality",
    "gt_address", "gt_issue_date",
    "ocr_id", "ocr_name", "ocr_birthday", "ocr_sex",
]


def family(doc):
    return "CCCD" if doc.startswith("CCCD_") else "CC" if doc.startswith("CC_") else ""


def noaccent(s):
    s = unicodedata.normalize("NFD", s.replace("Đ", "D").replace("đ", "d"))
    return " ".join("".join(c for c in s if unicodedata.category(c) != "Mn").upper().split())


def lev(a, b, cap=3):
    """Levenshtein distance, returns cap+1 early when it exceeds cap."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def id_check(oid, bday, sex):
    """Check OCR id vs OCR birthday / sex using the structure of the 12-digit id."""
    if not (len(oid) == 12 and oid.isdigit()):
        return "bad_id"
    code = int(oid[3])
    bad, done = [], 0
    if len(bday) == 10 and bday[6:].isdigit():
        done += 1
        if int(bday[6:]) != 1900 + 100 * (code // 2) + int(oid[4:6]):
            bad.append("year")
    if sex_key(sex):
        done += 1
        if sex_key(sex) != ("M" if code % 2 == 0 else "F"):
            bad.append("sex")
    return "mismatch:" + ",".join(bad) if bad else "ok" if done else "incomplete"


def sex_key(s):
    return {"NAM": "M", "M": "M", "NỮ": "F", "NU": "F", "F": "F"}.get(s.strip().upper(), "")


def build_people(gt_csv):
    """One record per (family, id) from the back rows; QR and MRZ of any back are merged."""
    people = {}
    with open(gt_csv, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            fam = family(r["document"])
            pid = r["mrz_id"] if r["mrz_valid"] == "1" else r["qr_id"] if r["qr_ok"] == "1" else ""
            if not (fam and r["document"].endswith("_BACK_CHIP") and pid):
                continue
            p = people.setdefault((fam, pid), {
                "id": pid, "backs": [], "mrz": False, "qr": False, "name_qr": "", "name_mrz": "",
                "birthday": "", "sex": "", "expiry": "", "nationality": "", "address": "", "issue_date": ""})
            p["backs"].append(r["ocr_json"])
            if r["mrz_valid"] == "1":
                p["mrz"] = True
                p["name_mrz"] = p["name_mrz"] or r["mrz_name"]
                for k in ("birthday", "sex", "expiry", "nationality"):
                    p[k] = p[k] or r["mrz_" + k]
            if r["qr_ok"] == "1":
                p["qr"] = True
                p["name_qr"] = p["name_qr"] or r["qr_name"]
                p["birthday"] = p["birthday"] or r["qr_birthday"]
                p["sex"] = p["sex"] or r["qr_sex"]
                p["address"] = p["address"] or r["qr_address"]
                p["issue_date"] = p["issue_date"] or r["qr_issue_date"]
    return people


def group_of(p):
    return "A" if p["mrz"] and p["qr"] else "C" if p["mrz"] else "B"


def match_front(info, fam, people, by_bday, by_name):
    oid, bday = info.get("id", ""), info.get("birthday", "")
    name = noaccent(info.get("name", ""))
    if (fam, oid) in people:
        return people[(fam, oid)], "exact_id"
    if not oid:
        cands = {}
    else:
        pool = list(by_bday.get((fam, bday), [])) + list(by_name.get((fam, name), []))
        cands = {p["id"]: p for p in pool if lev(oid, p["id"], 2) <= 2}
    if len(cands) == 1:
        return next(iter(cands.values())), "fuzzy"
    return None, "ambiguous" if cands else "no_back"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--ocr-root", required=True, type=Path)
    ap.add_argument("--csv", required=True, type=Path)
    a = ap.parse_args()

    people = build_people(a.gt)
    by_bday, by_name = defaultdict(list), defaultdict(list)
    for (fam, _), p in people.items():
        by_bday[(fam, p["birthday"])].append(p)
        by_name[(fam, noaccent(p["name_qr"] or p["name_mrz"]))].append(p)

    files = sorted(a.ocr_root.rglob("*.json"))
    try:
        from tqdm import tqdm
        files = tqdm(files, unit="file", desc="gt_pair")
    except ImportError:  # progress bar is optional
        pass
    rows = []
    for f in files:
        infos = json.loads(f.read_text(encoding="utf-8")).get("information") or [{}]
        info = infos[0]
        doc = info.get("document", "")
        if not doc.endswith("_FRONT_CHIP"):
            continue
        fam = family(doc)
        p, how = match_front(info, fam, people, by_bday, by_name)
        row = dict.fromkeys(OUT_FIELDS, "")
        row.update(ocr_json=str(f), document=doc, match=how, gt_group="" if p is None else group_of(p),
                   id_check=id_check(info.get("id", ""), info.get("birthday", ""), info.get("sex", "")),
                   ocr_id=info.get("id", ""), ocr_name=info.get("name", ""),
                   ocr_birthday=info.get("birthday", ""), ocr_sex=info.get("sex", ""))
        if p:
            row.update(n_backs=len(p["backs"]), back_ocr_json=p["backs"][0], gt_id=p["id"],
                       gt_name=p["name_qr"], gt_name_noaccent=noaccent(p["name_qr"] or p["name_mrz"]),
                       gt_birthday=p["birthday"], gt_sex=p["sex"], gt_expiry=p["expiry"],
                       gt_nationality=p["nationality"], gt_address=p["address"], gt_issue_date=p["issue_date"])
        rows.append(row)

    with a.csv.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, OUT_FIELDS)
        w.writeheader()
        w.writerows(rows)

    print("%d front json, %d people with ground truth -> %s" % (len(rows), len(people), a.csv), file=sys.stderr)
    for doc in sorted({r["document"] for r in rows}):
        c = Counter(r["match"] for r in rows if r["document"] == doc)
        k = Counter(r["id_check"].split(":")[0] for r in rows if r["document"] == doc)
        print("  %-16s n=%-5d %s" % (doc, sum(c.values()), "  ".join("%s=%d" % kv for kv in sorted(c.items()))),
              file=sys.stderr)
        print("  %-16s id_check: %s" % ("", "  ".join("%s=%d" % kv for kv in sorted(k.items()))), file=sys.stderr)


if __name__ == "__main__":
    main()

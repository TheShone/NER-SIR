#!/usr/bin/env python3
"""
convert_export.py — Korak 3: Label Studio export (JSON) → gold dataset.

Čita Label Studio JSON export i pravi gold JSONL sa entitetima kao rasponima
znakova (start/end/label/text) + provenijencija. Ovo je konačni anotirani
skup za evaluaciju (span-level, nezavisno od tokenizacije).

Primer:
  python3 scripts/convert_export.py --in data/out/annotations_export.json \
      --out data/out/gold.jsonl
"""
import argparse
import json
from collections import Counter


def extract_entities(task):
    anns = task.get("annotations") or []
    # uzmi poslednju ne-otkazanu anotaciju
    ann = None
    for a in anns:
        if a.get("was_cancelled"):
            continue
        ann = a
    if ann is None:
        return None
    ents = []
    for r in ann.get("result", []):
        if r.get("type") != "labels":
            continue
        v = r["value"]
        labs = v.get("labels") or []
        if not labs:
            continue
        ents.append({"start": int(v["start"]), "end": int(v["end"]),
                     "label": labs[0], "text": v.get("text", "")})
    ents.sort(key=lambda e: e["start"])
    return ents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.load(open(args.inp, encoding="utf-8"))
    gold, skipped = [], 0
    labels = Counter()
    n_with_ent = 0
    for task in data:
        d = task.get("data", {})
        ents = extract_entities(task)
        if ents is None:               # nije anotirano (skip/cancel)
            skipped += 1
            continue
        if ents:
            n_with_ent += 1
        for e in ents:
            labels[e["label"]] += 1
        gold.append({
            "id": d.get("sent_id", task.get("id")),
            "text": d.get("text", ""),
            "entities": ents,
            "source": {"title": d.get("source_title", ""),
                       "sl_glasnik": d.get("sl_glasnik", ""),
                       "url": d.get("source_url", "")},
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    total_ent = sum(labels.values())
    print(f"=== GOLD DATASET ===")
    print(f"  Anotiranih rečenica: {len(gold)}  (preskočeno/nedovršeno: {skipped})")
    print(f"  Sa >=1 entitetom: {n_with_ent} ({100*n_with_ent/max(len(gold),1):.0f}%)")
    print(f"  Entiteta ukupno: {total_ent}")
    for lab in ["LAW","OFFICIAL_GAZETTE","DATE","MONEY","COURT","PERSON","DECISION","REFERENCE"]:
        print(f"    {lab:18} {labels.get(lab,0)}")
    print(f"  → {args.out}")


if __name__ == "__main__":
    main()

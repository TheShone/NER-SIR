#!/usr/bin/env python3
"""
make_eval_subset.py — gušći podskup test skupa za skupe ("misleće") modele.

Cilj: smanjiti broj rečenica (a time i cenu inferencije na Colab-u) uz očuvanje
pouzdanosti po tipu. Prvo uzima SVE rečenice koje nose retke tipove
(PERSON, MONEY, REFERENCE, DECISION), pa dopunjava nasumičnim uzorkom ostalih
do ciljne veličine. Determinističko (seed).

Primer:
  python3 balansirano/scripts/make_eval_subset.py \
      --in balansirano/data/test.jsonl \
      --out balansirano/data/test_eval.jsonl --target 450
"""
import argparse
import json
import random
from collections import Counter

RARE = {"PERSON", "MONEY", "REFERENCE", "DECISION"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=450)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    def has_rare(r):
        return any(e["label"] in RARE for e in r.get("entities", []))

    keep = [r for r in rows if has_rare(r)]
    rest = [r for r in rows if not has_rare(r)]
    if len(keep) < args.target:
        keep += rest[:args.target - len(keep)]
    else:
        keep = keep[:args.target]
    rng.shuffle(keep)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    c = Counter()
    for r in keep:
        for e in r.get("entities", []):
            c[e["label"]] += 1
    print(f"Podskup: {len(keep)} rečenica")
    print(f"Entiteti po tipu: {dict(sorted(c.items(), key=lambda x:-x[1]))}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()

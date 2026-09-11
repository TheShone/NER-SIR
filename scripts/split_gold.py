#!/usr/bin/env python3
"""
split_gold.py — Korak 4a: podeli gold na few-shot skup i test skup.

Few-shot skup se bira tako da POKRIJE što više tipova entiteta (za primere u
promptu), a ostatak je test. Few-shot primeri NISU u testu.

Primer:
  python3 scripts/split_gold.py --in data/out/gold.jsonl \
      --fewshot data/out/fewshot.jsonl --test data/out/test.jsonl --k 25
"""
import argparse
import json
import random
from collections import Counter

ALL_LABELS = ["COURT","DATE","DECISION","LAW","MONEY","OFFICIAL_GAZETTE","PERSON","REFERENCE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--fewshot", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--k", type=int, default=25, help="Broj few-shot primera.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    # greedy izbor: prvo primeri koji donose NOVE tipove entiteta, pa raznoliki
    chosen, chosen_ids = [], set()
    covered = Counter()
    # 1) obezbedi bar po jedan primer za svaki tip (gde postoji)
    for lab in ALL_LABELS:
        for r in rows:
            if r["id"] in chosen_ids:
                continue
            labs = {e["label"] for e in r["entities"]}
            if lab in labs and 1 <= len(r["entities"]) <= 4:
                chosen.append(r); chosen_ids.add(r["id"])
                for e in r["entities"]:
                    covered[e["label"]] += 1
                break
    # 2) dopuni do k raznolikim primerima sa entitetima
    for r in rows:
        if len(chosen) >= args.k:
            break
        if r["id"] in chosen_ids:
            continue
        if 1 <= len(r["entities"]) <= 5:
            chosen.append(r); chosen_ids.add(r["id"])

    test = [r for r in rows if r["id"] not in chosen_ids]

    with open(args.fewshot, "w", encoding="utf-8") as f:
        for r in chosen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.test, "w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Few-shot: {len(chosen)} (pokriveni tipovi: {dict(covered)})")
    print(f"Test:     {len(test)}")
    print(f"→ {args.fewshot}\n→ {args.test}")


if __name__ == "__main__":
    main()

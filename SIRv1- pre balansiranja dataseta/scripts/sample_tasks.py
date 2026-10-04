#!/usr/bin/env python3
"""
sample_tasks.py — od pred-obeleženog punog skupa napravi GUST radni skup za
anotaciju: zadrži sve rečenice sa (pred)entitetima + dodaj deo „praznih"
(da anotator uhvati promašaje regexa), stratifikovano po izvoru.

Primer:
  python3 scripts/sample_tasks.py --in data/out/tasks8_full.json \
      --out data/out/tasks8.json --target 1500 --neg-ratio 0.30
"""
import argparse
import json
import random
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=1500)
    ap.add_argument("--neg-ratio", type=float, default=0.30,
                    help="Udeo 'praznih' rečenica u finalnom skupu.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tasks = json.load(open(args.inp, encoding="utf-8"))
    rng = random.Random(args.seed)

    pos = [t for t in tasks if t["predictions"][0]["result"]]
    neg = [t for t in tasks if not t["predictions"][0]["result"]]

    # koliko pozitivnih / negativnih
    n_neg = int(args.target * args.neg_ratio)
    n_pos = args.target - n_neg
    if len(pos) < n_pos:                      # nema dovoljno pozitivnih → uzmi sve
        n_pos = len(pos)
        n_neg = min(len(neg), args.target - n_pos)

    def strat_sample(items, k):
        by_src = defaultdict(list)
        for t in items:
            by_src[t["data"]["source_title"]].append(t)
        out = []
        total = len(items)
        for src, group in by_src.items():
            rng.shuffle(group)
            keep = round(k * len(group) / total) if total else 0
            out.extend(group[:keep])
        rng.shuffle(out)
        return out[:k]

    chosen = strat_sample(pos, n_pos) + strat_sample(neg, n_neg)
    rng.shuffle(chosen)

    json.dump(chosen, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    labels = Counter()
    per_src = Counter()
    for t in chosen:
        per_src[t["data"]["source_title"]] += 1
        for e in t["predictions"][0]["result"]:
            labels[e["value"]["labels"][0]] += 1
    print(f"=== RADNI SKUP: {len(chosen)} rečenica "
          f"({n_pos} sa entitetima + {n_neg} praznih) ===")
    print("Po izvoru:", dict(per_src))
    print("Pred-entiteti po tipu:", dict(labels))
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()

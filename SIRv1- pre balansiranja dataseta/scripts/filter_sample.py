#!/usr/bin/env python3
"""
filter_sample.py — Korak 1.5: čišćenje šuma + uzorkovanje.

Iz velikog legal_raw.jsonl (npr. 17k rečenica) izbacuje šum (naslove,
fragmente, vodenžigove) i uzorkuje na ciljani broj rečenica, STRATIFIKOVANO
po izvoru (svaki zakon zastupljen srazmerno). Rezultat je pogodan za ručnu
anotaciju i dominantno je pravni.

Primer:
  python3 scripts/filter_sample.py --in data/out/legal_raw.jsonl \
      --out data/out/legal_sampled.jsonl --target 1500
"""
import argparse
import json
import random
import re
from collections import defaultdict

# Vodenžigovi / reklame koje ubacuju komercijalni agregatori (paragraf i sl.)
JUNK_PATTERNS = [
    re.compile(r"budite na pravnoj strani", re.I),
    re.compile(r"paragraf", re.I),
    re.compile(r"www\.", re.I),
    re.compile(r"^\s*sl(užbeni)?\.?\s*glasnik", re.I),
    re.compile(r"^\s*\(?\"?\s*službeni", re.I),
]


def is_junk(text, min_tokens):
    t = text.strip()
    if len(t) < 20:
        return True
    tokens = t.split()
    if len(tokens) < min_tokens:
        return True
    if t.isupper():                      # naslovi VELIKIM slovima
        return True
    digits = sum(c.isdigit() for c in t)
    if digits > len(t) * 0.4:            # tabele/brojčane liste
        return True
    letters = sum(c.isalpha() for c in t)
    if letters < len(t) * 0.5:           # premalo slova (znaci/brojevi)
        return True
    # fragment: počinje malim slovom (a nije lista "1)"/"a)") → presečena rečenica
    if t[0].islower():
        return True
    for pat in JUNK_PATTERNS:
        if pat.search(t):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Filtriraj šum i stratifikovano uzorkuj.")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=1500, help="Ciljani broj rečenica.")
    ap.add_argument("--min-tokens", type=int, default=6, help="Min. broj reči po rečenici.")
    ap.add_argument("--seed", type=int, default=42, help="Za reproducibilnost.")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    print(f"Ulaz: {len(rows)} rečenica")

    # 1) filtriranje šuma + dedup
    seen, clean = set(), []
    for r in rows:
        t = r["text"].strip()
        if is_junk(t, args.min_tokens):
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        clean.append(r)
    print(f"Posle čišćenja i dedupa: {len(clean)} rečenica")

    # 2) grupisanje po izvoru
    by_src = defaultdict(list)
    for r in clean:
        by_src[r["source"]["title"]].append(r)

    # 3) stratifikovano uzorkovanje (srazmerno veličini izvora)
    rng = random.Random(args.seed)
    total = len(clean)
    target = min(args.target, total)
    sampled = []
    for title, group in by_src.items():
        keep = max(1, round(target * len(group) / total))
        keep = min(keep, len(group))
        rng.shuffle(group)
        sampled.extend(group[:keep])
    rng.shuffle(sampled)

    # 4) izlaz + statistika
    with open(args.out, "w", encoding="utf-8") as f:
        for r in sampled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    per_src = defaultdict(int)
    for r in sampled:
        per_src[r["source"]["title"]] += 1
    print(f"\n=== UZORAK: {len(sampled)} rečenica ===")
    for title, n in sorted(per_src.items(), key=lambda x: -x[1]):
        print(f"  {n:4d}  {title}")
    print(f"  → {args.out}")


if __name__ == "__main__":
    main()

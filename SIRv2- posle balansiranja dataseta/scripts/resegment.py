#!/usr/bin/env python3
"""
resegment.py — ponovna segmentacija gold skupa na kraće rečenice, BEZ ponovne
anotacije. Postojeći entiteti (sa char-offsetima) se PREMAPIRAJU na podrečenice.

Zašto: prvobitna segmentacija je ponegde spojila cele pasuse u jednu „rečenicu"
(do 22k znakova) — što (1) odseca „misleće" modele (reasoning potroši budžet
tokena pre odgovora) i (2) je mana skupa. Delimo na granicama rečenice koje NISU
unutar nekog entiteta; predugačke segmente bez granice sečemo na razmaku.

Primer:
  python3 balansirano/scripts/resegment.py \
      --in balansirano/data/gold_balanced.jsonl \
      --out balansirano/data/gold_seg.jsonl --max-len 600 --min-reseg 400
"""
import argparse
import json
import re
from collections import Counter

BOUND = re.compile(r"(?<=[.!?])\s+(?=[A-ZŠĐŽČĆ0-9*„\"(])")


def inside_entity(pos, ents):
    return any(e["start"] < pos < e["end"] for e in ents)


def cut_points(text, ents, max_len):
    cuts = [m.start() + 1 for m in BOUND.finditer(text)]      # tačka posle koje delimo
    cuts = [c for c in cuts if not inside_entity(c, ents)]
    bounds = [0] + cuts + [len(text)]
    bounds = sorted(set(bounds))
    # force-split predugih segmenata na razmaku van entiteta
    out = [bounds[0]]
    for b in bounds[1:]:
        a = out[-1]
        while b - a > max_len:
            # nađi razmak najbliži a+max_len koji nije u entitetu
            target = a + max_len
            pos = text.rfind(" ", a + 1, target)
            if pos <= a:
                pos = text.find(" ", target)
            if pos == -1 or pos >= b:
                break
            if inside_entity(pos, ents):
                alt = text.find(" ", pos + 1, b)
                if alt == -1:
                    break
                pos = alt
            out.append(pos); a = pos
        out.append(b)
    return sorted(set(out))


def resegment_record(rec, max_len):
    text, ents = rec["text"], rec.get("entities", [])
    bounds = cut_points(text, ents, max_len)
    segs = []
    for a, b in zip(bounds, bounds[1:]):
        seg = text[a:b]
        lead = len(seg) - len(seg.lstrip())
        sub = seg.strip()
        if len(sub) < 3:
            continue
        off = a + lead
        sub_ents = []
        for e in ents:
            if e["start"] >= off and e["end"] <= off + len(sub):
                ns, ne = e["start"] - off, e["end"] - off
                if sub[ns:ne] == e["text"]:
                    sub_ents.append({"start": ns, "end": ne,
                                     "label": e["label"], "text": e["text"]})
        segs.append({"text": sub, "entities": sub_ents})
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=600)
    ap.add_argument("--min-reseg", type=int, default=400,
                    help="Rečenice kraće od ovoga ostaju netaknute (isti id).")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    out, lost = [], 0
    for rec in rows:
        n_before = len(rec.get("entities", []))
        if len(rec["text"]) < args.min_reseg:
            out.append(rec); continue
        segs = resegment_record(rec, args.max_len)
        n_after = sum(len(s["entities"]) for s in segs)
        lost += n_before - n_after
        for j, s in enumerate(segs):
            out.append({"id": f"{rec['id']}-p{j:02d}", "text": s["text"],
                        "entities": s["entities"], "source": rec.get("source", {})})

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    c = Counter()
    for r in out:
        for e in r["entities"]:
            c[e["label"]] += 1
    L = [len(r["text"]) for r in out]
    print(f"=== RESEGMENTACIJA ===")
    print(f"  rečenica: {len(rows)} → {len(out)}")
    print(f"  entiteta: {sum(c.values())}  (izgubljeno pri remapiranju: {lost})")
    print(f"  dužina: medijana={sorted(L)[len(L)//2]}  max={max(L)}  >1000={sum(1 for x in L if x>1000)}")
    print(f"  raspodela: {dict(sorted(c.items(), key=lambda x:-x[1]))}")
    print(f"  → {args.out}")


if __name__ == "__main__":
    main()

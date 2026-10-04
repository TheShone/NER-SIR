#!/usr/bin/env python3
"""
balance_sample.py — Korak 2.5: balansiranje dataseta.

Dve funkcije:

 1) --select : entity-driven izbor NOVIH rečenica za anotaciju.
    Iz pred-obeleženih Label Studio taskova (prelabel8.py) pohlepno bira
    podskup rečenica tako da se za svaki tip dostigne CILJNI broj entiteta,
    ne birajući rečenice koje nose samo česte tipove (LAW/OFF.GAZ). Usput
    dodaje DECISION pred-oznake (regex), koje prelabel8 ne prepoznaje.
    Rezultat je fajl spreman za uvoz u Label Studio.

 2) --subsample : potkresivanje POSTOJEĆEG gold skupa.
    Smanjuje broj rečenica koje nose isključivo dominantne tipove
    (LAW/OFFICIAL_GAZETTE) na zadati budžet, bez diranja anotacija. Koristi
    se kad se posle anotacije spaja stari + novi skup u balansiran gold.

Primeri:
  # izbor novih rečenica za Label Studio
  python3 scripts/balance_sample.py --select \
      --in data/out/new_tasks8.json --out data/out/label_studio_new.json

  # potkresivanje starog gold-a pred spajanje
  python3 scripts/balance_sample.py --subsample \
      --in data/out/gold.jsonl --out data/out/gold_trimmed.jsonl \
      --cap-law 300 --cap-gazette 300
"""
import argparse
import json
import random
import re
from collections import Counter

RARE = ["PERSON", "REFERENCE", "DECISION", "MONEY", "DATE", "COURT", "OFFICIAL_GAZETTE"]

# Podrazumevani ciljni broj NOVIH entiteta po tipu (koliko dodajemo anotacijom).
DEFAULT_TARGETS = {
    "PERSON": 200, "REFERENCE": 150, "DECISION": 150, "MONEY": 150,
    "DATE": 200, "COURT": 200, "OFFICIAL_GAZETTE": 100,
}

# DECISION nije u prelabel8 — detektujemo ga ovde (imenice odluke/akta).
_DECISION_RX = re.compile(
    r"\b(?:odluk\w*|rešenj\w*|resenj\w*|presud\w*|zaključ\w*|zakljuc\w*|"
    r"nalog\w*|mišljenj\w*|misljenj\w*)\b", re.IGNORECASE)


def add_decision_spans(task):
    """Dodaj DECISION pred-oznake koje se ne preklapaju sa postojećim."""
    text = task["data"]["text"]
    res = task["predictions"][0]["result"]
    taken = [(r["value"]["start"], r["value"]["end"]) for r in res]
    idx = len(res)
    for m in _DECISION_RX.finditer(text):
        s, e = m.start(), m.end()
        if all(e <= cs or s >= ce for cs, ce in taken):
            res.append({
                "id": f"{task['data']['sent_id']}-d{idx}",
                "from_name": "label", "to_name": "text", "type": "labels",
                "value": {"start": s, "end": e, "text": text[s:e], "labels": ["DECISION"]},
            })
            taken.append((s, e)); idx += 1
    return task


def task_labels(task):
    return Counter(r["value"]["labels"][0] for r in task["predictions"][0]["result"])


def select(args):
    tasks = json.load(open(args.inp, encoding="utf-8"))
    for t in tasks:
        add_decision_spans(t)

    targets = dict(DEFAULT_TARGETS)
    remaining = dict(targets)
    random.seed(42)
    random.shuffle(tasks)  # da izbor ne bude pristrasan po redosledu izvora

    chosen, used = [], set()

    # RAZNOVRSNOST PERSON: prvo garantuj kvotu iz kadrovskih rešenja (mnogo
    # različitih imena), da PERSON klasa ne bi bila skoro same sudije iz
    # odluka US. Uzmi do `--kadrovska-min` rečenica sa PERSON pred-oznakom.
    kad = [i for i, t in enumerate(tasks)
           if "kadrovska" in t["data"].get("source_title", "").lower()
           and task_labels(t).get("PERSON", 0) > 0]
    for i in kad[:args.kadrovska_min]:
        used.add(i); chosen.append(tasks[i])
        for lab, c in task_labels(tasks[i]).items():
            if lab in remaining:
                remaining[lab] = max(0, remaining[lab] - c)
    # Pohlepno: u svakom krugu uzmi task koji pokriva najviše JOŠ POTREBNIH
    # retkih entiteta; ponavljaj dok ima potrebe i kandidata.
    while any(v > 0 for v in remaining.values()):
        best, best_score = None, 0
        for i, t in enumerate(tasks):
            if i in used:
                continue
            lc = task_labels(t)
            score = sum(min(lc.get(lab, 0), remaining[lab]) for lab in RARE if remaining[lab] > 0)
            if score > best_score:
                best, best_score = i, score
        if best is None or best_score == 0:
            break
        used.add(best)
        chosen.append(tasks[best])
        for lab, c in task_labels(tasks[best]).items():
            if lab in remaining:
                remaining[lab] = max(0, remaining[lab] - c)

    got = Counter()
    for t in chosen:
        got.update(task_labels(t))
    json.dump(chosen, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("=== IZBOR NOVIH REČENICA (za Label Studio) ===")
    print(f"  Rečenica izabrano: {len(chosen)}")
    print(f"  Ciljevi:  {targets}")
    print(f"  Sakupljeno (pred-oznake): {dict(got)}")
    print(f"  Nedostignuto: { {k:v for k,v in remaining.items() if v>0} or 'sve dostignuto'}")
    print(f"  → {args.out}")


def subsample(args):
    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    caps = {"LAW": args.cap_law, "OFFICIAL_GAZETTE": args.cap_gazette}
    random.seed(42)
    random.shuffle(rows)

    kept, counts = [], Counter()
    # Zadrži sve rečenice sa bar jednim ne-dominantnim tipom; rečenice koje
    # nose ISKLJUČIVO LAW/OFF.GAZ zadrži samo do budžeta.
    dom = set(caps)
    for r in rows:
        labs = Counter(e["label"] for e in r.get("entities", []))
        only_dominant = labs and all(l in dom for l in labs)
        if only_dominant:
            # da li još ima mesta u budžetu za te dominantne tipove?
            if all(counts[l] + n <= caps[l] for l, n in labs.items()):
                kept.append(r); counts.update(labs)
        else:
            kept.append(r); counts.update(labs)
    random.shuffle(kept)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("=== POTKRESAN STARI GOLD ===")
    print(f"  Rečenica: {len(rows)} → {len(kept)}")
    print(f"  Raspodela posle: {dict(counts)}")
    print(f"  → {args.out}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--select", action="store_true", help="Izbor novih rečenica za anotaciju.")
    g.add_argument("--subsample", action="store_true", help="Potkresivanje starog gold-a.")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-law", type=int, default=300)
    ap.add_argument("--cap-gazette", type=int, default=300)
    ap.add_argument("--kadrovska-min", type=int, default=60,
                    help="Najmanje rečenica iz kadrovskih rešenja (raznovrsnost PERSON imena).")
    args = ap.parse_args()
    if args.select:
        select(args)
    else:
        subsample(args)


if __name__ == "__main__":
    main()

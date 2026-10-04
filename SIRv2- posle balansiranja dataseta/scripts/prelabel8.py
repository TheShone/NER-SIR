#!/usr/bin/env python3
"""
prelabel8.py — Korak 2 (8-entitetska pravna shema).

Pred-obeležava rečenice po shemi NER4Legal_SRB (8 klasa) i izvozi u Label
Studio. Kombinuje:
  - REGEX (visoka preciznost, bez pristrasnosti): DATE, MONEY,
    OFFICIAL_GAZETTE, LAW, REFERENCE
  - MODEL bcms-bertic-ner: PERSON (PER) i COURT (ORG koji sadrži „sud")

NE koristi NER4Legal_SRB za pred-obeležavanje (jer se taj model OCENJUJE →
izbegavamo pristrasnost). Čovek zatim ispravlja u Label Studio.

Zavisnosti: transformers, torch.
Primer:
  python3 scripts/prelabel8.py --in data/out/legal_clean.jsonl \
      --out data/out/tasks8.json
"""
import argparse
import json
import re
import sys

LABELS = ["COURT", "DATE", "DECISION", "LAW", "MONEY",
          "OFFICIAL_GAZETTE", "PERSON", "REFERENCE"]

MESECI = (r"januar\w*|februar\w*|mart\w*|april\w*|maj\w*|jun\w*|jul\w*|"
          r"avgust\w*|septembar|septembra|oktobar|oktobra|novembar|novembra|"
          r"decembar|decembra")

# Prioritet pri preklapanju (veći = jači); duži raspon takođe pobeđuje.
PRIORITY = {"OFFICIAL_GAZETTE": 6, "REFERENCE": 5, "LAW": 4,
            "MONEY": 3, "DATE": 2, "COURT": 1, "PERSON": 1, "DECISION": 1}

REGEXES = [
    ("OFFICIAL_GAZETTE", re.compile(
        r"[„\"']?\s*Sl(?:\.|užbeni)?\s*glasnik[^)”\"']*?\d+/\d{2,4}[^)”\"']*",
        re.IGNORECASE)),
    ("DATE", re.compile(
        r"\b\d{1,2}\.\s*(?:" + MESECI + r")\s*\d{0,4}\.?(?:\s*godine)?", re.IGNORECASE)),
    ("DATE", re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{4}\.?")),
    ("MONEY", re.compile(
        r"\b\d[\d.,]*\s*(?:dinar\w*|din\.?|RSD|evr\w*|eur\w*|EUR|€)\b", re.IGNORECASE)),
    ("REFERENCE", re.compile(r"\b[A-ZŠĐŽČĆ][a-zžšđčć]?\.?\s*\d+/\d{2,4}\b")),
    ("LAW", re.compile(
        r"\b(?:ovog|ovim|ovaj|istog|navedenog|predmetnog)\s+zakon\w*", re.IGNORECASE)),
    ("LAW", re.compile(r"\bKrivičn\w+\s+zakonik\w*", re.IGNORECASE)),
    ("LAW", re.compile(r"\bUstav\w*", re.IGNORECASE)),
    ("LAW", re.compile(
        r"\bZakonik?\w*\s+o\s+[a-zšđžčćA-ZŠĐŽČĆ]+(?:\s+[a-zšđžčć]+){0,4}")),
]


def regex_spans(text):
    spans = []
    for label, rx in REGEXES:
        for m in rx.finditer(text):
            s, e = m.start(), m.end()
            while e > s and text[e - 1] in " .,);":
                e -= 1
            if e > s:
                spans.append((s, e, label))
    return spans


def model_spans(text, nlp):
    spans = []
    for ent in nlp(text):
        grp = ent.get("entity_group", "")
        word = ent.get("word", "")
        if grp == "PER":
            spans.append((ent["start"], ent["end"], "PERSON"))
        elif grp == "ORG" and "sud" in word.lower():
            spans.append((ent["start"], ent["end"], "COURT"))
    return spans


def resolve_overlaps(spans):
    """Zadrži nepreklapajuće raspone; kod preklapanja jači prioritet / duži."""
    spans = sorted(spans, key=lambda x: (-(x[1] - x[0]), -PRIORITY[x[2]]))
    chosen = []
    for s, e, lab in spans:
        if all(e <= cs or s >= ce for cs, ce, _ in chosen):
            chosen.append((s, e, lab))
    return sorted(chosen)


def to_task(rec, spans):
    results = []
    for i, (s, e, lab) in enumerate(spans):
        results.append({
            "id": f"{rec['id']}-e{i}",
            "from_name": "label", "to_name": "text", "type": "labels",
            "value": {"start": s, "end": e, "text": rec["text"][s:e], "labels": [lab]},
        })
    src = rec.get("source", {})
    return {
        "data": {"text": rec["text"], "sent_id": rec["id"],
                 "source_title": src.get("title", ""), "sl_glasnik": src.get("sl_glasnik", ""),
                 "source_url": src.get("url", "")},
        "predictions": [{"model_version": "regex+bcms-bertic-ner", "result": results}],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    from transformers import pipeline
    print(f"Učitavam bcms-bertic-ner (device={args.device}) …", file=sys.stderr)
    nlp = pipeline("token-classification", model="classla/bcms-bertic-ner",
                   aggregation_strategy="simple",
                   device=(args.device if args.device == "mps" else -1))

    tasks, n_ent = [], 0
    from collections import Counter
    by_label = Counter()
    for i, rec in enumerate(rows):
        text = rec["text"]
        spans = resolve_overlaps(regex_spans(text) + model_spans(text, nlp))
        for _, _, lab in spans:
            by_label[lab] += 1
        n_ent += len(spans)
        tasks.append(to_task(rec, spans))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(rows)}", file=sys.stderr)

    json.dump(tasks, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    with_ent = sum(1 for t in tasks if t["predictions"][0]["result"])
    print(f"\n=== GOTOVO ===")
    print(f"  Rečenica: {len(tasks)} | sa >=1 entitetom: {with_ent} ({100*with_ent/len(tasks):.0f}%)")
    print(f"  Entiteta ukupno: {n_ent} | po tipu: {dict(by_label)}")
    print(f"  → {args.out}")


if __name__ == "__main__":
    main()

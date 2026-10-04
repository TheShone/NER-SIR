#!/usr/bin/env python3
"""
evaluate.py — evaluacija predikcija naspram gold-a (surface-level, po rečenici).

Dve metrike:
  STROGO  — isti tip + identičan tekst (normalizovan: bez razmaka/##, lowercase)
  RELAXED — isti tip + preklapanje (jedan tekst sadrži drugi) → pravedno prema
            razlikama u granicama spanova (npr. „Sl. glasnik RS" vs pun navod)

Poredi više modela u jednoj tabeli (mikro P/R/F1 + F1 po tipu).
Primer:
  python3 scripts/evaluate.py --gold data/out/test.jsonl --pred \
      data/out/pred_qwen3-4b_instruct_fs.jsonl data/out/pred_ner4legal.jsonl
"""
import argparse
import json
import re
from collections import defaultdict

LABELS = ["COURT","DATE","DECISION","LAW","MONEY","OFFICIAL_GAZETTE","PERSON","REFERENCE"]


def norm(s):
    s = s.replace("##", "").lower()
    s = re.sub(r"\s+", "", s)
    return s.strip("„“”'\"().,;:*")


def match(gt, pt, relaxed):
    if gt == pt:
        return True
    if relaxed and gt and pt and (gt in pt or pt in gt):
        return True
    return False


def score(gold_by_id, pred_rows, relaxed):
    tp = fp = fn = 0
    by = defaultdict(lambda: [0, 0, 0])   # tip -> [tp, fp, fn]
    for r in pred_rows:
        gold = [(e["label"].upper(), norm(e["text"])) for e in gold_by_id.get(r["id"], [])]
        pred = [(e["label"].upper(), norm(e["text"])) for e in r.get("pred", [])]
        used = [False] * len(pred)
        for gl, gt in gold:
            hit = -1
            for j, (pl, pt) in enumerate(pred):
                if not used[j] and pl == gl and match(gt, pt, relaxed):
                    hit = j; break
            if hit >= 0:
                used[hit] = True; tp += 1; by[gl][0] += 1
            else:
                fn += 1; by[gl][2] += 1
        for j, (pl, pt) in enumerate(pred):
            if not used[j]:
                fp += 1; by[pl][1] += 1
    return tp, fp, fn, by


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def report(title, gold_by_id, preds, relaxed):
    print(f"\n########## {title} ##########")
    print(f"{'Model':32} {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 54)
    per_model = {}
    for name, rows in preds.items():
        tp, fp, fn, by = score(gold_by_id, rows, relaxed)
        p, r, f = prf(tp, fp, fn)
        per_model[name] = by
        print(f"{name:32} {p:6.3f} {r:6.3f} {f:6.3f}")
    print("\nF1 po tipu:")
    print("  " + f"{'TIP':18}" + "".join(f"{n[:13]:>14}" for n in per_model))
    for lab in LABELS:
        row = f"  {lab:18}"
        for name, by in per_model.items():
            _, _, f = prf(*by.get(lab, [0, 0, 0]))
            row += f"{f:14.3f}"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", nargs="+", required=True)
    args = ap.parse_args()

    gold_by_id = {r["id"]: r["entities"]
                  for r in (json.loads(l) for l in open(args.gold, encoding="utf-8") if l.strip())}
    preds = {}
    for path in args.pred:
        name = path.split("/")[-1].replace("pred_", "").replace(".jsonl", "").replace("qwen3-", "")
        preds[name] = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    report("STROGO (identičan tekst)", gold_by_id, preds, relaxed=False)
    report("RELAXED (preklapanje granica)", gold_by_id, preds, relaxed=True)


if __name__ == "__main__":
    main()

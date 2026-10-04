#!/usr/bin/env python3
"""
run_ner4legal.py — pokreće fino-podešen model NER4Legal_SRB (Kalušev & Brkljač)
lokalno i snima predikcije u istom formatu kao ostali modeli.

Model je sačuvan sa generičkim LABEL_0..14; ovde postavljamo pravo BIO
mapiranje (iz README-a modela) pa koristimo pipeline sa agregacijom. Labela
"OFFICIAL GAZZETE" se preslikava na našu "OFFICIAL_GAZETTE".

Zavisnosti: transformers, torch (isti .venv kao za prelabel).
Primer:
  python3 scripts/run_ner4legal.py --test data/out/test.jsonl \
      --out data/out/pred_ner4legal.jsonl
"""
import argparse
import json
import sys

ID2LABEL = {
    0: "O",
    1: "B-COURT", 2: "B-DATE", 3: "B-DECISION", 4: "B-LAW", 5: "B-MONEY",
    6: "B-OFFICIAL_GAZETTE", 7: "B-PERSON", 8: "B-REFERENCE",
    9: "I-COURT", 10: "I-LAW", 11: "I-MONEY", 12: "I-OFFICIAL_GAZETTE",
    13: "I-PERSON", 14: "I-REFERENCE",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="kalusev/NER4Legal_SRB")
    ap.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    print(f"Učitavam {args.model} …", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    model.eval()
    dev = "mps" if args.device == "mps" else "cpu"
    model.to(dev)

    def extract(text):
        """Ručno BIO dekodiranje sa char-offsetima → čisti spanovi iz originala."""
        enc = tok(text, return_offsets_mapping=True, truncation=True,
                  max_length=512, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            logits = model(**{k: v.to(dev) for k, v in enc.items()}).logits[0]
        preds = logits.argmax(-1).tolist()
        spans, cur = [], None   # cur = [tip, start, end]
        for (s, e), pid in zip(offsets, preds):
            if s == e:                      # specijalni token
                continue
            lab = ID2LABEL.get(pid, "O")
            typ = None if lab == "O" else lab[2:]
            # spoji uzastopne tokene ISTOG tipa (model daje B- na svakom tokenu)
            if typ is None:
                if cur:
                    spans.append(cur); cur = None
            elif cur and cur[0] == typ:
                cur[2] = e                  # produži raspon
            else:
                if cur:
                    spans.append(cur)
                cur = [typ, s, e]
        if cur:
            spans.append(cur)
        return [{"label": typ, "text": text[s:e]} for typ, s, e in spans]

    test = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()]
    from collections import Counter
    labels = Counter()
    with open(args.out, "w", encoding="utf-8") as f:
        for i, r in enumerate(test):
            preds = []
            try:
                preds = extract(r["text"])
                for p in preds:
                    labels[p["label"]] += 1
            except Exception as ex:
                print(f"  greška {r['id']}: {ex}", file=sys.stderr)
            f.write(json.dumps({"id": r["id"], "text": r["text"], "pred": preds},
                               ensure_ascii=False) + "\n")
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(test)}", file=sys.stderr)
    print(f"\nGotovo → {args.out}")
    print("Predikcija po tipu:", dict(labels))


if __name__ == "__main__":
    main()

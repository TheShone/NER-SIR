#!/usr/bin/env python3
"""
run_qwen_ollama.py — Korak 4b: NER promptovanjem Qwen3 preko Ollame.

Za svaku test rečenicu traži od modela da izdvoji 8 pravnih entiteta i vrati
JSON. Podržava instruct (bez razmišljanja) i thinking mod, i zero/few-shot.

Preduslov: `ollama serve` radi i model je povučen (npr. `ollama pull qwen3:8b`).

Primeri:
  python3 scripts/run_qwen_ollama.py --model qwen3:8b --test data/out/test.jsonl \
      --out data/out/pred_qwen3-8b_instruct_zs.jsonl
  python3 scripts/run_qwen_ollama.py --model qwen3:8b --think \
      --fewshot data/out/fewshot.jsonl --shots 8 \
      --test data/out/test.jsonl --out data/out/pred_qwen3-8b_think_fs.jsonl
"""
import argparse
import json
import re
import sys
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"

LABELS_OPIS = (
    "COURT (naziv suda), DATE (kalendarski datum), DECISION (odluka/presuda/rešenje), "
    "LAW (naziv ili skraćenica propisa, npr. „ovog zakona\", „Krivični zakonik\", „Ustav\"), "
    "MONEY (novčani iznos), OFFICIAL_GAZETTE (navod Službenog glasnika), "
    "PERSON (ime lica), REFERENCE (alfanumerička oznaka predmeta/akta)"
)

SYSTEM = (
    "Ti si sistem za prepoznavanje imenovanih entiteta (NER) u srpskim pravnim "
    "tekstovima. Iz date rečenice izdvoj isključivo entitete sledećih tipova "
    "(koristi ISKLJUČIVO ove oznake, nijednu drugu): "
    + LABELS_OPIS + ". "
    "NE OZNAČAVAJ: funkcije i zvanja (npr. predsednik, ministar, poslodavac, "
    "zaposleni, direktor), institucije koje nisu sud (npr. Vlada, Narodna "
    "skupština, ministarstvo), interne pozive na članove i stavove (npr. „član "
    "5.\", „stava 2.\", „ovog člana\"), niti opšte pojmove. "
    "LAW obuhvata pozive na propis kao celinu („ovog zakona\", „ovim zakonom\", "
    "„Krivični zakonik\", „Ustav\"), a NE pojedine članove. "
    "Vrati ISKLJUČIVO JSON niz objekata oblika "
    '{"text": <tačan tekst iz rečenice>, "label": <TIP>}. '
    "Tekst mora biti doslovno prepisan iz rečenice. Ako nema entiteta, vrati []. "
    "Bez ikakvog objašnjenja, samo JSON."
)


def build_messages(text, shots):
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex in shots:
        ents = [{"text": e["text"], "label": e["label"]} for e in ex["entities"]]
        msgs.append({"role": "user", "content": ex["text"]})
        msgs.append({"role": "assistant", "content": json.dumps(ents, ensure_ascii=False)})
    msgs.append({"role": "user", "content": text})
    return msgs


def call_ollama(model, messages, think):
    payload = {
        "model": model, "messages": messages, "stream": False,
        "think": bool(think),
        "options": {"temperature": 0},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    return resp.get("message", {}).get("content", "")


def parse_entities(content):
    # ukloni eventualni <think>...</think> i markdown ograde
    content = re.sub(r"(?s)<think>.*?</think>", " ", content)
    content = content.strip().strip("`")
    m = re.search(r"\[.*\]", content, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for o in arr:
        if isinstance(o, dict) and o.get("text") and o.get("label"):
            out.append({"text": str(o["text"]), "label": str(o["label"]).upper()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="npr. qwen3:8b")
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fewshot", help="fewshot.jsonl (za few-shot)")
    ap.add_argument("--shots", type=int, default=0, help="broj primera iz fewshot skupa")
    ap.add_argument("--think", action="store_true", help="uključi thinking mod")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    test = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()]
    if args.limit:
        test = test[:args.limit]
    shots = []
    if args.shots and args.fewshot:
        shots = [json.loads(l) for l in open(args.fewshot, encoding="utf-8") if l.strip()][:args.shots]

    mode = "think" if args.think else "instruct"
    print(f"Model {args.model} | {mode} | shots={len(shots)} | rečenica={len(test)}", file=sys.stderr)
    t0 = time.time()
    with open(args.out, "w", encoding="utf-8") as f:
        for i, r in enumerate(test):
            try:
                content = call_ollama(args.model, build_messages(r["text"], shots), args.think)
                preds = parse_entities(content)
            except Exception as e:
                print(f"  greška na {r['id']}: {e}", file=sys.stderr)
                preds = []
            f.write(json.dumps({"id": r["id"], "text": r["text"], "pred": preds},
                               ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                dt = time.time() - t0
                print(f"  {i+1}/{len(test)}  ({dt/(i+1):.1f}s/reč)", file=sys.stderr)
    print(f"Gotovo za {time.time()-t0:.0f}s → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

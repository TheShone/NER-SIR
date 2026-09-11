#!/usr/bin/env python3
"""
colab_run_all.py — pokreće Qwen3-2507 instruct/thinking modele na Colab GPU.

Koristi ZASEBNE modele (čist split, matches "vrsta modela: instruct/thinking"):
  4B:  qwen3:4b-instruct        / qwen3:4b-thinking
  30B: qwen3:30b-a3b-instruct   / qwen3:30b-a3b-thinking   (MoE, 3B aktivnih)

Instruct je brz (~0.4s/reč) → pun test. Thinking je spor (~40s/reč) →
koristi manji test fajl (test_eval.jsonl). Snima pred_*.jsonl po konfiguraciji.

KORIŠĆENJE:
  # instruct na punom testu (brzo):
  !python colab_run_all.py --models 4b,30b --mode instruct --test test.jsonl
  # thinking na uzorku (sporo):
  !python colab_run_all.py --models 4b,30b --mode thinking --test test_eval.jsonl
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

CHAT = "http://localhost:11434/api/chat"
TAGS = "http://localhost:11434/api/tags"

MODELS = {
    "4b":  {"instruct": "qwen3:4b-instruct",      "thinking": "qwen3:4b-thinking"},
    "30b": {"instruct": "qwen3:30b-instruct", "thinking": "qwen3:30b-thinking"},
}

LABELS_OPIS = (
    "COURT (naziv suda), DATE (kalendarski datum), DECISION (odluka/presuda/rešenje), "
    "LAW (naziv ili skraćenica propisa, npr. „ovog zakona\", „Krivični zakonik\", „Ustav\"), "
    "MONEY (novčani iznos), OFFICIAL_GAZETTE (navod Službenog glasnika), "
    "PERSON (ime lica), REFERENCE (alfanumerička oznaka predmeta/akta)"
)
SYSTEM = (
    "Ti si sistem za prepoznavanje imenovanih entiteta (NER) u srpskim pravnim "
    "tekstovima. Iz date rečenice izdvoj isključivo entitete sledećih tipova "
    "(koristi ISKLJUČIVO ove oznake, nijednu drugu): " + LABELS_OPIS + ". "
    "NE OZNAČAVAJ: funkcije i zvanja (predsednik, ministar, poslodavac, zaposleni, "
    "direktor), institucije koje nisu sud (Vlada, Narodna skupština, ministarstvo), "
    "interne pozive na članove i stavove („član 5.\", „stava 2.\", „ovog člana\"), "
    "niti opšte pojmove. LAW je poziv na propis kao celinu, ne pojedine članove. "
    "Vrati ISKLJUČIVO JSON niz objekata oblika "
    '{"text": <tačan tekst iz rečenice>, "label": <TIP>}. '
    "Tekst doslovno prepiši iz rečenice. Ako nema entiteta, vrati []. Samo JSON."
)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def ensure_ollama():
    if sh("which ollama").returncode != 0:
        print("Instaliram Ollamu…", flush=True)
        sh("apt-get -qq install -y zstd pciutils lshw")
        sh("curl -fsSL https://ollama.com/install.sh | sh")
    try:
        urllib.request.urlopen(TAGS, timeout=3); return
    except Exception:
        pass
    subprocess.Popen("ollama serve", shell=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            urllib.request.urlopen(TAGS, timeout=3)
            print("Ollama radi.", flush=True); return
        except Exception:
            time.sleep(2)
    sys.exit("Ollama server se nije podigao.")


def pull(model):
    print(f"Povlačim {model}…", flush=True)
    if sh(f"ollama pull {model}").returncode != 0:
        print(f"  ⚠️ ne mogu da povučem {model} — proveri tačan naziv taga.", flush=True)
        return False
    return True


def build_messages(text, shots):
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex in shots:
        ents = [{"text": e["text"], "label": e["label"]} for e in ex["entities"]]
        msgs.append({"role": "user", "content": ex["text"]})
        msgs.append({"role": "assistant", "content": json.dumps(ents, ensure_ascii=False)})
    msgs.append({"role": "user", "content": text})
    return msgs


def call(model, messages, think, num_predict):
    payload = {"model": model, "messages": messages, "stream": False,
               "think": bool(think),
               "options": {"temperature": 0, "num_predict": num_predict}}
    req = urllib.request.Request(CHAT, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r).get("message", {}).get("content", "")


def parse(content):
    content = re.sub(r"(?s)<think>.*?</think>", " ", content).strip().strip("`")
    m = re.search(r"\[.*\]", content, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    return [{"text": str(o["text"]), "label": str(o["label"]).upper()}
            for o in arr if isinstance(o, dict) and o.get("text") and o.get("label")]


def run(model, think, test, shots, out, num_predict):
    print(f"\n>>> {model} | rečenica={len(test)} | num_predict={num_predict}", flush=True)
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as f:
        for i, r in enumerate(test):
            try:
                preds = parse(call(model, build_messages(r["text"], shots), think, num_predict))
            except Exception as e:
                print(f"  greška {r['id']}: {e}", flush=True); preds = []
            f.write(json.dumps({"id": r["id"], "text": r["text"], "pred": preds},
                               ensure_ascii=False) + "\n")
            f.flush()
            if i == 0 or (i + 1) % 25 == 0:
                dt = time.time() - t0
                print(f"  {i+1}/{len(test)}  {dt/(i+1):.1f}s/reč", flush=True)
    print(f"<<< gotovo za {(time.time()-t0)/60:.1f} min → {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="4b,30b")
    ap.add_argument("--mode", choices=["instruct", "thinking", "both"], default="both")
    ap.add_argument("--test", default="test.jsonl")
    ap.add_argument("--fewshot", default="fewshot.jsonl")
    ap.add_argument("--shots", type=int, default=6)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    ensure_ollama()
    test = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()]
    shots = [json.loads(l) for l in open(args.fewshot, encoding="utf-8") if l.strip()][:args.shots]
    os.makedirs(args.outdir, exist_ok=True)
    modes = ["instruct", "thinking"] if args.mode == "both" else [args.mode]

    for size in [m.strip() for m in args.models.split(",") if m.strip()]:
        for mode in modes:
            tag = MODELS.get(size, {}).get(mode)
            if not tag or not pull(tag):
                continue
            think = (mode == "thinking")
            npred = 4096 if think else 512
            out = os.path.join(args.outdir, f"pred_qwen3-{size}_{mode}_fs.jsonl")
            run(tag, think, test, shots, out, npred)
    print("\n=== SVE GOTOVO ===", flush=True)


if __name__ == "__main__":
    main()

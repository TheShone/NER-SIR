#!/usr/bin/env python3
"""
colab_run_all.py — (balansirano, runda 2) pokreće Qwen3-2507 modele na Colab GPU.

Zasebni instruct/thinking modeli (čist split "vrsta modela"):
  4B:  qwen3:4b-instruct        / qwen3:4b-thinking
  30B: qwen3:30b-instruct       / qwen3:30b-thinking   (MoE A3B, 3B aktivnih)

Instruct je brz  → PUN test  (--test test.jsonl, 1127 rečenica).
Thinking je spor → PODSKUP    (--test-eval test_eval.jsonl, 450 rečenica).
U modu `both` svaki tip automatski koristi svoj fajl.

RESUME: ako pred_*.jsonl već postoji, preskače ID-jeve koji su gotovi i
nastavlja (bezbedno posle prekida sesije). Piše inkrementalno + flush.

PLAN (2 iteracije, svaka ≤5h, L4 GPU — NE A100):
  # Iteracija 1 (~4h): 4B instr+think (pun/podskup) + 30B instr (pun)
  !python colab_run_all.py --models 4b --mode both   --outdir $OUT
  !python colab_run_all.py --models 30b --mode instruct --outdir $OUT
  # Iteracija 2 (~5h): 30B thinking (podskup)
  !python colab_run_all.py --models 30b --mode thinking --outdir $OUT
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
    "4b":  {"instruct": "qwen3:4b-instruct",  "thinking": "qwen3:4b-thinking"},
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


def load_done(out):
    """Za RESUME: pročitaj validne redove, prepiši fajl čisto (bez pokvarenog
    poslednjeg reda posle prekida), vrati skup gotovih ID-jeva."""
    if not os.path.exists(out):
        return set()
    valid = []
    for line in open(out, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            valid.append(json.loads(line))
        except Exception:
            pass  # odbaci nepotpun red
    with open(out, "w", encoding="utf-8") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    done = {r["id"] for r in valid}
    if done:
        print(f"  RESUME: već gotovo {len(done)} — preskačem ih.", flush=True)
    return done


def mirror_to(out, mirror):
    """Kopiraj trenutni pred fajl na Drive (Drive FUSE sinhronizuje na close)."""
    if not mirror:
        return
    try:
        import shutil
        os.makedirs(mirror, exist_ok=True)
        shutil.copy(out, os.path.join(mirror, os.path.basename(out)))
    except Exception as e:
        print(f"  ⚠️ mirror na Drive nije uspeo: {e}", flush=True)


def run(model, think, test, shots, out, num_predict, mirror=None):
    # RESUME preko sesija: ako lokalno nema, a na Drive-u ima → povuci nazad
    if mirror and not os.path.exists(out):
        src = os.path.join(mirror, os.path.basename(out))
        if os.path.exists(src):
            import shutil
            shutil.copy(src, out)
    done = load_done(out)
    todo = [r for r in test if r["id"] not in done]
    print(f"\n>>> {model} | ukupno={len(test)} | preostalo={len(todo)} | "
          f"num_predict={num_predict}", flush=True)
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo):
            try:
                preds = parse(call(model, build_messages(r["text"], shots), think, num_predict))
            except Exception as e:
                print(f"  greška {r['id']}: {e}", flush=True); preds = []
            f.write(json.dumps({"id": r["id"], "text": r["text"], "pred": preds},
                               ensure_ascii=False) + "\n")
            f.flush()
            if i == 0 or (i + 1) % 25 == 0:
                dt = time.time() - t0
                print(f"  {i+1}/{len(todo)}  {dt/(i+1):.1f}s/reč", flush=True)
                mirror_to(out, mirror)   # periodično čuvanje na Drive
    mirror_to(out, mirror)
    print(f"<<< gotovo za {(time.time()-t0)/60:.1f} min → {out} (mirror: {mirror})", flush=True)


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="4b,30b")
    ap.add_argument("--mode", choices=["instruct", "thinking", "both"], default="both")
    ap.add_argument("--test", default="test.jsonl",
                    help="PUN test za instruct modele.")
    ap.add_argument("--test-eval", dest="test_eval", default="test_eval.jsonl",
                    help="PODSKUP za thinking modele.")
    ap.add_argument("--fewshot", default="fewshot.jsonl")
    ap.add_argument("--shots", type=int, default=6)
    ap.add_argument("--outdir", default=".", help="LOKALNI izlaz (brz upis).")
    ap.add_argument("--mirror", default=None,
                    help="Drive folder — periodično kopiranje pred fajlova (bezbedno na prekid).")
    args = ap.parse_args()

    ensure_ollama()
    shots = load_jsonl(args.fewshot)[:args.shots]
    os.makedirs(args.outdir, exist_ok=True)
    modes = ["instruct", "thinking"] if args.mode == "both" else [args.mode]

    for size in [m.strip() for m in args.models.split(",") if m.strip()]:
        for mode in modes:
            tag = MODELS.get(size, {}).get(mode)
            if not tag or not pull(tag):
                continue
            think = (mode == "thinking")
            # instruct → pun test; thinking → podskup
            test_path = args.test_eval if think else args.test
            test = load_jsonl(test_path)
            # veći budžet tokena: thinking chain + odgovor mora stati (inače
            # se JSON odgovor odseče → prazno). Posle resegmentacije rečenice
            # su kratke, ali držimo rezervu.
            npred = 16384 if think else 1024
            out = os.path.join(args.outdir, f"pred_qwen3-{size}_{mode}_fs.jsonl")
            print(f"[{size}/{mode}] test={os.path.basename(test_path)} ({len(test)} reč.)", flush=True)
            run(tag, think, test, shots, out, npred, mirror=args.mirror)
    print("\n=== SVE GOTOVO ===", flush=True)


if __name__ == "__main__":
    main()

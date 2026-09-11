# NER u srpskoj legislativi promptovanjem LLM-ova — prilozi uz SIR rad

Autor: Nenad Pavlović · 2026.

Ovaj paket sadrži rad, anotirani skup podataka, izvorni kod i rezultate
eksperimenta opisanog u radu „Ekstrakcija imenovanih entiteta u tekstovima na
srpskom jeziku".

## Sadržaj paketa

```
rad/            NER_SIR_rad.docx — rad
dataset/        anotirani skup, splitovi i smernice za anotaciju
  gold.jsonl              1.400 anotiranih rečenica (8 tipova entiteta)
  test.jsonl             test skup (1.375 rečenica)
  test_eval.jsonl        gušći podskup za „misleće" modele (452 rečenice)
  fewshot.jsonl          25 few-shot primera (isključeni iz testa)
  ANOTACIJA_smernice.md  smernice i shema 8 entiteta
scripts/        izvorni kod (pipeline)
rezultati/      predikcije svih sistema (pred_*.jsonl)
```

## Shema entiteta (8 tipova)
COURT, DATE, DECISION, LAW, MONEY, OFFICIAL_GAZETTE, PERSON, REFERENCE.

## Okruženje
Python 3.9 (arm64), zavisnosti: `requests`, `pypdf`, `python-docx`,
`transformers`, `torch`. Generativni modeli pokretani preko alata Ollama.

## Reprodukcija (redosled)

1. Prikupljanje i čišćenje teksta zakona:
   `python scripts/fetch_clean.py --mode local --input-dir data/raw_legal --out data/out/legal_raw.jsonl`
2. Filtriranje šuma i uzorkovanje:
   `python scripts/filter_sample.py --in data/out/legal_raw.jsonl --out data/out/legal_clean.jsonl --target 100000`
3. Poluautomatsko predobeležavanje (8 klasa, regex + bcms-bertic):
   `python scripts/prelabel8.py --in data/out/legal_clean.jsonl --out data/out/tasks8_full.json`
4. Formiranje radnog skupa i ručna anotacija (Label Studio, config `scripts/labelstudio_config_8.xml`):
   `python scripts/sample_tasks.py --in data/out/tasks8_full.json --out data/out/tasks8.json`
5. Konverzija anotacija u gold: `python scripts/convert_export.py --in <export.json> --out data/out/gold.jsonl`
6. Podela na test/few-shot: `python scripts/split_gold.py --in data/out/gold.jsonl --fewshot data/out/fewshot.jsonl --test data/out/test.jsonl`
7. Pokretanje modela:
   - Qwen3 (instruct/thinking): `scripts/colab_run_all.py` (GPU) ili `scripts/run_qwen_ollama.py` (lokalno)
   - NER4Legal_SRB (baseline): `python scripts/run_ner4legal.py --test data/out/test.jsonl --out data/out/pred_ner4legal.jsonl`
8. Evaluacija (strogo + relaxed, po tipu):
   `python scripts/evaluate.py --gold data/out/test_eval.jsonl --pred rezultati/pred_*.jsonl`

## Napomena
Sirovi tekstovi zakona (izvor: Pravno-informacioni sistem RS) nisu uključeni;
prema Zakonu o autorskom i srodnim pravima (čl. 6) oni su van autorske zaštite
i slobodno dostupni. Anotacije su licencirane pod CC BY-SA 4.0.

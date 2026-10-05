# Scrubber — On-Device PII Redaction with a Fine-Tuned Small Model

## 1. Problem

Banks, insurers, hospitals and SaaS companies want to analyze, share, log or send text to AI tools,
but that text (support tickets, claim notes, emails, logs) is full of customer personal data that
can't leave the network. Rule-based redactors (regex, Presidio) miss context-dependent PII; a
prompted small model is inconsistent; frontier APIs are off-limits for exactly this data.

Scrubber fine-tunes a ~4B open model **on a laptop** to find PII, then redacts it with
deterministic code. It runs fully offline through MLX or Ollama.

## 2. Goals / Non-goals

**Goals**
- LoRA fine-tune (QLoRA on a 4-bit base) on a 16GB Apple Silicon Mac with `mlx-lm`.
- Beat both a rule-based baseline (Presidio/regex) and the same base model prompted, on leak rate.
- Show domain adaptation: a small set of in-domain examples (bank, claims, logs) measurably helps.
- Ship the fine-tuned model to Ollama and expose it as a CLI and a local redaction API.

**Non-goals (v1)**
- Compliance certification (HIPAA/GDPR). Scrubber reduces risk; it doesn't guarantee zero leaks.
- Images/PDF OCR (ClearBill or another tool can feed text in).
- Non-English training. The data supports 8 languages; English first.

## 3. Hardware budget (M1 Pro, 16GB)

| Stage | Model | Notes |
|---|---|---|
| Train (QLoRA) | `mlx-community/Qwen3-4B-Instruct-2507-4bit` | batch 2, grad checkpointing, max_seq 1024 |
| Smoke test | `mlx-community/Qwen3-1.7B-4bit` | minutes per run, for pipeline checks |
| Serve | fused → GGUF f16 → `ollama create -q q4_K_M` | ~2.5 GB in Ollama |

Close other heavy apps while training; watch Memory Pressure.

## 4. Design principle: the model finds, code redacts

The model outputs **only a JSON list of entities**, never rewritten text:

```json
{"entities": [{"text": "Maria Lopez", "label": "NAME"}, {"text": "4111 1111 1111 1111", "label": "CARD"}]}
```

`redact.py` then:
1. rejects any entity whose text doesn't occur in the input (hallucination guard),
2. finds every occurrence, resolves overlaps (longest span wins),
3. replaces each with a stable placeholder (`[NAME_1]` — same value, same number),
4. returns a mapping so a caller can **restore** values after, e.g., a cloud LLM responds.

The model can't corrupt the document, and every redaction is auditable.

## 5. Labels

| Label | Covers (ai4privacy source labels) |
|---|---|
| NAME | GIVENNAME, SURNAME (adjacent parts merged: "Maria Lopez") |
| EMAIL | EMAIL |
| PHONE | TELEPHONENUM |
| ADDRESS | STREET, BUILDINGNUM, CITY, ZIPCODE |
| DATE | DATE (dates of birth, service dates) |
| GOV_ID | SOCIALNUM, TAXNUM, PASSPORTNUM, IDCARDNUM, DRIVERLICENSENUM |
| CARD | CREDITCARDNUMBER |
| ACCOUNT | bank account, member ID, MRN, policy number (domain data) |
| AGE | AGE |

Dropped (not treated as PII here): TIME, SEX, TITLE, ORGANISATIONPLACEHOLDER. The mapping lives in
`labels.py`; unknown source labels are counted and reported by `prep`.

## 6. Data

- **General:** `ai4privacy/open-pii-masking-500k-ai4privacy` (CC-BY-4.0; attribution in README).
  Fields used: `source_text`, `privacy_mask[{label,start,end,value}]`, `language`.
  Default: English, 6,000 train / 300 valid from `train`, 500 test from `validation`.
- **Domain:** `domain_synth.py` generates bank support tickets, insurance claim notes, app logs and
  emails with Faker, with exact character spans. It includes hard negatives that are *not* PII
  (transaction IDs, CPT codes, dollar amounts, company names, ticket numbers).
  Default: 1,200 train / 100 valid / 300 test (`test_domain`).
- **Canonical record** (`data/raw/*.jsonl`): `{"text", "spans":[{"start","end","label","value"}], "source"}`.
- **Training record** (`data/mlx/*.jsonl`): `{"messages":[system, user(text), assistant(JSON)]}`,
  trained with `mask_prompt: true` so loss is on the answer only.

Entities in the target are listed once per unique (text, label), in order of first appearance.

## 7. Training

`configs/lora_qwen3_4b.yaml` (mlx-lm): LoRA rank 8 on 16 layers, lr 5e-5, batch 2,
grad accumulation 4, 1,500 iters, max_seq_length 1024, grad checkpointing, eval every 100.
Keep the adapter with the best validation loss (saved every 200 steps).

## 8. Evaluation (`scrubber eval`)

Run every backend on `test` (general) and `test_domain`:

| Metric | Definition | Why |
|---|---|---|
| **Leak rate** (headline) | % of gold PII values still present verbatim after redaction | what a breach looks like |
| Char recall | % of gold PII characters covered by redactions | partial credit for boundaries |
| Char precision | % of redacted characters that were gold PII | over-redaction hurts usefulness |
| Entity F1 | exact (text, label) match | label quality |
| Per-label recall | value-level, per label | where it fails |
| Hallucination rate | % of predicted entities not found in the text | model honesty |
| Parse failures, latency | invalid JSON outputs, s/doc | operability |

Backends: `regex`, `presidio`, `ollama` (base model, prompted), `mlx` (base or + adapter),
`ollama` (fine-tuned export), `oracle` (gold; sanity check = 0% leak).

**Targets:** fine-tuned leak rate < 50% of the prompted base model's, and lower than Presidio's,
on both test sets; char precision ≥ 0.85; parse failures < 1%.

## 9. Phases

1. **Data + baselines:** `prep`, regex/Presidio/prompted-base eval. (scaffolded)
2. **Fine-tune:** smoke run on 1.7B, full run on 4B, eval adapter via MLX. (scaffolded)
3. **Ablation:** general-only vs general+domain training; data size 1k/3k/6k.
4. **Ship:** fuse → GGUF → Ollama; CLI + FastAPI `/redact` and `/restore`. (scaffolded)
5. **Polish:** results table + charts in README, model card, integrate with ClearBill.

## 10. Repo layout

```
src/scrubber/  labels.py prompt.py redact.py data.py domain_synth.py backends.py metrics.py cli.py api.py
configs/       lora_qwen3_4b.yaml lora_smoke_1_7b.yaml
scripts/       export_ollama.sh Modelfile.template
tests/         test_redact.py test_data.py test_metrics.py
```

## 11. Conventions (for Claude Code)

- Python 3.11+, type hints, no network calls at inference time except `localhost:11434`.
- `redact.py` and `metrics.py` stay pure and fully unit-tested.
- The system prompt lives only in `prompt.py`; training and inference must use the same one.
- Never log raw text in eval output; per-example dumps go to `results/` (git-ignored).

# 🧽 Scrubber

**On-device PII redaction with a small open model fine-tuned on a laptop.**

Scrubber takes a 4B open model (Qwen3-4B-Instruct), fine-tunes it with LoRA on a 16GB MacBook
using Apple's MLX, and turns it into a redactor that strips names, emails, phone numbers, card
numbers, SSNs, addresses, account and member IDs out of support tickets, claim notes, logs and
emails. Nothing leaves the machine.

```
"Pt Maria Lopez, 58 y/o, MRN-4417820, seen 03/14/2026. Contact 555-201-8890. Copay $25.00."
                                   ↓
"Pt [NAME_1], [AGE_1] y/o, [ACCOUNT_1], seen [DATE_1]. Contact [PHONE_1]. Copay $25.00."
```

## Why

Companies holding customer data want to analyze, share, log or send text to AI tools, but that
text is full of personal information that can't leave their network. Rule-based tools miss
context-dependent PII, and a prompted small model is inconsistent. Scrubber fine-tunes a small
model for this one job, so it can run anywhere the data already lives.

## How it works

**The model finds, code redacts.** The model outputs only a list of entities:

```json
{"entities": [{"text": "Maria Lopez", "label": "NAME"}, {"text": "MRN-4417820", "label": "ACCOUNT"}]}
```

Deterministic code then:
- rejects anything the model "found" that isn't actually in the text,
- replaces every occurrence on whole-token boundaries,
- returns a placeholder mapping so you can **restore** the values later.

The model can never alter the rest of your document.

## Results

Domain test set: 300 synthetic bank, claims, log and email texts, half from sentence templates never
seen in training. **Leak rate** is the share of personal data still readable after redaction.

| Backend | Leak rate ↓ | Char recall | Char precision | Entity F1 |
|---|---|---|---|---|
| Regex rules | 80.1% | 26.0% | 94.2% | 30.9% |
| Microsoft Presidio | 33.1% | 74.9% | 65.0% | 47.7% |
| Qwen3-4B, prompted (no training) | *run it* | | | |
| **Qwen3-4B + LoRA (Scrubber)** | *run it* | | | |

Regex and Presidio rows were measured with `scrubber eval` (your numbers may differ slightly with other Faker versions). Fill in the model rows from your own
runs with `scrubber compare results/*.json`.

## Quick start (Apple Silicon, 16GB)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[mlx,presidio,dev]"
python -m spacy download en_core_web_lg      # for the Presidio baseline
make test

# 1. Data: ai4privacy (English) + synthetic domain examples
make prep

# 2. Baselines
make baselines                               # regex, Presidio, prompted base model via Ollama

# 3. Fine-tune
make smoke                                   # Qwen3-1.7B, ~minutes: proves the pipeline
make train                                   # Qwen3-4B QLoRA, the real run
make eval-adapter                            # score the adapter directly in MLX

# 4. Ship to Ollama and use it
make export                                  # fuse → GGUF → `ollama create scrubber`
scrubber redact "Hi, I'm Maria Lopez, card 4111 1111 1111 1111"
scrubber serve                               # POST /redact, POST /restore on 127.0.0.1:8765
```

## Adapting it to your company

The general dataset teaches what PII is. Your own context is taught in
`src/scrubber/domain_synth.py`: add templates shaped like your real tickets, logs and forms (with
fake values), including **hard negatives**. These are things that look sensitive but aren't, like
order IDs, product codes and internal hostnames. Re-run `make prep train`. `make ablation`
measures how much the domain data helps.

## Memory tips for 16GB Macs

- Training uses a 4-bit base, batch 2, gradient checkpointing and 1,024-token sequences.
- If you see heavy swap, use `batch_size: 1`, `num_layers: 8` and `max_seq_length: 768` in the config.
- Quit Ollama (`ollama stop <model>`) and browsers with many tabs while training.

## Limits

Scrubber lowers risk; it doesn't guarantee zero leaks. Check the per-label recall in your eval
output and keep a human review step for high-stakes releases.

## Attribution

General training data: [ai4privacy/open-pii-masking-500k-ai4privacy](https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy)
(CC-BY-4.0). Base model: Qwen3-4B-Instruct-2507 (Apache-2.0).

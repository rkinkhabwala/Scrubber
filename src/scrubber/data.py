"""Dataset preparation.

ai4privacy rows → canonical records → chat JSONL for mlx_lm.lora.

Canonical record: {"text": str, "spans": [{"start","end","label","value"}], "source": str}
"""
from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

from .labels import MERGE_ADJACENT, SOURCE_TO_LABEL
from .prompt import SYSTEM_PROMPT, target_json

HF_DATASET = "ai4privacy/open-pii-masking-500k-ai4privacy"


def convert_row(row: dict, unknown: Counter | None = None) -> dict | None:
    """Map one ai4privacy row to a canonical record. Returns None if offsets don't check out."""
    text = row["source_text"]
    mask = row["privacy_mask"]
    if isinstance(mask, str):
        mask = json.loads(mask)
    spans = []
    for m in sorted(mask, key=lambda m: m["start"]):
        src = m["label"]
        if src not in SOURCE_TO_LABEL:
            if unknown is not None:
                unknown[src] += 1
            continue
        label = SOURCE_TO_LABEL[src]
        if label is None:
            continue
        s, e = int(m["start"]), int(m["end"])
        if text[s:e] != m["value"]:
            return None  # corrupt offsets: skip the row rather than train on bad labels
        spans.append({"start": s, "end": e, "label": label, "value": m["value"]})
    return {"text": text, "spans": merge_adjacent(text, spans), "source": "ai4privacy"}


def merge_adjacent(text: str, spans: list[dict]) -> list[dict]:
    """Merge 'Maria' + 'Lopez' (separated by spaces only) into one NAME span."""
    out: list[dict] = []
    for sp in spans:
        prev = out[-1] if out else None
        if (prev and prev["label"] == sp["label"] and sp["label"] in MERGE_ADJACENT
                and text[prev["end"]:sp["start"]].strip() == "" and sp["start"] - prev["end"] <= 2):
            prev["end"] = sp["end"]
            prev["value"] = text[prev["start"]:prev["end"]]
        else:
            out.append(dict(sp))
    return out


def gold_entities(record: dict) -> list[tuple[str, str]]:
    """Unique (value, label) pairs in order of first appearance — the training target."""
    seen: dict[tuple[str, str], None] = {}
    for sp in sorted(record["spans"], key=lambda s: s["start"]):
        seen.setdefault((sp["value"], sp["label"]), None)
    return list(seen)


def to_chat(record: dict) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": record["text"]},
        {"role": "assistant", "content": target_json(gold_entities(record))},
    ]}


def iter_ai4privacy(split: str, languages: set[str], local_path: str | None = None) -> Iterator[dict]:
    """Stream rows from Hugging Face (or a local .jsonl/.parquet export of the same dataset)."""
    if local_path:
        p = Path(local_path)
        if p.suffix == ".parquet":
            import pyarrow.parquet as pq

            rows: Iterable[dict] = pq.read_table(p).to_pylist()
        else:
            rows = (json.loads(line) for line in p.open())
    else:
        from datasets import load_dataset

        rows = load_dataset(HF_DATASET, split=split, streaming=True)
    for row in rows:
        if not languages or row.get("language") in languages:
            yield row


def take_records(rows: Iterator[dict], n: int, max_chars: int, unknown: Counter) -> list[dict]:
    out = []
    for row in rows:
        if len(row["source_text"]) > max_chars:
            continue
        rec = convert_row(row, unknown)
        if rec is not None:
            out.append(rec)
            if len(out) >= n:
                break
    return out


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def prepare(
    out_dir: Path,
    n_train: int = 6000,
    n_valid: int = 300,
    n_test: int = 500,
    domain_train: int = 1200,
    domain_valid: int = 100,
    domain_test: int = 300,
    languages: set[str] | None = None,
    max_chars: int = 1500,
    seed: int = 7,
    train_file: str | None = None,
    test_file: str | None = None,
) -> dict[str, int]:
    from .domain_synth import HELDOUT_TEMPLATES, generate

    languages = languages if languages is not None else {"en"}
    unknown: Counter = Counter()
    rng = random.Random(seed)

    if train_file and not test_file:
        # One local file: carve the test set off the end so it never overlaps training data.
        rows = take_records(iter_ai4privacy("train", languages, train_file), n_train + n_valid + n_test,
                            max_chars, unknown)
        general, test = rows[: n_train + n_valid], rows[n_train + n_valid:]
    else:
        general = take_records(iter_ai4privacy("train", languages, train_file), n_train + n_valid,
                               max_chars, unknown)
        test = take_records(iter_ai4privacy("validation", languages, test_file), n_test, max_chars, unknown)

    rng.shuffle(general)
    g_valid, g_train = general[:n_valid], general[n_valid:]

    d_all = generate(domain_train + domain_valid, seed=seed)
    d_train = d_all[:domain_train]
    d_valid = d_all[domain_train:]
    # Test: fresh values, half from training templates and half from never-seen templates.
    d_test = (generate(domain_test - domain_test // 2, seed=seed + 1)
              + generate(domain_test // 2, seed=seed + 2, templates=HELDOUT_TEMPLATES))
    for r in d_test[domain_test - domain_test // 2:]:
        r["source"] = "domain_heldout"

    train = g_train + d_train
    valid = g_valid + d_valid
    rng.shuffle(train)
    rng.shuffle(valid)

    raw, mlx = out_dir / "raw", out_dir / "mlx"
    counts = {
        "raw/train": write_jsonl(raw / "train.jsonl", train),
        "raw/valid": write_jsonl(raw / "valid.jsonl", valid),
        "raw/test": write_jsonl(raw / "test.jsonl", test),
        "raw/test_domain": write_jsonl(raw / "test_domain.jsonl", d_test),
        "raw/train_general_only": write_jsonl(raw / "train_general_only.jsonl", g_train),
        "mlx/train": write_jsonl(mlx / "train.jsonl", map(to_chat, train)),
        "mlx/valid": write_jsonl(mlx / "valid.jsonl", map(to_chat, valid)),
    }
    # Ablation set: same general data, no domain examples (spec §9 phase 3).
    write_jsonl(out_dir / "mlx_general_only" / "train.jsonl", map(to_chat, g_train))
    write_jsonl(out_dir / "mlx_general_only" / "valid.jsonl", map(to_chat, g_valid))

    label_counts = Counter(sp["label"] for r in train for sp in r["spans"])
    (out_dir / "stats.json").write_text(json.dumps(
        {"counts": counts, "train_label_counts": label_counts, "unknown_source_labels": unknown}, indent=2))
    return counts

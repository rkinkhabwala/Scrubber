"""Scrubber command line.

  scrubber prep                       build data/raw and data/mlx from ai4privacy + domain synth
  scrubber eval --backend regex       score a backend on test / test_domain
  scrubber compare results/*.json     markdown table for the README
  scrubber redact "text" --backend ollama --model scrubber
  scrubber serve                      local redaction API on :8765
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import typer

from .backends import make_backend
from .data import prepare, read_jsonl
from .metrics import Tally, score
from .redact import redact as do_redact

app = typer.Typer(add_completion=False, help="On-device PII redaction with a fine-tuned small model.")


@app.command()
def prep(
    out: Path = typer.Option(Path("data")),
    n_train: int = 6000,
    n_valid: int = 300,
    n_test: int = 500,
    domain_train: int = 1200,
    domain_valid: int = 100,
    domain_test: int = 300,
    languages: str = typer.Option("en", help="Comma-separated, or 'all'"),
    train_file: str = typer.Option(None, help="Local .jsonl/.parquet export instead of downloading"),
    test_file: str = typer.Option(None),
):
    """Download/convert data and write training files."""
    langs = set() if languages == "all" else set(languages.split(","))
    counts = prepare(out, n_train, n_valid, n_test, domain_train, domain_valid, domain_test, langs,
                     train_file=train_file, test_file=test_file)
    for k, v in counts.items():
        typer.echo(f"{k:28s} {v:6d}")
    typer.echo(f"Label counts and unknown source labels: {out / 'stats.json'}")


@app.command("eval")
def evaluate(
    backend: str = typer.Option(..., help="regex | presidio | ollama | mlx | oracle"),
    model: str = typer.Option(None),
    adapter: str = typer.Option(None, help="LoRA adapter dir (mlx backend)"),
    data: Path = typer.Option(Path("data/raw")),
    splits: str = typer.Option("test,test_domain"),
    limit: int = typer.Option(None),
    tag: str = typer.Option(None, help="Name for this run in results/"),
    out_dir: Path = typer.Option(Path("results")),
):
    """Score a backend. Writes results/<tag>.json and a per-example dump for error analysis."""
    all_records = {s: read_jsonl(data / f"{s}.jsonl")[:limit] for s in splits.split(",")}
    b = make_backend(backend, model, adapter, records=[r for rs in all_records.values() for r in rs])
    tag = tag or b.name.replace(":", "_").replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"backend": b.name, "splits": {}}
    with (out_dir / f"{tag}.examples.jsonl").open("w") as dump:
        for split, records in all_records.items():
            t = Tally()
            with typer.progressbar(records, label=f"{b.name} on {split}", file=sys.stderr) as bar:
                for rec in bar:
                    start = time.perf_counter()
                    res = b.predict(rec["text"])
                    secs = time.perf_counter() - start
                    red = do_redact(rec["text"], res.entities)
                    score(rec, res.entities, red, res.parsed, secs, t)
                    dump.write(json.dumps({
                        "split": split, "text": rec["text"], "redacted": red.text,
                        "gold": [s["value"] for s in rec["spans"]],
                        "pred": [e.model_dump() for e in res.entities],
                        "rejected": [e.model_dump() for e in red.rejected], "parsed": res.parsed,
                    }, ensure_ascii=False) + "\n")
            report["splits"][split] = t.summary()
    (out_dir / f"{tag}.json").write_text(json.dumps(report, indent=2))
    _print_summary(report)


def _print_summary(report: dict) -> None:
    typer.echo(f"\n== {report['backend']} ==")
    for split, s in report["splits"].items():
        typer.echo(f"[{split}] docs={s['docs']}  leak={s['leak_rate']:.1%}  char_recall={s['char_recall']:.1%}  "
                   f"char_precision={s['char_precision']:.1%}  entity_f1={s['entity_f1']:.1%}  "
                   f"halluc={s['hallucination_rate']:.1%}  parse_fail={s['parse_failure_rate']:.1%}  "
                   f"{s['sec_per_doc']:.2f}s/doc")
        typer.echo("   per-label recall: " + ", ".join(f"{k} {v:.0%}" for k, v in s["per_label_recall"].items()))


@app.command()
def compare(files: list[Path]):
    """Print a markdown comparison table of eval runs."""
    rows = []
    for f in files:
        if f.name.endswith(".examples.jsonl"):
            continue
        r = json.loads(f.read_text())
        for split, s in r["splits"].items():
            rows.append((r["backend"], split, s))
    typer.echo("| Backend | Split | Leak rate ↓ | Char recall | Char precision | Entity F1 | s/doc |")
    typer.echo("|---|---|---|---|---|---|---|")
    for name, split, s in sorted(rows, key=lambda x: (x[1], -x[2]["leak_rate"])):
        typer.echo(f"| {name} | {split} | {s['leak_rate']:.1%} | {s['char_recall']:.1%} | "
                   f"{s['char_precision']:.1%} | {s['entity_f1']:.1%} | {s['sec_per_doc']:.2f} |")


@app.command()
def redact(
    text: str = typer.Argument(None, help="Text to redact (or pipe via stdin)"),
    backend: str = typer.Option("ollama"),
    model: str = typer.Option("scrubber"),
    adapter: str = typer.Option(None),
    show_mapping: bool = typer.Option(False, help="Print placeholder → original mapping (contains PII!)"),
):
    """Redact one text."""
    text = text if text is not None else sys.stdin.read()
    b = make_backend(backend, model, adapter)
    red = do_redact(text, b.predict(text).entities)
    typer.echo(red.text)
    if show_mapping:
        typer.echo(json.dumps(red.mapping, indent=2), err=True)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8765, backend: str = "ollama", model: str = "scrubber"):
    """Run the local redaction API."""
    import os

    import uvicorn

    os.environ["SCRUBBER_BACKEND"], os.environ["SCRUBBER_MODEL"] = backend, model
    uvicorn.run("scrubber.api:app", host=host, port=port)


if __name__ == "__main__":
    app()

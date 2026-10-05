"""Scoring. Pure functions — no model calls.

Per example we compare gold spans (from the dataset) with the spans the redaction engine actually
replaced, so the metrics measure the real output a user would get.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .data import gold_entities
from .prompt import Entity
from .redact import Redaction


@dataclass
class Tally:
    n_docs: int = 0
    gold_values: int = 0
    leaked_values: int = 0
    gold_chars: int = 0
    covered_gold_chars: int = 0
    redacted_chars: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    pred_entities: int = 0
    hallucinated: int = 0
    parse_failures: int = 0
    seconds: float = 0.0
    per_label_gold: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_label_caught: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def summary(self) -> dict:
        def div(a: float, b: float) -> float:
            return a / b if b else 0.0

        p = div(self.tp, self.tp + self.fp)
        r = div(self.tp, self.tp + self.fn)
        return {
            "docs": self.n_docs,
            "leak_rate": div(self.leaked_values, self.gold_values),
            "char_recall": div(self.covered_gold_chars, self.gold_chars),
            "char_precision": div(self.covered_gold_chars, self.redacted_chars) if self.redacted_chars else 1.0,
            "entity_precision": p,
            "entity_recall": r,
            "entity_f1": div(2 * p * r, p + r),
            "hallucination_rate": div(self.hallucinated, self.pred_entities),
            "parse_failure_rate": div(self.parse_failures, self.n_docs),
            "sec_per_doc": div(self.seconds, self.n_docs),
            "per_label_recall": {k: div(self.per_label_caught[k], v) for k, v in sorted(self.per_label_gold.items())},
        }


def _chars(spans) -> set[int]:
    out: set[int] = set()
    for s in spans:
        start, end = (s["start"], s["end"]) if isinstance(s, dict) else (s.start, s.end)
        out.update(range(start, end))
    return out


def score(record: dict, predicted: list[Entity], redaction: Redaction, parsed: bool, seconds: float,
          t: Tally) -> None:
    t.n_docs += 1
    t.seconds += seconds
    t.parse_failures += 0 if parsed else 1
    t.pred_entities += len(predicted)
    t.hallucinated += len(redaction.rejected)

    # Leak rate: a gold value leaks if ANY of its occurrences is not fully covered by a redaction.
    # (Position-based, so "12" as an age isn't confused with "$12.40" elsewhere in the text.)
    p = _chars(redaction.spans)
    gold = gold_entities(record)
    for value, label in gold:
        occurrences = [s for s in record["spans"] if s["value"] == value and s["label"] == label]
        leaked = any(not set(range(s["start"], s["end"])) <= p for s in occurrences)
        t.gold_values += 1
        t.per_label_gold[label] += 1
        if leaked:
            t.leaked_values += 1
        else:
            t.per_label_caught[label] += 1

    # Character coverage.
    g = _chars(record["spans"])
    t.gold_chars += len(g)
    t.redacted_chars += len(p)
    t.covered_gold_chars += len(g & p)

    # Exact entity match on (text, label).
    gold_set = set(gold)
    pred_set = {(e.text, e.label) for e in predicted}
    t.tp += len(gold_set & pred_set)
    t.fp += len(pred_set - gold_set)
    t.fn += len(gold_set - pred_set)

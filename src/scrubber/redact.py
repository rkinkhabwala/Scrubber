"""Deterministic redaction: entities → verified spans → placeholders (and back).

Pure functions, no I/O, no model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .prompt import Entity


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    value: str


@dataclass
class Redaction:
    text: str
    spans: list[Span]
    mapping: dict[str, str] = field(default_factory=dict)  # placeholder -> original value
    rejected: list[Entity] = field(default_factory=list)  # entities not found in the text


def _on_boundary(text: str, start: int, end: int) -> bool:
    """Reject matches inside a longer token: age '58' must not match within '$1585.85'."""
    if start > 0 and text[start].isalnum() and text[start - 1].isalnum():
        return False
    if end < len(text) and text[end - 1].isalnum() and text[end].isalnum():
        return False
    return True


def _occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    out, i = [], text.find(needle)
    while i != -1:
        out.append((i, i + len(needle)))
        i = text.find(needle, i + 1)
    if not out:
        # Fallback: tolerate whitespace differences ("4111  1111" vs "4111 1111"), exact otherwise.
        pattern = r"\s+".join(re.escape(p) for p in needle.split())
        if pattern:
            out = [(m.start(), m.end()) for m in re.finditer(pattern, text)]
    return [(s, e) for s, e in out if _on_boundary(text, s, e)]


def locate(text: str, entities: list[Entity]) -> tuple[list[Span], list[Entity]]:
    """Find every occurrence of each entity; drop overlaps (longest wins). Returns (spans, rejected)."""
    candidates: list[Span] = []
    rejected: list[Entity] = []
    for e in entities:
        occ = _occurrences(text, e.text)
        if not occ:
            rejected.append(e)
        candidates += [Span(s, t, e.label, text[s:t]) for s, t in occ]
    candidates.sort(key=lambda s: (-(s.end - s.start), s.start))
    taken: list[Span] = []
    for c in candidates:
        if all(c.end <= t.start or c.start >= t.end for t in taken):
            taken.append(c)
    return sorted(taken, key=lambda s: s.start), rejected


def apply(text: str, spans: list[Span]) -> Redaction:
    """Replace spans with [LABEL_n]; identical (label, value) pairs share one number."""
    counters: dict[str, int] = {}
    ids: dict[tuple[str, str], str] = {}
    mapping: dict[str, str] = {}
    out, cursor = [], 0
    for s in sorted(spans, key=lambda s: s.start):
        key = (s.label, s.value)
        if key not in ids:
            counters[s.label] = counters.get(s.label, 0) + 1
            ids[key] = f"[{s.label}_{counters[s.label]}]"
            mapping[ids[key]] = s.value
        out.append(text[cursor:s.start])
        out.append(ids[key])
        cursor = s.end
    out.append(text[cursor:])
    return Redaction(text="".join(out), spans=spans, mapping=mapping)


def redact(text: str, entities: list[Entity]) -> Redaction:
    spans, rejected = locate(text, entities)
    r = apply(text, spans)
    r.rejected = rejected
    return r


def restore(text: str, mapping: dict[str, str]) -> str:
    """Put original values back, e.g. into a cloud LLM's answer about redacted text."""
    for placeholder in sorted(mapping, key=len, reverse=True):
        text = text.replace(placeholder, mapping[placeholder])
    return text

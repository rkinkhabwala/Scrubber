"""The one system prompt used for training AND inference, plus output parsing."""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

from .labels import LABELS

SYSTEM_PROMPT = (
    "You find personal data (PII) in text. Return JSON only: "
    '{"entities": [{"text": <exact substring>, "label": <LABEL>}]}. '
    "Copy each entity exactly as it appears. List each distinct entity once, in order. "
    "Do not flag amounts, company names, product or ticket codes, or generic dates like 'today'. "
    "If there is no PII return {\"entities\": []}.\nLabels:\n"
    + "\n".join(f"- {k}: {v}" for k, v in LABELS.items())
)


class Entity(BaseModel):
    text: str
    label: str


class Prediction(BaseModel):
    entities: list[Entity]


def target_json(entities: list[tuple[str, str]]) -> str:
    """Serialize gold entities exactly as the model is trained to emit them."""
    return json.dumps({"entities": [{"text": t, "label": lab} for t, lab in entities]}, ensure_ascii=False)


_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_output(raw: str) -> tuple[list[Entity], bool]:
    """Parse model output. Returns (entities, ok). Tolerates think tags, code fences and trailing text."""
    text = _THINK.sub("", raw).strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    if start == -1:
        return [], False
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        pred = Prediction.model_validate(obj)
    except (json.JSONDecodeError, ValidationError):
        return [], False
    ents = [Entity(text=e.text, label=e.label.upper()) for e in pred.entities if e.text.strip()]
    ents = [e for e in ents if e.label in LABELS]
    return ents, True

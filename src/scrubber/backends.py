"""Predictors: text → list[Entity]. One interface for baselines and fine-tuned models.

    regex     rule baseline (emails, phones, SSNs, Luhn-valid cards)
    presidio  Microsoft Presidio + spaCy (pip install -e ".[presidio]")
    ollama    any Ollama model, prompted with the same system prompt (base or fine-tuned export)
    mlx       MLX model, optionally with a LoRA adapter (pip install -e ".[mlx]", Apple Silicon)
    oracle    returns the gold entities (sanity check)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .prompt import SYSTEM_PROMPT, Entity, Prediction, parse_output


@dataclass
class Result:
    entities: list[Entity]
    parsed: bool = True


class Backend(Protocol):
    name: str

    def predict(self, text: str) -> Result: ...


# ---------------------------------------------------------------- regex
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE = re.compile(r"(?:\+?1[\s.-]?)?\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b(?:\s?x\d+)?")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn(digits: str) -> bool:
    nums = [int(c) for c in digits][::-1]
    total = sum(n if i % 2 == 0 else (n * 2 - 9 if n * 2 > 9 else n * 2) for i, n in enumerate(nums))
    return total % 10 == 0


class RegexBackend:
    name = "regex"

    def predict(self, text: str) -> Result:
        ents: list[Entity] = []
        for m in _EMAIL.finditer(text):
            ents.append(Entity(text=m.group(), label="EMAIL"))
        for m in _SSN.finditer(text):
            ents.append(Entity(text=m.group(), label="GOV_ID"))
        for m in _CARD.finditer(text):
            digits = re.sub(r"\D", "", m.group())
            if 13 <= len(digits) <= 19 and _luhn(digits):
                ents.append(Entity(text=m.group().strip(" -"), label="CARD"))
        for m in _PHONE.finditer(text):
            if not _SSN.fullmatch(m.group()):
                ents.append(Entity(text=m.group().strip(), label="PHONE"))
        return Result(ents)


# ---------------------------------------------------------------- presidio
PRESIDIO_MAP = {
    "PERSON": "NAME", "EMAIL_ADDRESS": "EMAIL", "PHONE_NUMBER": "PHONE", "LOCATION": "ADDRESS",
    "DATE_TIME": "DATE", "US_SSN": "GOV_ID", "US_PASSPORT": "GOV_ID", "US_DRIVER_LICENSE": "GOV_ID",
    "US_ITIN": "GOV_ID", "CREDIT_CARD": "CARD", "US_BANK_NUMBER": "ACCOUNT", "IBAN_CODE": "ACCOUNT",
}


class PresidioBackend:
    name = "presidio"

    def __init__(self, threshold: float = 0.4):
        from presidio_analyzer import AnalyzerEngine

        self.engine = AnalyzerEngine()
        self.threshold = threshold

    def predict(self, text: str) -> Result:
        found = self.engine.analyze(text=text, language="en", entities=list(PRESIDIO_MAP))
        return Result([Entity(text=text[r.start:r.end], label=PRESIDIO_MAP[r.entity_type])
                       for r in found if r.score >= self.threshold])


# ---------------------------------------------------------------- ollama
class OllamaBackend:
    def __init__(self, model: str, host: str = "http://localhost:11434", constrained: bool = True):
        from ollama import Client

        self.client = Client(host=host)
        self.model = model
        self.name = f"ollama:{model}"
        self.constrained = constrained

    def predict(self, text: str) -> Result:
        resp = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
            format=Prediction.model_json_schema() if self.constrained else None,
            options={"temperature": 0, "num_ctx": 4096},
        )
        ents, ok = parse_output(resp.message.content)
        return Result(ents, ok)


# ---------------------------------------------------------------- mlx
class MLXBackend:
    def __init__(self, model: str, adapter_path: str | None = None, max_tokens: int = 512):
        from mlx_lm import load

        self.model, self.tokenizer = load(model, adapter_path=adapter_path)
        self.max_tokens = max_tokens
        self.name = f"mlx:{model.split('/')[-1]}" + ("+lora" if adapter_path else "")

    def predict(self, text: str) -> Result:
        from mlx_lm import generate

        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=False,
        )
        out = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=self.max_tokens, verbose=False)
        ents, ok = parse_output(out)
        return Result(ents, ok)


# ---------------------------------------------------------------- oracle
class OracleBackend:
    """Looks up gold answers by text. Used to sanity-check the metrics (should score perfectly)."""

    name = "oracle"

    def __init__(self, records: list[dict]):
        from .data import gold_entities

        self.gold = {r["text"]: [Entity(text=t, label=lab) for t, lab in gold_entities(r)] for r in records}

    def predict(self, text: str) -> Result:
        return Result(self.gold.get(text, []))


def make_backend(kind: str, model: str | None = None, adapter: str | None = None,
                 records: list[dict] | None = None) -> Backend:
    if kind == "regex":
        return RegexBackend()
    if kind == "presidio":
        return PresidioBackend()
    if kind == "ollama":
        return OllamaBackend(model or "qwen3:4b-instruct-2507-q4_K_M")
    if kind == "mlx":
        return MLXBackend(model or "mlx-community/Qwen3-4B-Instruct-2507-4bit", adapter)
    if kind == "oracle":
        return OracleBackend(records or [])
    raise ValueError(f"unknown backend {kind}")

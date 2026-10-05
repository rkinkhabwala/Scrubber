"""Local redaction API (binds to 127.0.0.1 by default).

POST /redact  {"text": "..."}                -> {"redacted", "entities", "mapping"}
POST /restore {"text": "...", "mapping": {}}  -> {"text"}

Typical use: redact → send the redacted text to any other tool or LLM → restore its answer.
The mapping contains the original PII; the server keeps nothing, so the caller must protect it.
"""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel

from .backends import make_backend
from .redact import redact, restore

app = FastAPI(title="Scrubber", version="0.1.0")


@lru_cache(maxsize=1)
def backend():
    return make_backend(os.getenv("SCRUBBER_BACKEND", "ollama"), os.getenv("SCRUBBER_MODEL", "scrubber"))


class RedactIn(BaseModel):
    text: str


class RestoreIn(BaseModel):
    text: str
    mapping: dict[str, str]


@app.get("/health")
def health():
    return {"ok": True, "backend": backend().name}


@app.post("/redact")
def redact_endpoint(body: RedactIn):
    res = backend().predict(body.text)
    r = redact(body.text, res.entities)
    return {
        "redacted": r.text,
        "entities": [{"label": s.label, "start": s.start, "end": s.end} for s in r.spans],
        "mapping": r.mapping,
        "model_output_valid": res.parsed,
    }


@app.post("/restore")
def restore_endpoint(body: RestoreIn):
    return {"text": restore(body.text, body.mapping)}

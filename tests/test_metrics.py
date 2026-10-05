from scrubber.backends import OracleBackend, RegexBackend
from scrubber.domain_synth import generate
from scrubber.metrics import Tally, score
from scrubber.prompt import Entity
from scrubber.redact import redact


def run(backend, records):
    t = Tally()
    for rec in records:
        res = backend.predict(rec["text"])
        score(rec, res.entities, redact(rec["text"], res.entities), res.parsed, 0.0, t)
    return t.summary()


def test_oracle_is_perfect():
    recs = generate(64)
    s = run(OracleBackend(recs), recs)
    assert s["leak_rate"] == 0.0
    assert s["char_recall"] == 1.0 and s["char_precision"] == 1.0 and s["entity_f1"] == 1.0


def test_nothing_redacted_leaks_everything():
    recs = [r for r in generate(32) if r["spans"]]

    class Null:
        name = "null"

        def predict(self, text):
            return OracleBackend([]).predict(text)

    s = run(Null(), recs)
    assert s["leak_rate"] == 1.0 and s["char_recall"] == 0.0


def test_leak_is_position_based():
    rec = {"text": "Age 12, paid $12.40", "spans": [{"start": 4, "end": 6, "label": "AGE", "value": "12"}]}
    t = Tally()
    ents = [Entity(text="12", label="AGE")]
    score(rec, ents, redact(rec["text"], ents), True, 0.0, t)
    assert t.summary()["leak_rate"] == 0.0


def test_partial_name_counts_as_leak():
    rec = {"text": "Maria Lopez", "spans": [{"start": 0, "end": 11, "label": "NAME", "value": "Maria Lopez"}]}
    t = Tally()
    ents = [Entity(text="Lopez", label="NAME")]
    score(rec, ents, redact(rec["text"], ents), True, 0.0, t)
    s = t.summary()
    assert s["leak_rate"] == 1.0
    assert 0.4 < s["char_recall"] < 0.5


def test_regex_baseline_catches_structured_pii():
    res = RegexBackend().predict("mail a@b.co, ssn 123-45-6789, card 4111 1111 1111 1111, call (555) 201-8890")
    got = {(e.text, e.label) for e in res.entities}
    assert ("a@b.co", "EMAIL") in got
    assert ("123-45-6789", "GOV_ID") in got
    assert ("4111 1111 1111 1111", "CARD") in got
    assert any(lab == "PHONE" for _, lab in got)

import json
from collections import Counter
from pathlib import Path

from scrubber.data import convert_row, gold_entities, prepare, read_jsonl, to_chat
from scrubber.domain_synth import HELDOUT_TEMPLATES, TEMPLATES, generate
from scrubber.prompt import SYSTEM_PROMPT, parse_output

FIXTURE = Path(__file__).parent / "fixtures" / "ai4privacy_sample.jsonl"
ROWS = [json.loads(line) for line in FIXTURE.open()]


def test_convert_merges_names_and_maps_labels():
    rec = convert_row(ROWS[0])
    assert gold_entities(rec) == [("Maria Lopez", "NAME"), ("maria.lopez@example.com", "EMAIL"),
                                  ("555-201-8890", "PHONE")]


def test_dropped_labels_and_title_not_merged():
    rec = convert_row(ROWS[1])
    assert gold_entities(rec) == [("Chen", "NAME")]  # TIME and TITLE dropped


def test_address_parts_stay_separate_and_ids_map():
    labels = [lab for _, lab in gold_entities(convert_row(ROWS[3]))]
    assert labels == ["ADDRESS", "ADDRESS", "ADDRESS", "ADDRESS", "CARD", "GOV_ID"]


def test_bad_offsets_skipped_unknown_counted():
    assert convert_row(ROWS[4]) is None
    unknown = Counter()
    assert gold_entities(convert_row(ROWS[5], unknown)) == []
    assert unknown == {"USERNAME": 1}


def test_domain_spans_are_exact():
    for rec in generate(200) + generate(40, templates=HELDOUT_TEMPLATES):
        for s in rec["spans"]:
            assert rec["text"][s["start"]:s["end"]] == s["value"]


def test_every_template_renders():
    assert len(generate(len(TEMPLATES))) == len(TEMPLATES)


def test_chat_record_round_trips_through_parser():
    chat = to_chat(convert_row(ROWS[0]))
    roles = [m["role"] for m in chat["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert chat["messages"][0]["content"] == SYSTEM_PROMPT
    ents, ok = parse_output(chat["messages"][2]["content"])
    assert ok and [(e.text, e.label) for e in ents] == gold_entities(convert_row(ROWS[0]))


def test_prepare_from_local_file(tmp_path):
    counts = prepare(tmp_path, n_train=1, n_valid=1, n_test=1, domain_train=20, domain_valid=4,
                     domain_test=10, train_file=str(FIXTURE))
    assert counts["raw/train"] == 1 + 20
    assert counts["raw/test"] == 1
    assert counts["raw/test_domain"] == 10
    train_texts = {r["text"] for r in read_jsonl(tmp_path / "raw/train.jsonl")}
    test_texts = {r["text"] for r in read_jsonl(tmp_path / "raw/test.jsonl")}
    assert not train_texts & test_texts
    sources = Counter(r["source"] for r in read_jsonl(tmp_path / "raw/test_domain.jsonl"))
    assert sources == {"domain": 5, "domain_heldout": 5}
    first = json.loads((tmp_path / "mlx/train.jsonl").open().readline())
    assert set(first) == {"messages"}

from scrubber.prompt import Entity, parse_output
from scrubber.redact import redact, restore


def E(text, label):
    return Entity(text=text, label=label)


def test_basic_redaction_and_restore():
    text = "Maria Lopez emailed maria@x.com. Maria Lopez again."
    r = redact(text, [E("Maria Lopez", "NAME"), E("maria@x.com", "EMAIL")])
    assert r.text == "[NAME_1] emailed [EMAIL_1]. [NAME_1] again."
    assert restore(r.text, r.mapping) == text


def test_distinct_values_get_distinct_numbers():
    r = redact("Ann met Bob.", [E("Ann", "NAME"), E("Bob", "NAME")])
    assert r.text == "[NAME_1] met [NAME_2]."


def test_hallucinated_entity_is_rejected_not_applied():
    r = redact("Call 555-1234.", [E("555-9999", "PHONE")])
    assert r.text == "Call 555-1234."
    assert [e.text for e in r.rejected] == ["555-9999"]


def test_overlap_longest_wins():
    r = redact("Card 4111 1111 1111 1111", [E("4111", "ACCOUNT"), E("4111 1111 1111 1111", "CARD")])
    assert r.text == "Card [CARD_1]"


def test_whitespace_tolerant_match():
    r = redact("Card 4111  1111 1111 1111", [E("4111 1111 1111 1111", "CARD")])
    assert r.text == "Card [CARD_1]"


def test_no_match_inside_longer_token():
    r = redact("Age 58. Copay $1585.85, ref TXN5812.", [E("58", "AGE")])
    assert r.text == "Age [AGE_1]. Copay $1585.85, ref TXN5812."


def test_restore_handles_placeholder_prefixes():
    mapping = {"[NAME_1]": "Ann", "[NAME_10]": "Zed"}
    assert restore("[NAME_10] and [NAME_1]", mapping) == "Zed and Ann"


def test_parse_output_tolerates_noise():
    raw = '<think>hmm</think>\n```json\n{"entities": [{"text": "Ann", "label": "name"}]}\n``` trailing'
    ents, ok = parse_output(raw)
    assert ok and ents == [E("Ann", "NAME")]


def test_parse_output_drops_unknown_labels_and_flags_garbage():
    ents, ok = parse_output('{"entities": [{"text": "x", "label": "COLOR"}]}')
    assert ok and ents == []
    assert parse_output("sorry, I can't") == ([], False)

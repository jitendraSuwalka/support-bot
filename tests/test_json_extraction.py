"""Pulling JSON out of model output that is never quite bare JSON."""

import pytest

import agent


@pytest.mark.parametrize(
    ("label", "raw", "should_parse"),
    [
        ("clean object", '{"a": 1}', True),
        ("trailing prose (the crash case)", '{"a": 1}\nAll good.', True),
        ("code fence", '```json\n{"a": 1}\n```', True),
        ("prose both sides", 'Verdict: {"a": 1} hope that helps', True),
        ("braces inside a string", '{"a": "has {braces}"}', True),
        ("array", '["x","y"]', True),
        ("no JSON present", "I cannot do that.", False),
    ],
)
def test_extract_json(label, raw, should_parse):
    got = agent.extract_json(raw)
    assert (got is not None) == should_parse, f"{label}: {got!r}"

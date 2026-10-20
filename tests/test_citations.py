"""The deterministic citation guard: every URL and number must be evidenced."""

import agent

EVIDENCE = (
    "There are 21 offices across the UK. Call 0800 023 4567. "
    "Source: https://www.evelyn.com/offices/bristol/"
)


def test_truncated_url_is_caught():
    # The real bug from testing: a parent path derived from a deeper URL.
    violations = agent.check_citations("See evelyn.com/offices", EVIDENCE)
    assert any("shorten" in v for v in violations), violations


def test_invented_url_is_caught():
    assert agent.check_citations("Visit example.com/help", EVIDENCE) != []


def test_invented_phone_number_is_caught():
    assert agent.check_citations("Call 0800 111 2222", EVIDENCE) != []


def test_exact_url_passes():
    assert agent.check_citations("See https://www.evelyn.com/offices/bristol/", EVIDENCE) == []


def test_correct_numbers_pass():
    assert agent.check_citations("There are 21 offices. Call 0800 023 4567.", EVIDENCE) == []


def test_small_numbers_do_not_false_alarm():
    assert agent.check_citations("There are 3 options.", EVIDENCE) == []

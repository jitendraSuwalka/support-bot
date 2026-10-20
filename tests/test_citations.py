"""The deterministic citation guard: every URL and number must be evidenced."""

import agent

EVIDENCE = (
    "There are 21 offices across the UK. Call 0800 023 4567. "
    "Source: https://www.fernhillwealth.example/offices/bristol/"
)


def test_truncated_url_is_caught():
    # The real bug from testing: a parent path derived from a deeper URL.
    violations = agent.check_citations("See fernhillwealth.example/offices", EVIDENCE)
    assert any("shorten" in v for v in violations), violations


def test_invented_url_is_caught():
    assert agent.check_citations("Visit example.com/help", EVIDENCE) != []


def test_invented_phone_number_is_caught():
    assert agent.check_citations("Call 0800 111 2222", EVIDENCE) != []


def test_exact_url_passes():
    answer = "See https://www.fernhillwealth.example/offices/bristol/"
    assert agent.check_citations(answer, EVIDENCE) == []


def test_correct_numbers_pass():
    assert agent.check_citations("There are 21 offices. Call 0800 023 4567.", EVIDENCE) == []


def test_small_numbers_do_not_false_alarm():
    assert agent.check_citations("There are 3 options.", EVIDENCE) == []


def test_number_assembled_from_two_evidence_numbers_is_caught():
    # The earlier implementation joined every evidence digit into one string
    # ("10011998") and accepted any substring, so "0119 98" slipped through.
    evidence = "Reference FW-1001. The firm was founded in 1998."
    assert agent.check_citations("Call 0119 98 to confirm.", evidence) != []


def test_number_hidden_inside_a_longer_evidence_number_is_caught():
    evidence = "Call 0117 496 0100 for the switchboard."
    assert agent.check_citations("Call 4960 for the switchboard.", evidence) != []


def test_number_tokens_are_digit_only_and_deduplicated():
    assert agent.number_tokens("Call 0117 496 0100 or 0117-496-0100 by 31 December 2025") == {
        "01174960100",
        "31",
        "2025",
    }


def test_differently_punctuated_phone_number_still_passes():
    evidence = "Call 0117 496 0100."
    assert agent.check_citations("Ring 0117-496-0100 any weekday.", evidence) == []

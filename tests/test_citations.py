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


# --- numbers next to each other in the evidence -----------------------------------
# The baseline evaluation (reports/live-baseline.json: portal-02, portal-07)
# found the guard rejecting a helpline number copied exactly, because the
# evidence wrote the number and the opening hours in one breath.


def test_a_number_followed_by_a_comma_and_another_number_is_still_one_number():
    evidence = "call the portal access helpline on 0117 496 0233, 09:00 to 17:30 on working days."
    assert agent.number_tokens(evidence) == {"01174960233", "0900", "1730"}
    assert agent.check_citations("Call the helpline on 0117 496 0233.", evidence) == []
    assert agent.check_citations("Open 09:00 to 17:30.", evidence) == []


def test_everyday_figures_keep_their_own_shape():
    text = "£20,000 in 2026/27, from 6 April 2026; 1.5% a year; £1,000,000; FW-1001; M2 3ZX"
    expected = {"20000", "202627", "6", "2026", "15", "1000000", "1001", "2", "3"}
    assert agent.number_tokens(text) == expected


def test_a_phone_number_is_one_token_but_a_short_group_after_it_is_not():
    assert agent.number_tokens("Telephone 0161 496 0170. Nearest tram stop") == {"01614960170"}
    assert agent.number_tokens("0117 496 0100 9 to 5") == {"01174960100", "9", "5"}


def test_a_number_assembled_across_a_comma_is_caught():
    evidence = "call 0117 496 0233, 09:00 to 17:30"
    assert agent.check_citations("Call 0117 496 0233 0900.", evidence) != []


# --- the customer's own numbers ----------------------------------------------------
# Step 4.3's caching runs, isa-07: asked "I'm 70. Does the new cash ISA cap
# apply to me?", the answer said "You told me you are 70" and the guard
# flagged 70 as not in the evidence, twice, in both runs.


def test_a_number_the_customer_wrote_may_be_repeated_back():
    evidence = "The £12,000 cap applies to investors under the age of 65."
    answer = "You told me you are 70, so the under-65 cap does not apply."
    assert agent.check_citations(answer, evidence) != []  # without the customer's words
    assert agent.check_citations(answer, evidence, customer_text="I'm 70. Does it apply?") == []


def test_a_url_the_customer_wrote_is_still_not_repeated():
    customer = "Is fernhill-wealth-login.example/verify really you?"
    answer = "Yes, fernhill-wealth-login.example/verify is ours."
    assert agent.check_citations(answer, "Our site is fernhillwealth.example", customer) != []


def test_a_phone_number_the_customer_gives_is_still_not_evidence():
    """Repeating it would lend the firm's authority to a number a fraudster
    may have given them."""
    evidence = "Call the switchboard on 0117 496 0100."
    asked = "Is 0117 496 9999 your fraud line?"
    answer = "Yes, our fraud line is 0117 496 9999."
    assert agent.check_citations(answer, evidence, customer_text=asked) == [
        "Number '0117 496 9999' does not appear in the evidence."
    ]
    international = "They said +44 117 496 9999"
    assert "441174969999" not in agent.customer_numbers(international)


def test_ages_amounts_and_years_the_customer_gives_may_be_repeated():
    said = "I'm 70, I have £1,000,000 saved, and I opened it in 2019."
    assert agent.customer_numbers(said) == {"70", "1000000", "2019"}

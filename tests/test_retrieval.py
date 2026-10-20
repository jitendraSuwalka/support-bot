"""Lexical retrieval: ranking, stopwords, stemming and the relevance floor."""

import agent


def top(query, extra=""):
    tokens = agent.tokenize(query) | agent.tokenize(extra)
    scored = [r for r in agent.score_documents(tokens) if r[0] >= agent.RELEVANCE_FLOOR]
    return [entry["topic"] for _, _, entry in scored[:2]]


def test_relevant_query_ranks_the_right_entry_first():
    assert top("where is your bristol office")[:1] == ["office locations bristol local offices"]


def test_complaint_query_beats_the_long_financial_planning_entry():
    assert top("how long does a complaint take")[:1] == ["how to make a complaint"]


def test_off_topic_query_returns_nothing():
    assert top("what is the weather in Paris") == []


def test_expansion_terms_rescue_a_reworded_query():
    assert any("complaint" in t for t in top("I am unhappy", "complaint resolution ombudsman"))


def test_stopwords_drop_generic_query_words():
    assert "long" not in agent.tokenize("how long does it take")


def test_plurals_fold_together():
    assert agent.tokenize("offices") == agent.tokenize("office")


def test_good_matches_clear_the_floor_with_margin_not_by_luck():
    # A healthy config leaves room between real matches and the floor. Testing
    # once found the complaint query at 0.91 against a floor of 0.90 -- passing
    # by 0.01, which is luck, not correctness.
    scored = agent.score_documents(agent.tokenize("how do I make a complaint"))
    best = scored[0][0] if scored else 0
    assert best >= agent.RELEVANCE_FLOOR + 0.3, f"best {best:.2f} vs floor {agent.RELEVANCE_FLOOR}"

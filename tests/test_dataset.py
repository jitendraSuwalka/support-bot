"""The labelled question set loads, is internally consistent, and every label
is checked against the corpus so a wrong label fails the build."""

from collections import Counter
from pathlib import Path

import pytest
import yaml

import agent
from eval.dataset import EXPECTED, KINDS, DatasetError, load_cases, parse_case, summarise
from rag.corpus import load_corpus
from tests.conftest import make_chunk

# --- the real set -----------------------------------------------------------------


@pytest.fixture(scope="module")
def cases():
    return load_cases()


def test_the_set_is_large_and_every_case_resolved(cases):
    assert len(cases) >= 100
    for case in cases:
        if case.gold:
            assert case.gold_chunk_ids, case.id


def test_every_kind_and_every_expected_behaviour_is_represented(cases):
    assert set(KINDS) == {case.kind for case in cases}
    assert set(EXPECTED) == {case.expected for case in cases}


def test_every_document_has_questions(cases):
    covered = Counter(ref.split("/", 1)[0] for case in cases for ref in case.gold)
    for document in load_corpus():
        assert covered[document.doc_id] >= 3, document.doc_id


def test_behaviours_come_with_the_right_tools(cases):
    for case in cases:
        if case.expected == "greeting":
            assert case.tools == (), case.id
        if case.kind in ("advice", "handoff"):
            assert "escalate_to_adviser" in case.tools, case.id
        if case.kind == "status" and case.expected == "answer":
            assert case.tools == ("look_up_client_query",), case.id
        if case.expected == "answer" and case.kind != "status":
            assert "search_knowledge_base" in case.tools, case.id
        if case.expected == "escalate" and case.gold:
            assert "search_knowledge_base" in case.tools, case.id  # signposting needs a lookup
        if case.expected == "clarify":
            assert "ask_for_clarification" in case.tools, case.id
        if case.expected == "decline":
            assert case.tools == ("decline_out_of_scope",), case.id  # nothing else is run


def test_conversations_with_earlier_turns_are_labelled_on_the_final_turn(cases):
    multi = [case for case in cases if case.before]
    assert multi, "the clarify-once policy needs at least one two-turn case"
    for case in multi:
        assert all(message.strip() for message in case.before), case.id


def test_status_cases_quote_the_live_data_exactly(cases):
    statuses = set(agent.CLIENT_QUERIES.values())
    for case in cases:
        if case.kind == "status" and case.expected == "answer":
            assert case.must_contain and set(case.must_contain) <= statuses, case.id


def test_the_summary_counts_add_up(cases):
    summary = summarise(cases)
    assert summary["cases"] == len(cases)
    assert sum(summary["by_kind"].values()) == len(cases)
    assert sum(summary["by_expected"].values()) == len(cases)
    assert len(summary["documents_with_gold"]) == 18


# --- validation -------------------------------------------------------------------


CHUNKS = [
    make_chunk("isa-guide", "The allowance", "Adults can pay up to £20,000 into ISAs."),
    make_chunk("contact", "Manchester", "41 Mosley Street, Manchester, M2 3ZX."),
]
BY_SECTION = {"isa-guide/The allowance": [CHUNKS[0]], "contact/Manchester": [CHUNKS[1]]}


def good(**overrides):
    raw = {
        "id": "isa-x",
        "question": "How much can I pay in?",
        "kind": "factual",
        "expected": "answer",
        "gold": ["isa-guide/The allowance"],
        "must_contain": ["£20,000"],
        "tools": ["search_knowledge_base"],
    }
    raw.update(overrides)
    return raw


def test_a_good_case_resolves_to_its_chunk_ids():
    case = parse_case(good(), BY_SECTION)
    assert case.gold_chunk_ids == {CHUNKS[0].chunk_id}
    assert case.as_dict()["gold_chunk_ids"] == [CHUNKS[0].chunk_id]
    assert case.before == ()


def test_an_escalate_case_may_signpost_gold_sections():
    case = parse_case(
        good(expected="escalate", tools=["search_knowledge_base", "escalate_to_adviser"]),
        BY_SECTION,
    )
    assert case.gold_chunk_ids and case.expected == "escalate"


def test_earlier_turns_are_kept_in_order():
    case = parse_case(good(before=["quiz", "the quiz"]), BY_SECTION)
    assert case.before == ("quiz", "the quiz")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"gold": ["isa-guide/Allowances"]}, "gold section not in the corpus"),
        ({"must_contain": ["£25,000"]}, "is not in the gold text"),
        ({"kind": "trivia"}, "kind must be one of"),
        ({"expected": "guess"}, "expected must be one of"),
        ({"tools": ["search_web"]}, "tools must be from"),
        ({"gold": []}, "needs at least one gold section"),
        ({"expected": "clarify", "gold": ["isa-guide/The allowance"]}, "only answer and escalate"),
        ({"expected": "decline", "gold": [], "must_contain": ["£20,000"]}, "needs gold sections"),
        ({"question": "   "}, "empty question"),
        ({"extra": 1}, "fields must be"),
        ({"before": ["", "x"]}, "before must be a list"),
    ],
)
def test_bad_labels_are_rejected_with_the_case_id(overrides, message):
    with pytest.raises(DatasetError, match=f"isa-x: .*{message}"):
        parse_case(good(**overrides), BY_SECTION)


def test_duplicate_ids_are_rejected(tmp_path: Path):
    path = tmp_path / "q.yaml"
    path.write_text(yaml.safe_dump([good(), good()]), encoding="utf-8")
    with pytest.raises(DatasetError, match="duplicate ids: isa-x"):
        load_cases(path, CHUNKS)


def test_an_empty_file_is_rejected(tmp_path: Path):
    path = tmp_path / "q.yaml"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(DatasetError, match="non-empty list"):
        load_cases(path, CHUNKS)


def test_the_cli_prints_the_summary(capsys):
    from eval.dataset import main

    assert main() == 0
    assert '"cases": ' in capsys.readouterr().out

"""BM25 over chunks: tokenizing, scoring normalised to the query's ceiling,
exact-string matching, and deterministic order."""

import pytest

from rag.corpus import chunk_corpus
from rag.lexical import BM25Index, LexicalHit, singular, tokenize
from tests.conftest import make_chunk

# --- tokenize -----------------------------------------------------------------------


def test_a_full_stop_is_not_part_of_the_word_before_it():
    # The agent's earlier tokenizer kept it, so "ISAs." and "ISAs" were different terms.
    assert tokenize("ISAs. ISAs") == ["isa", "isa"]


def test_dots_inside_a_token_survive_for_emails_and_domains():
    assert tokenize("write to hello@fernhillwealth.example.") == [
        "write",
        "hello@fernhillwealth.example",
    ]


def test_digits_are_tokens_so_numbers_can_be_matched_exactly():
    assert tokenize("Telephone 0161 496 0170 about £20,000.") == [
        "telephone",
        "0161",
        "496",
        "0170",
        "20",
        "000",
    ]


def test_stopwords_go_and_plurals_fold_but_ss_words_keep_their_s():
    assert tokenize("how do I address the offices") == ["address", "office"]


@pytest.mark.parametrize(
    ("plural", "expected"),
    [
        ("offices", "office"),
        ("isas", "isa"),
        ("fees", "fee"),
        ("branches", "branch"),  # was "branche" in Phase 1
        ("addresses", "address"),  # was "addresse"
        ("taxes", "tax"),
        ("bonus", "bonus"),  # was "bonu"
        ("analysis", "analysis"),
        ("class", "class"),
        ("bus", "bus"),
        ("yes", "yes"),  # too short to touch
    ],
)
def test_plural_folding_rules(plural, expected):
    assert singular(plural) == expected


# --- the index ------------------------------------------------------------------------


@pytest.fixture
def chunks():
    return [
        make_chunk("contact", "Manchester", "41 Mosley Street, Manchester. Call 0161 496 0170.", 0),
        make_chunk("contact", "Bristol", "Fernhill House, Bristol. Call 0117 496 0100.", 1),
        make_chunk("isa-guide", "Allowance", "The ISA allowance is £20,000 a tax year.", 0),
        make_chunk(
            "isa-guide",
            "Transfers",
            "An ISA transfer takes 15 working days for cash and 30 days for other ISA types. "
            "ISA ISA ISA transfer transfer.",
            1,
        ),
    ]


@pytest.fixture
def index(chunks):
    return BM25Index(chunks)


def test_the_index_counts_its_chunks_and_knows_their_terms(index):
    assert len(index) == 4
    assert {"manchester", "isa", "0170"} <= index.idf.keys()
    assert "the" not in index.idf  # stopword


def test_a_phone_number_finds_the_one_chunk_that_contains_it(index):
    hits = index.search(tokenize("0161 496 0170"))
    assert hits[0].chunk.section == "Manchester"
    assert hits[0].matched == frozenset({"0161", "496", "0170"})
    # "496" is in both offices' numbers; only Manchester has all three terms.
    assert [hit.chunk.section for hit in hits] == ["Manchester", "Bristol"]
    assert hits[1].matched == frozenset({"496"})


def test_scores_are_shares_of_the_ceiling_between_zero_and_one(index):
    hits = index.search({"isa", "allowance", "transfer", "manchester"})
    assert all(isinstance(hit, LexicalHit) and 0.0 < hit.score <= 1.0 for hit in hits)
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_unknown_query_terms_do_not_lower_a_good_match(index):
    known = index.search({"manchester", "mosley"})[0].score
    with_noise = index.search({"manchester", "mosley", "premises", "zzzunknown"})[0].score
    assert with_noise == known


def test_a_query_of_only_unknown_terms_matches_nothing(index):
    assert index.search({"zzz", "qqq"}) == []
    assert index.search([]) == []


def test_rarer_terms_weigh_more(index):
    # "isa" is in two chunks, "manchester" in one: the rarer term carries more idf.
    assert index.idf["manchester"] > index.idf["isa"]


def test_term_frequency_counts_with_saturation(index):
    hits = {hit.chunk.section: hit.score for hit in index.search({"isa"})}
    assert hits["Transfers"] > hits["Allowance"]  # "isa" occurs more often there
    assert hits["Transfers"] < 1.0  # but repetition cannot reach the ceiling


def test_k_limits_the_hits_and_ties_break_on_chunk_id(chunks):
    twins = [
        make_chunk("a", "S", "same words here", 0),
        make_chunk("b", "S", "same words here", 0),
    ]
    index = BM25Index(twins)
    hits = index.search({"same", "word"})
    assert [hit.chunk.doc_id for hit in hits] == sorted(hit.chunk.doc_id for hit in hits)
    assert len(index.search({"same"}, k=1)) == 1


def test_the_real_corpus_indexes_and_finds_a_phone_number():
    index = BM25Index(chunk_corpus())
    hits = index.search(tokenize("0161 496 0170"), k=1)
    assert hits[0].chunk.section == "Our offices › Manchester"

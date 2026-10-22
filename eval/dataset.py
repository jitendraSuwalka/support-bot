"""
The labelled question set: load it, validate it against the corpus, resolve it.

A label that is wrong is worse than no label, so loading checks every case:
ids are unique, the kind and expected behaviour are from the known lists, every
gold section exists in the corpus (so a renamed heading breaks the build, not
the metrics), and every string a correct answer must copy appears verbatim in
one of the gold sections (so the set never demands a figure the evidence does
not contain). Gold sections are named by document and heading rather than by
chunk id, because chunk ids are content hashes and change when a sentence is
edited; the heading is what a human labels.

Run `python -m eval.dataset` for a summary.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from rag.corpus import Chunk, chunk_corpus

DEFAULT_PATH = Path(__file__).resolve().parent / "questions.yaml"

KINDS = (
    "factual",  # answerable from the documents
    "exact-string",  # a number, postcode, address or email identifies the answer
    "multi-part",  # two facts, possibly from two documents
    "denial",  # the documents answer it with a "no"
    "status",  # live case data, quoted verbatim
    "advice",  # must reach a human
    "out-of-scope",  # nothing in the documents answers it
    "ambiguous",  # too little to act on: clarify once
    "greeting",  # no tools
)
EXPECTED = ("answer", "escalate", "clarify", "greeting")
TOOLS = ("search_knowledge_base", "look_up_client_query", "escalate_to_adviser")
FIELDS = {"id", "question", "kind", "expected", "gold", "must_contain", "tools"}


class DatasetError(ValueError):
    """A label is malformed or contradicts the corpus. Raised at load time."""


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    kind: str
    expected: str
    gold: tuple[str, ...]  # "doc-id/Section name"
    must_contain: tuple[str, ...]
    tools: tuple[str, ...]
    gold_chunk_ids: frozenset[str]  # resolved against the corpus

    def as_dict(self) -> dict:
        data = asdict(self)
        data["gold_chunk_ids"] = sorted(self.gold_chunk_ids)
        return data


def section_key(chunk: Chunk) -> str:
    return f"{chunk.doc_id}/{chunk.section}"


def _check(condition: bool, case_id: str, message: str) -> None:
    if not condition:
        raise DatasetError(f"{case_id}: {message}")


def parse_case(raw: dict, by_section: dict[str, list[Chunk]]) -> Case:
    case_id = str(raw.get("id", "<no id>"))
    _check(
        isinstance(raw, dict) and set(raw) == FIELDS, case_id, f"fields must be {sorted(FIELDS)}"
    )
    _check(isinstance(raw["question"], str) and raw["question"].strip(), case_id, "empty question")
    _check(raw["kind"] in KINDS, case_id, f"kind must be one of {KINDS}")
    _check(raw["expected"] in EXPECTED, case_id, f"expected must be one of {EXPECTED}")
    for name in ("gold", "must_contain", "tools"):
        _check(isinstance(raw[name], list), case_id, f"{name} must be a list")
    _check(set(raw["tools"]) <= set(TOOLS), case_id, f"tools must be from {TOOLS}")

    gold = tuple(raw["gold"])
    if raw["expected"] == "answer" and raw["kind"] != "status":
        _check(bool(gold), case_id, "an answerable question needs at least one gold section")
    else:
        _check(not gold, case_id, "only answerable questions have gold sections")

    chunk_ids: set[str] = set()
    gold_text = ""
    for ref in gold:
        _check(ref in by_section, case_id, f"gold section not in the corpus: {ref!r}")
        chunk_ids.update(chunk.chunk_id for chunk in by_section[ref])
        gold_text += " ".join(chunk.text for chunk in by_section[ref]) + " "

    must_contain = tuple(raw["must_contain"])
    if gold:
        for needle in must_contain:
            _check(needle in gold_text, case_id, f"must_contain {needle!r} is not in the gold text")

    return Case(
        id=case_id,
        question=raw["question"].strip(),
        kind=raw["kind"],
        expected=raw["expected"],
        gold=gold,
        must_contain=must_contain,
        tools=tuple(raw["tools"]),
        gold_chunk_ids=frozenset(chunk_ids),
    )


def load_cases(
    path: Path | str = DEFAULT_PATH, chunks: Iterable[Chunk] | None = None
) -> list[Case]:
    """Every case in the file, validated and resolved against the corpus."""
    chunks = list(chunk_corpus() if chunks is None else chunks)
    by_section: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_section.setdefault(section_key(chunk), []).append(chunk)

    with open(path, encoding="utf-8") as handle:
        raw_cases = yaml.safe_load(handle)
    if not isinstance(raw_cases, list) or not raw_cases:
        raise DatasetError(f"{path}: expected a non-empty list of cases")

    cases = [parse_case(raw, by_section) for raw in raw_cases]
    ids = Counter(case.id for case in cases)
    duplicates = sorted(name for name, count in ids.items() if count > 1)
    if duplicates:
        raise DatasetError(f"duplicate ids: {', '.join(duplicates)}")
    return cases


def summarise(cases: list[Case]) -> dict:
    documents = Counter(ref.split("/", 1)[0] for case in cases for ref in case.gold)
    return {
        "cases": len(cases),
        "by_kind": dict(sorted(Counter(case.kind for case in cases).items())),
        "by_expected": dict(sorted(Counter(case.expected for case in cases).items())),
        "documents_with_gold": dict(sorted(documents.items())),
        "gold_sections": len({ref for case in cases for ref in case.gold}),
    }


def main() -> int:
    print(json.dumps(summarise(load_cases()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

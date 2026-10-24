"""
Judge the fact-checker by hand: was each rewrite it forced actually needed?

    python -m eval.label_factcheck --by "Your Name"     # label a sample, resumable
    python -m eval.label_factcheck --summary            # the precision so far

The fact-checker is a second model that reads every answer against its
evidence and forces a rewrite when it finds a claim the evidence does not
support. It fires on about a fifth of answers; this tool measures how often
it was right to. The evaluation runs keep, for every forced rewrite, the
claims it flagged, the draft it read and the evidence it read it against
(eval.live). Here a sample of those is shown one at a time, and a person
decides:

- needed: the draft really did say something the evidence does not support;
- not needed: everything in the draft was in the evidence (a false alarm);
- unsure: the evidence is ambiguous.

Precision is needed / (needed + not needed), with a 95% Wilson interval. The
sample is drawn the same way every time (by a hash of the case and draft),
so it can be labelled over several sittings: every answer is saved to
reports/factcheck-labels.json as it is given, and a rerun carries on from
the first unlabelled item. The README's table is rendered from that file.

The file records the runs its sample was drawn from. Once those runs are
replaced by new ones, it still describes the fact-checker as it was then, and
labelling refuses to add the new runs' rewrites to it: a new sample goes in a
new file (--labels reports/factcheck-labels-new.json).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
import time
from collections.abc import Callable
from pathlib import Path

from eval.dataset import load_cases

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
LABELS = REPORTS / "factcheck-labels.json"
VERDICTS = {"y": "needed", "n": "not needed", "u": "unsure"}
SAMPLE = 40


def _run_paths(directory: Path) -> list[Path]:
    paths = [directory / "live.json", *sorted((directory / "repeats").glob("*.json"))]
    return [path for path in paths if path.exists()]


def runs(directory: Path = REPORTS) -> dict[str, str | None]:
    """The runs the rewrites are drawn from: each file, and when it was generated."""
    return {
        path.relative_to(directory).as_posix(): json.loads(path.read_text(encoding="utf-8")).get(
            "generated"
        )
        for path in _run_paths(directory)
    }


def rewrites(directory: Path = REPORTS) -> list[dict]:
    """Every forced rewrite in the current run and its repeats, once per case and draft."""
    seen, found = set(), []
    for path in _run_paths(directory):
        for row in json.loads(path.read_text(encoding="utf-8")).get("rows", []):
            record = row.get("factcheck")
            if not record:
                continue
            key = (row["id"], record["before"])
            if key in seen:
                continue
            seen.add(key)
            found.append({"id": row["id"], "run": path.name, **record})
    return found


def sample(items: list[dict], size: int = SAMPLE) -> list[dict]:
    """A fixed sample: ordered by a hash of case and draft, so it never changes."""

    def key(item: dict) -> str:
        return hashlib.sha256(f"{item['id']}\n{item['before']}".encode()).hexdigest()

    return sorted(items, key=key)[:size]


def load_labels(path: Path = LABELS) -> dict:
    """The labels so far. `drawn_from` names the runs the sample came from and
    `population` how many rewrites they held, so a file always says which
    fact-checker it describes, after those runs are replaced."""
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {
        "labelled_by": saved.get("labelled_by"),
        "started": saved.get("started"),
        "drawn_from": saved.get("drawn_from"),
        "population": saved.get("population"),
        "items": saved.get("items", []),
    }


def save_labels(labels: dict, path: Path = LABELS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8", newline="\n")


def precision(labels: dict) -> dict:
    """Counts per verdict and the precision of the rewrites judged either way."""
    counts = {verdict: 0 for verdict in VERDICTS.values()}
    for item in labels["items"]:
        counts[item["verdict"]] += 1
    judged = counts["needed"] + counts["not needed"]
    return {
        **counts,
        "labelled": len(labels["items"]),
        "precision": round(counts["needed"] / judged, 3) if judged else None,
        "not_needed_ids": [i["id"] for i in labels["items"] if i["verdict"] == "not needed"],
    }


def _block(title: str, text: str) -> str:
    body = "\n".join(
        textwrap.fill(line, 96, subsequent_indent="  ") if line.strip() else ""
        for line in text.splitlines()
    )
    return f"--- {title} ---\n{body}\n"


def show(item: dict, question: str, position: str) -> str:
    claims = "\n".join(f"- {claim}" for claim in item["claims"])
    return "\n".join(
        [
            "=" * 96,
            f"{position}  case {item['id']}  ({item['run']})",
            _block("CUSTOMER", question),
            _block("WHAT THE FACT-CHECKER FLAGGED", claims),
            _block("THE DRAFT IT READ", item["before"]),
            _block("THE EVIDENCE", item["evidence"]),
            _block("WHAT THE CUSTOMER GOT AFTER THE REWRITE", item["after"]),
        ]
    )


PROMPT = (
    "Did the DRAFT say something the EVIDENCE does not support?\n"
    "  [y] yes, the rewrite was needed   [n] no, a false alarm   [u] unsure   "
    "[s] skip for now   [q] quit\n> "
)


def label(
    directory: Path = REPORTS,
    path: Path = LABELS,
    by: str | None = None,
    size: int = SAMPLE,
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
) -> dict:
    """Show each unlabelled item of the sample and record the verdict as it is given."""
    labels = load_labels(path)
    drawn = runs(directory)
    if labels["items"] and labels["drawn_from"] != drawn:
        say(
            f"{path.name} holds labels drawn from other runs than the reports now hold. "
            "Label these runs into a new file with --labels, so that two samples are "
            "never mixed."
        )
        return labels
    if by:
        labels["labelled_by"] = by
    labels["started"] = labels["started"] or time.strftime("%Y-%m-%d")
    questions = {case.id: case.question for case in load_cases()}
    found = rewrites(directory)
    labels["drawn_from"], labels["population"] = drawn, len(found)
    items = sample(found, size)
    done = {(item["id"], item["before"]) for item in labels["items"]}
    todo = [item for item in items if (item["id"], item["before"]) not in done]
    say(f"{len(items)} rewrites in the sample, {len(items) - len(todo)} already labelled.")
    for number, item in enumerate(todo, start=len(items) - len(todo) + 1):
        say(
            show(
                item,
                questions.get(item["id"], "(not in the question set)"),
                f"[{number}/{len(items)}]",
            )
        )
        while True:
            # One letter exactly: a note pasted at this prompt once began with
            # "s" and was taken as skip.
            answer = ask(PROMPT).strip().lower()
            if answer in VERDICTS or answer in ("s", "q"):
                break
            say("Please type one letter: y, n, u, s or q.")
        if answer == "q":
            break
        if answer == "s":
            continue
        note = ask("A short note (or press Enter): ").strip() if answer in ("n", "u") else ""
        labels["items"].append(
            {
                "id": item["id"],
                "run": item["run"],
                "claims": item["claims"],
                "before": item["before"],
                "after": item["after"],
                "verdict": VERDICTS[answer],
                "note": note,
            }
        )
        save_labels(labels, path)  # every answer is kept, whatever happens next
    save_labels(labels, path)
    say(render(precision(labels)))
    return labels


def render(result: dict) -> str:
    share = "n/a" if result["precision"] is None else f"{100 * result['precision']:.1f}%"
    return (
        f"labelled {result['labelled']}: needed {result['needed']}, "
        f"not needed {result['not needed']}, unsure {result['unsure']}; precision {share}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.label_factcheck",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--by", help="who is labelling, recorded in the labels file")
    parser.add_argument("--size", type=int, default=SAMPLE, help="how many rewrites to sample")
    parser.add_argument("--summary", action="store_true", help="print the precision and stop")
    parser.add_argument(
        "--labels", type=Path, default=LABELS, help="the labels file (default: %(default)s)"
    )
    args = parser.parse_args(argv)
    if args.summary:
        print(render(precision(load_labels(args.labels))))
        return 0
    label(path=args.labels, by=args.by, size=args.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())

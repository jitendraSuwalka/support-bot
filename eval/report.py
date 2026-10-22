"""
The README's numbers, generated from the committed reports.

    python -m eval.report            # print the Markdown block
    python -m eval.report --write    # replace the block in README.md

Reads reports/retrieval.json (python -m eval.retrieval) and the end-to-end
runs reports/live-baseline.json and reports/live.json (python -m eval.live),
and renders two tables: retrieval recall and MRR per search mode, and the
end-to-end behaviour, escalation, correctness, guard, latency and token
figures per run, baseline first. The README carries the block between
``<!-- eval:start -->`` and ``<!-- eval:end -->``, and a test asserts that it
is exactly what this module renders from the reports in the repository, so a
number in the README can be measured and regenerated, never typed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
README = ROOT / "README.md"
START = "<!-- eval:start -->"
END = "<!-- eval:end -->"
LIVE_RUNS = (("baseline", "live-baseline.json"), ("current", "live.json"))
RETRIEVAL = "retrieval.json"


def load_reports(directory: Path = REPORTS) -> tuple[dict | None, dict[str, dict]]:
    """The retrieval report, if present, and the live runs that exist, in order."""
    retrieval_path = directory / RETRIEVAL
    retrieval = (
        json.loads(retrieval_path.read_text(encoding="utf-8")) if retrieval_path.exists() else None
    )
    runs = {}
    for label, name in LIVE_RUNS:
        path = directory / name
        if path.exists():
            runs[label] = json.loads(path.read_text(encoding="utf-8"))
    return retrieval, runs


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def retrieval_table(report: dict) -> str:
    lines = [
        f"Retrieval alone, before expansion and reranking: the {report['cases_with_gold']} cases "
        f"with gold sections of {report['cases']}, {report['chunks']} chunks, embedder "
        f"`{report['embedder']}`, fusion k = {report.get('rrf_k', 60)}, dense floor "
        f"{report['dense_floor']} (`python -m eval.retrieval`, {report['generated'][:10]}).",
        "",
        "| Mode | recall@1 | recall@3 | recall@6 | MRR | misses |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, summary in report["modes"].items():
        lines.append(
            f"| {mode} | {summary['recall@1']:.3f} | {summary['recall@3']:.3f} | "
            f"{summary['recall@6']:.3f} | {summary['mrr']:.3f} | "
            f"{len(summary['misses'])} of {summary['n']} |"
        )
    return "\n".join(lines)


def live_table(runs: dict[str, dict]) -> str:
    first = next(iter(runs.values()))
    models = first["models"]
    lines = [
        f"End to end: every case through the real agent (`{models['main']}` answering, "
        f"`{models['helper']}` for expansion, reranking and fact-checking), scored against "
        "its label (`python -m eval.live`).",
        "",
        "| Run | Date | Cases | Behaviour accuracy | Answers correct | Escalation precision | "
        "Escalation recall | Untouched by guards | Median latency | Tokens per case |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, report in runs.items():
        s = report["summary"]
        lines.append(
            f"| {label} | {report['generated'][:10]} | {s['cases']} | "
            f"{_pct(s['behaviour_accuracy'])} | {_pct(s['answers']['correct'])} | "
            f"{_pct(s['escalation']['precision'])} | {_pct(s['escalation']['recall'])} | "
            f"{_pct(s['answers']['untouched_by_guards'])} | "
            f"{s['latency_seconds']['median']:.1f} s | {s['tokens']['per_case_mean']:,} |"
        )
    lines += [
        "",
        "Behaviour accuracy: the agent did what the label expects (answer, escalate, clarify, "
        "decline, greeting). Answers correct: answerable questions answered with every required "
        "fact. Escalation precision: escalations that were wanted; recall: wanted escalations "
        "that happened. Untouched: answers that passed the citation guard and the fact-check "
        "without a rewrite. Every failure is listed by id in the report.",
    ]
    return "\n".join(lines)


def render(retrieval: dict | None, runs: dict[str, dict]) -> str:
    parts = []
    if retrieval:
        parts.append(retrieval_table(retrieval))
    if runs:
        parts.append(live_table(runs))
    if not parts:
        return "_No reports yet: run `python -m eval.retrieval` and `python -m eval.live`._"
    return "\n\n".join(parts)


def current_block(directory: Path = REPORTS) -> str:
    return render(*load_reports(directory))


def readme_block(text: str) -> str:
    """The block the README carries between the markers (without the markers)."""
    try:
        return text.split(START + "\n", 1)[1].split("\n" + END, 1)[0]
    except IndexError as error:
        raise ValueError(f"README has no {START} … {END} block") from error


def update_readme(path: Path, block: str) -> bool:
    """Replace the block between the markers; return whether the file changed."""
    text = path.read_text(encoding="utf-8")
    before = readme_block(text)
    if before == block:
        return False
    head, rest = text.split(START + "\n", 1)
    _, tail = rest.split("\n" + END, 1)
    path.write_text(head + START + "\n" + block + "\n" + END + tail, encoding="utf-8", newline="\n")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.report", description=__doc__)
    parser.add_argument("--write", action="store_true", help="update the block in the README")
    parser.add_argument("--reports", type=Path, default=REPORTS, help="where the reports are")
    parser.add_argument("--readme", type=Path, default=README, help="the README to update")
    args = parser.parse_args(argv)

    block = current_block(args.reports)
    if not args.write:
        print(block)
        return 0
    changed = update_readme(args.readme, block)
    print(f"{args.readme.name}: {'updated' if changed else 'already current'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

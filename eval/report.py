"""
The README's numbers, generated from the committed reports.

    python -m eval.report            # print the Markdown block
    python -m eval.report --write    # replace the block in README.md

Reads reports/retrieval.json (python -m eval.retrieval) and the end-to-end
runs reports/live-baseline.json and reports/live.json (python -m eval.live),
and renders two tables: retrieval recall and MRR per search mode, and the
end-to-end behaviour, escalation, correctness, guard, latency and token
figures per run, baseline first. If reports/caching-off.json and
reports/caching-on.json exist (the same cases with prompt caching off and
on), a third table compares their cost, tokens and latency. A second pair
of markers near the top of the README, ``<!-- headline:start -->`` and
``<!-- headline:end -->``, carries the four headline numbers from the same
reports. The README carries the block between
``<!-- eval:start -->`` and ``<!-- eval:end -->``, and a test asserts that it
is exactly what this module renders from the reports in the repository, so a
number in the README can be measured and regenerated, never typed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
README = ROOT / "README.md"
START = "<!-- eval:start -->"
END = "<!-- eval:end -->"
# The four headline numbers near the top of the README, from the same reports.
HEADLINE_START = "<!-- headline:start -->"
HEADLINE_END = "<!-- headline:end -->"
LIVE_RUNS = (("baseline", "live-baseline.json"), ("current", "live.json"))
RETRIEVAL = "retrieval.json"
# The same cases run with prompt caching off and on (Step 4.3).
CACHING_RUNS = (("off", "caching-off.json"), ("on", "caching-on.json"))


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


def load_caching(directory: Path = REPORTS) -> dict[str, dict]:
    """The prompt-caching comparison runs that exist, off first."""
    runs = {}
    for label, name in CACHING_RUNS:
        path = directory / name
        if path.exists():
            runs[label] = json.loads(path.read_text(encoding="utf-8"))
    return runs


def caching_table(runs: dict[str, dict]) -> str:
    """Cost, tokens and latency with prompt caching off and on, on the same cases."""
    lines = [
        "Prompt caching, measured: the same cases through the real agent with caching off, "
        "then on (`SUPPORT_BOT_PROMPT_CACHING`, `python -m eval.live --limit … --out …`).",
        "",
        "| Prompt caching | Cases | Cost per answer | Uncached input tokens per answer | "
        "Cache reads per answer | Median latency | Behaviour accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    per_case = {}
    for label, report in runs.items():
        s = report["summary"]
        n = s["cases"] or 1
        cost = (s.get("cost_usd") or {}).get("per_case_mean")
        per_case[label] = cost
        lines.append(
            f"| {label} | {s['cases']} | "
            + (f"${cost:.4f}" if cost is not None else "unpriced")
            + f" | {round(s['tokens']['input'] / n):,} | "
            f"{round(s['tokens'].get('cache_read', 0) / n):,} | "
            f"{s['latency_seconds']['median']:.1f} s | {_pct(s['behaviour_accuracy'])} |"
        )
    off, on = per_case.get("off"), per_case.get("on")
    if off and on is not None:
        lines += [
            "",
            f"Caching cut the cost of an answer by {100 * (1 - on / off):.0f}%, "
            f"from ${off:.4f} to ${on:.4f}.",
        ]
    return "\n".join(lines)


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


def headline(
    retrieval: dict | None, runs: dict[str, dict], caching: dict[str, dict] | None = None
) -> str:
    """The numbers a reader should see first, each from a report, as bullets."""
    lines = []
    if runs:
        latest = list(runs.values())[-1]["summary"]
        first = list(runs.values())[0]["summary"]
        line = (
            f"- **{_pct(latest['behaviour_accuracy'])}** of {latest['cases']} labelled customer "
            "questions handled as the label says — answered, escalated, clarified or declined"
        )
        if len(runs) > 1:
            line += f" (the first measurement: {_pct(first['behaviour_accuracy'])})"
        lines.append(line + ".")
        esc = latest["escalation"]
        lines.append(
            f"- **{_pct(esc['precision'])}** of escalations were wanted, and "
            f"**{_pct(esc['recall'])}** of the questions that needed a person reached one."
        )
    if retrieval and "hybrid" in retrieval["modes"]:
        hybrid = retrieval["modes"]["hybrid"]
        lines.append(
            f"- **{_pct(hybrid['recall@6'])}** of answerable questions have the passage that "
            "answers them among the six retrieved (hybrid BM25 and vector search)."
        )
    if caching and "off" in caching and "on" in caching:
        off = (caching["off"]["summary"].get("cost_usd") or {}).get("per_case_mean")
        on = (caching["on"]["summary"].get("cost_usd") or {}).get("per_case_mean")
        if off and on is not None:
            lines.append(
                f"- **${on:.4f}** an answer with prompt caching, "
                f"**{100 * (1 - on / off):.0f}%** less than without."
            )
    if not lines:
        return "_No reports yet._"
    return "\n".join(lines)


def render(
    retrieval: dict | None, runs: dict[str, dict], caching: dict[str, dict] | None = None
) -> str:
    parts = []
    if retrieval:
        parts.append(retrieval_table(retrieval))
    if runs:
        parts.append(live_table(runs))
    if caching:
        parts.append(caching_table(caching))
    if not parts:
        return "_No reports yet: run `python -m eval.retrieval` and `python -m eval.live`._"
    return "\n\n".join(parts)


def current_block(directory: Path = REPORTS) -> str:
    return render(*load_reports(directory), load_caching(directory))


def current_headline(directory: Path = REPORTS) -> str:
    return headline(*load_reports(directory), load_caching(directory))


def _block_pattern(start: str, end: str) -> re.Pattern:
    # The block is everything between the marker lines; it may be empty.
    return re.compile(re.escape(start) + r"\n(.*?)\n?" + re.escape(end), re.DOTALL)


def readme_block(text: str, start: str = START, end: str = END) -> str:
    """The block the README carries between the markers (without the markers)."""
    match = _block_pattern(start, end).search(text)
    if match is None:
        raise ValueError(f"README has no {start} … {end} block")
    return match.group(1)


def update_readme(path: Path, block: str, start: str = START, end: str = END) -> bool:
    """Replace the block between the markers; return whether the file changed."""
    text = path.read_text(encoding="utf-8")
    if readme_block(text, start, end) == block:
        return False
    updated = _block_pattern(start, end).sub(
        lambda _: start + "\n" + block + "\n" + end, text, count=1
    )
    path.write_text(updated, encoding="utf-8", newline="\n")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.report", description=__doc__)
    parser.add_argument("--write", action="store_true", help="update the block in the README")
    parser.add_argument("--reports", type=Path, default=REPORTS, help="where the reports are")
    parser.add_argument("--readme", type=Path, default=README, help="the README to update")
    args = parser.parse_args(argv)

    block = current_block(args.reports)
    top = current_headline(args.reports)
    if not args.write:
        print(top + "\n\n" + block)
        return 0
    changed = update_readme(args.readme, block)
    text = args.readme.read_text(encoding="utf-8")
    if HEADLINE_START in text:
        changed = update_readme(args.readme, top, HEADLINE_START, HEADLINE_END) or changed
    print(f"{args.readme.name}: {'updated' if changed else 'already current'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

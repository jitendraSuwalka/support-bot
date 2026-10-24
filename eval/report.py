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
reports. Three further sections appear when their reports exist: the
variation across repeated runs of the same code (reports/repeats/*.json,
beside reports/live.json as run 1) with a 95% confidence interval, an
ablation table of the same cases with one component switched off or the
main model changed (reports/ablations/*.json), and the precision of the
fact-checker's rewrites as labelled by hand (reports/factcheck-labels.json).
The README carries the block between
``<!-- eval:start -->`` and ``<!-- eval:end -->``, and a test asserts that it
is exactly what this module renders from the reports in the repository, so a
number in the README can be measured and regenerated, never typed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from eval.label_factcheck import rewrites, runs

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
# Repeats of the current run on the same code: reports/repeats/*.json.
REPEATS = "repeats"
# The same cases with one thing changed, in the order the table shows them:
# file, label, and the one setting that must differ from run 1, and to what.
ABLATIONS = (
    ("no-rerank.json", "Reranking off", "use_rerank", False),
    ("no-factcheck.json", "Fact-check off", "verify_answers", False),
    (
        "no-citation-checks.json",
        "Citation checks off: the guard, its re-check and the last-line check",
        "check_citations",
        False,
    ),
    ("lexical-only.json", "BM25 only, no vector search", "retrieval_mode", "lexical"),
    ("haiku-main.json", "Haiku 5.5 answering", "main_model", "claude-haiku-5-5"),
)


class ReportMismatch(ValueError):
    """A report is not the comparison its file name says it is."""


def _setting_name(field: str) -> str:
    return "SUPPORT_BOT_" + field.upper()


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


def _same_cases(report: dict, first: dict, name: str) -> None:
    if report["summary"]["cases"] != first["summary"]["cases"]:
        raise ReportMismatch(
            f"{name} covers {report['summary']['cases']} cases, run 1 covers "
            f"{first['summary']['cases']}: compare like with like"
        )


def load_repeats(directory: Path = REPORTS) -> list[dict]:
    """Run 1 (reports/live.json) and its repeats, in name order; empty without repeats.

    A repeat must be the same code and settings over the same cases, or its
    spread would not be the model's own variation.
    """
    repeats = sorted((directory / REPEATS).glob("*.json"))
    first_path = directory / "live.json"
    if not repeats or not first_path.exists():
        return []
    first = json.loads(first_path.read_text(encoding="utf-8"))
    runs = [first]
    for path in repeats:
        run = json.loads(path.read_text(encoding="utf-8"))
        _same_cases(run, first, f"repeats/{path.name}")
        if run.get("settings") != first.get("settings"):
            raise ReportMismatch(f"repeats/{path.name} ran with different settings from run 1")
        runs.append(run)
    return runs


def load_ablations(directory: Path = REPORTS) -> list[tuple[str, dict]]:
    """The ablation runs that exist, with their labels, in table order.

    Each must have run with exactly the setting its file name promises, over
    the same cases as run 1; a stray file in the folder is refused rather
    than silently left out.
    """
    folder = directory / "ablations"
    if not folder.exists():
        return []
    expected = {name for name, *_ in ABLATIONS}
    stray = sorted(path.name for path in folder.glob("*.json") if path.name not in expected)
    if stray:
        raise ReportMismatch(
            f"ablations/{', '.join(stray)}: not a known ablation; expected names: "
            + ", ".join(sorted(expected))
        )
    first_path = directory / "live.json"
    first = json.loads(first_path.read_text(encoding="utf-8")) if first_path.exists() else None
    found = []
    for name, label, field, value in ABLATIONS:
        path = folder / name
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        ran_with = (report.get("settings") or {}).get(field)
        if ran_with != value:
            raise ReportMismatch(
                f"ablations/{name} ran with {_setting_name(field)}={ran_with!r}, "
                f"not {value!r}: rerun it with --set {_setting_name(field)}={value}"
            )
        if first is not None:
            _same_cases(report, first, f"ablations/{name}")
        setting = f"`{_setting_name(field)}={str(value).lower() if value is False else value}`"
        found.append((f"{label} ({setting})", report))
    return found


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval for a proportion: sound near 0% and 100%,
    where the textbook normal interval runs past the ends."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def behaviour_counts(summary: dict) -> tuple[int, int]:
    """(cases handled as labelled, cases scored), from the lists the summary keeps."""
    scored = summary["cases"] - len(summary.get("errors", []))
    return scored - len(summary.get("wrong_behaviour", {})), scored


def behaviour_interval(summary: dict) -> tuple[float, float]:
    return wilson(*behaviour_counts(summary))


def _unsupported(summary: dict) -> str:
    ids = summary.get("unsupported_replies")
    return "not measured" if ids is None else str(len(ids))


def _cost(summary: dict) -> str:
    cost = (summary.get("cost_usd") or {}).get("per_case_mean")
    return "unpriced" if cost is None else f"${cost:.4f}"


def stability_table(runs: list[dict], baseline: dict | None = None) -> str:
    """The same code run several times: each metric per run, its mean and spread."""
    metrics = (
        ("Behaviour accuracy", lambda s: s["behaviour_accuracy"]),
        ("Answers correct", lambda s: s["answers"]["correct"]),
        ("Escalation precision", lambda s: s["escalation"]["precision"]),
        ("Escalation recall", lambda s: s["escalation"]["recall"]),
        ("Untouched by guards", lambda s: s["answers"]["untouched_by_guards"]),
    )
    summaries = [run["summary"] for run in runs]
    header = " | ".join(f"Run {i}" for i in range(1, len(runs) + 1))
    lines = [
        f"The same code, run {len(runs)} times over all {summaries[0]['cases']} cases "
        "(`reports/live.json` and `reports/repeats/`): how much of a number is the "
        "model's own variation.",
        "",
        f"| Metric | {header} | Mean | Spread |",
        "|---|" + "---:|" * (len(runs) + 2),
    ]
    for name, read in metrics:
        values = [read(s) for s in summaries]
        cells = " | ".join(_pct(v) for v in values)
        spread = 100 * (max(values) - min(values))
        lines.append(f"| {name} | {cells} | {_pct(sum(values) / len(values))} | {spread:.1f} pts |")
    costs = [(s.get("cost_usd") or {}).get("per_case_mean") for s in summaries]
    if all(c is not None for c in costs):
        lines.append(
            "| Cost per answer | "
            + " | ".join(f"${c:.4f}" for c in costs)
            + f" | ${sum(costs) / len(costs):.4f} | ${max(costs) - min(costs):.4f} |"
        )
    low, high = behaviour_interval(summaries[0])
    ok, n = behaviour_counts(summaries[0])
    lines += [
        "",
        f"Run 1 handled {ok} of {n} questions as labelled: {_pct(ok / n)}, with a 95% "
        f"confidence interval of {_pct(low)} to {_pct(high)} (Wilson). The interval is "
        "about which questions happened to be in the set; the spread across runs is about "
        "the model answering the same question differently.",
    ]
    if baseline is not None:
        before = baseline["summary"]["behaviour_accuracy"]
        if before < low:
            lines[-1] += (
                f" The baseline's {_pct(before)} lies below that interval: the improvement is "
                "larger than the uncertainty that comes from the question set."
            )
        else:
            lines[-1] += (
                f" The baseline's {_pct(before)} lies inside that interval: the difference is "
                "within the uncertainty that comes from the question set."
            )
    return "\n".join(lines)


def behaviour_spread(repeats: list[dict] | None) -> float | None:
    """How far behaviour accuracy moved between runs of the same code, in points."""
    if not repeats or len(repeats) < 2:
        return None
    values = [run["summary"]["behaviour_accuracy"] for run in repeats]
    return round(100 * (max(values) - min(values)), 1)


def ablation_table(
    full: dict, ablations: list[tuple[str, dict]], spread: float | None = None
) -> str:
    """The same cases with one thing changed, against everything switched on.

    `spread` is how far behaviour accuracy moved between repeated runs of the
    same code: a change no larger than that is not evidence of an effect.
    """
    lines = [
        "Ablation: the same cases with one component switched off, or the main model "
        "changed, one at a time (`reports/ablations/`). *Unsupported* counts replies that "
        "reached the customer with a number or link nothing supports, checked after the "
        "turn whichever guards were on.",
        "",
        "| Configuration | Behaviour accuracy | Answers correct | Escalation precision | "
        "Escalation recall | Unsupported replies | Cost per answer | Median latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    base = full["summary"]["behaviour_accuracy"]
    for label, report in [("Everything on (run 1)", full), *ablations]:
        s = report["summary"]
        delta = 100 * (s["behaviour_accuracy"] - base)
        accuracy = _pct(s["behaviour_accuracy"]) + (
            f" ({delta:+.1f})" if report is not full else ""
        )
        lines.append(
            f"| {label} | {accuracy} | {_pct(s['answers']['correct'])} | "
            f"{_pct(s['escalation']['precision'])} | {_pct(s['escalation']['recall'])} | "
            f"{_unsupported(s)} | {_cost(s)} | {s['latency_seconds']['median']:.1f} s |"
        )
    if spread is not None:
        larger = sum(
            round(abs(100 * (report["summary"]["behaviour_accuracy"] - base)), 1) > spread
            for _, report in ablations
        )
        lines += [
            "",
            f"Repeated runs of the same code differ by up to {spread:.1f} points of behaviour "
            "accuracy (the table above), so a change no larger than that is not evidence that "
            f"the component matters to behaviour. {larger} of the {len(ablations)} changes "
            "differ by more.",
        ]
    return "\n".join(lines)


def load_labels(directory: Path = REPORTS) -> dict | None:
    """The hand labels of the fact-checker's rewrites, if any have been given."""
    path = directory / "factcheck-labels.json"
    if not path.exists():
        return None
    labels = json.loads(path.read_text(encoding="utf-8"))
    return labels if labels.get("items") else None


def _runs(count: int) -> str:
    return f"{count} full run" + ("" if count == 1 else "s")


def _dates(stamps) -> str:
    """'2026-10-24T19:48:52Z' stamps as '24 October 2026' (or '24 and 25 October 2026')."""
    days = sorted({time.strptime(stamp[:10], "%Y-%m-%d") for stamp in stamps if stamp})
    if not days:
        return ""
    text = [f"{day.tm_mday} {time.strftime('%B %Y', day)}" for day in days]
    if len({time.strftime("%B %Y", day) for day in days}) == 1 and len(days) > 1:
        return " and ".join(str(day.tm_mday) for day in days[:-1]) + f" and {text[-1]}"
    return " and ".join(text)


def factcheck_table(labels: dict, now: dict | None = None) -> str:
    """How often the fact-checker's forced rewrites were needed, judged by hand.

    `now` is the runs the reports hold today and the rewrites in them: when they
    are not the runs the sample was drawn from, the table says so, because the
    labels then describe the fact-checker as it was.
    """
    counts = {"needed": 0, "not needed": 0, "unsure": 0}
    for item in labels["items"]:
        counts[item["verdict"]] += 1
    judged = counts["needed"] + counts["not needed"]
    drawn, population = labels.get("drawn_from"), labels.get("population")
    count = len(labels["items"])
    if drawn and population:
        when = _dates(drawn.values())
        source = (
            f"{count} of the {population} rewrites it forced in {_runs(len(drawn))}"
            + (f" of {when}" if when else "")
            + ", sampled by a fixed hash"
        )
    else:
        source = f"{count} of its forced rewrites, sampled from the runs"
    who = f" Labelled by {labels['labelled_by']}" if labels.get("labelled_by") else " Labelled"
    lines = [
        f"The fact-checker, judged by hand: {source}, each read against the evidence it had."
        f"{who} (`python -m eval.label_factcheck`, `reports/factcheck-labels.json`).",
        "",
        "| Verdict | Rewrites |",
        "|---|---:|",
        f"| Needed: the draft said something the evidence does not support | {counts['needed']} |",
        f"| Not needed: a false alarm | {counts['not needed']} |",
        f"| Unsure | {counts['unsure']} |",
    ]
    if judged:
        low, high = wilson(counts["needed"], judged)
        lines += [
            "",
            f"Precision {_pct(counts['needed'] / judged)} (95% CI {_pct(low)} to {_pct(high)}, "
            f"Wilson, {judged} judged either way).",
        ]
    if now and drawn and now["drawn_from"] != drawn:
        lines += [
            "",
            "These labels describe the fact-checker as it was in those runs. The runs now in "
            f"`reports/` are different ones: in {_runs(len(now['drawn_from']))} it forced "
            f"{now['population']} rewrites.",
        ]
    return "\n".join(lines)


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


def _misses(summary: dict) -> int:
    """Gold chunks not in the top six. Computed from the per-case ranks, so a
    report written when misses meant "not in the pool of ten" reads the same."""
    outcomes = summary.get("outcomes")
    if outcomes is None:
        return len(summary["misses"])
    with_gold = outcomes[: summary["n"]]  # gold cases come first; the rest have none
    deepest = 6
    return sum(1 for o in with_gold if o["gold_rank"] is None or o["gold_rank"] > deepest)


def _embedder(name: str) -> str:
    """'BAAI/bge-small-en-v1.5-384' reads like a model id; show the dimension apart."""
    model, _, dims = name.rpartition("-")
    return f"`{model}` ({dims}-d)" if model and dims.isdigit() else f"`{name}`"


def _all_tokens_per_case(summary: dict) -> int:
    """Every token processed per case, cache reads and writes included."""
    tokens = summary["tokens"]
    total = sum(tokens.get(key, 0) for key in ("input", "output", "cache_read", "cache_write"))
    return round(total / (summary["cases"] or 1))


def retrieval_table(report: dict) -> str:
    lines = [
        f"Retrieval alone, before expansion and reranking: the {report['cases_with_gold']} cases "
        f"with gold sections of {report['cases']}, {report['chunks']} chunks, embedder "
        f"{_embedder(report['embedder'])}, fusion k = {report.get('rrf_k', 60)}, dense floor "
        f"{report['dense_floor']} (`python -m eval.retrieval`, {report['generated'][:10]}).",
        "",
        "| Mode | recall@1 | recall@3 | recall@6 | MRR | not in the top six |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, summary in report["modes"].items():
        lines.append(
            f"| {mode} | {summary['recall@1']:.3f} | {summary['recall@3']:.3f} | "
            f"{summary['recall@6']:.3f} | {summary['mrr']:.3f} | "
            f"{_misses(summary)} of {summary['n']} |"
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
        "Escalation recall | Untouched by guards | Median latency | Tokens per case, all |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, report in runs.items():
        s = report["summary"]
        lines.append(
            f"| {label} | {report['generated'][:10]} | {s['cases']} | "
            f"{_pct(s['behaviour_accuracy'])} | {_pct(s['answers']['correct'])} | "
            f"{_pct(s['escalation']['precision'])} | {_pct(s['escalation']['recall'])} | "
            f"{_pct(s['answers']['untouched_by_guards'])} | "
            f"{s['latency_seconds']['median']:.1f} s | {_all_tokens_per_case(s):,} |"
        )
    lines += [
        "",
        "Behaviour accuracy: the agent did what the label expects (answer, escalate, clarify, "
        "decline, greeting). Answers correct: answerable questions answered with every required "
        "fact. Escalation precision: escalations that were wanted; recall: wanted escalations "
        "that happened. Untouched: answers that passed the citation guard and the fact-check "
        "without a rewrite. Tokens per case counts every token processed, cache reads and "
        "writes included: prompt caching makes most of them cheaper, not fewer (the caching "
        "table below shows the cost). Every failure is listed by id in the report.",
    ]
    measured = [
        (label, r["summary"]["handoffs"])
        for label, r in runs.items()
        if r["summary"].get("handoffs", {}).get("with_facts")
    ]
    if measured:
        counts = "; ".join(f"{label} {h['given']} of {h['with_facts']}" for label, h in measured)
        label, last = measured[-1]
        failed = ", ".join(f"{case} ({', '.join(why)})" for case, why in last["failed"].items())
        lines += [
            "",
            "Hand-offs that must also give the customer a team or a number (the complaints "
            f"line for a complaint, the fraud team for a scam payment): {counts}."
            + (f" Missing in the {label} run: {failed}." if failed else ""),
        ]
    return "\n".join(lines)


def headline(
    retrieval: dict | None,
    runs: dict[str, dict],
    caching: dict[str, dict] | None = None,
    repeats: list[dict] | None = None,
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
        notes = []
        if repeats:
            low, high = behaviour_interval(repeats[0]["summary"])
            values = [run["summary"]["behaviour_accuracy"] for run in repeats]
            notes.append(f"95% CI {_pct(low)}–{_pct(high)}")
            notes.append(f"{len(repeats)} runs: {_pct(min(values))}–{_pct(max(values))}")
        if len(runs) > 1:
            notes.append(f"the first measurement: {_pct(first['behaviour_accuracy'])}")
        if notes:
            line += f" ({'; '.join(notes)})"
        lines.append(line + ".")
        esc = latest["escalation"]
        lines.append(
            f"- **{_pct(esc['precision'])}** of escalations were wanted, and "
            f"**{_pct(esc['recall'])}** of the questions that needed a person reached one."
        )
    if retrieval and "hybrid" in retrieval["modes"]:
        hybrid = retrieval["modes"]["hybrid"]
        lines.append(
            f"- **{_pct(hybrid['recall@6'])}** of the {hybrid['n']} labelled questions with a "
            "gold passage have it among the six retrieved (hybrid BM25 and vector search)."
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
    retrieval: dict | None,
    runs: dict[str, dict],
    caching: dict[str, dict] | None = None,
    repeats: list[dict] | None = None,
    ablations: list[tuple[str, dict]] | None = None,
    labels: dict | None = None,
    now: dict | None = None,
) -> str:
    parts = []
    if retrieval:
        parts.append(retrieval_table(retrieval))
    if runs:
        parts.append(live_table(runs))
    if repeats:
        parts.append(stability_table(repeats, runs.get("baseline")))
    if ablations and "current" in runs:
        parts.append(ablation_table(runs["current"], ablations, behaviour_spread(repeats)))
    if labels:
        parts.append(factcheck_table(labels, now))
    if caching:
        parts.append(caching_table(caching))
    if not parts:
        return "_No reports yet: run `python -m eval.retrieval` and `python -m eval.live`._"
    return "\n\n".join(parts)


def current_block(directory: Path = REPORTS) -> str:
    return render(
        *load_reports(directory),
        load_caching(directory),
        load_repeats(directory),
        load_ablations(directory),
        load_labels(directory),
        {"drawn_from": runs(directory), "population": len(rewrites(directory))},
    )


def current_headline(directory: Path = REPORTS) -> str:
    return headline(*load_reports(directory), load_caching(directory), load_repeats(directory))


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

    try:
        block = current_block(args.reports)
        top = current_headline(args.reports)
    except ReportMismatch as error:
        print(f"report refused: {error}", file=sys.stderr)
        return 1
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

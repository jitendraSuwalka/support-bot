"""
End-to-end evaluation: every labelled case through the real agent.

Each case is run as a conversation (earlier turns first, if it has any) and
the final turn's AgentResult is scored against its label:

- behaviour: what the agent did (answer, escalate, clarify, decline,
  greeting), read from the result — escalated, declined and clarified are
  recorded by the agent itself; a greeting is a reply with no tool and no
  evidence; a question asked in prose after NO_MATCH is a clarification —
  compared with what the label expects;
- facts: whether every must_contain string appears in the answer
  (case-insensitive);
- tools: whether every expected tool was used; an escalation forced by the
  policy counts as the escalation tool, since the label is about the outcome;
- guards: which guards fired — evidence retry, citation rewrite, fact-check
  rewrite, policy escalation — and whether any WARNING note remains;
- cost: seconds, and tokens in and out, per case.

From the per-case rows: behaviour accuracy overall, by expected behaviour and
by kind; escalation precision and recall (did advice reach a human, and did
factual questions stay with the assistant); answer correctness (right
behaviour and every fact present); the share of answers that passed both
checks untouched; guard activity; latency and tokens. Every number is
traceable to the rows, and the failures are listed by id.

This costs money: about four model calls per case. Run it with
    python -m eval.live                 # all cases, report to reports/live.json
    python -m eval.live --ids isa-01,status-01
    python -m eval.live --kind advice
    python -m eval.live --limit 10
    python -m eval.live --workers 4     # four cases at a time: same results, a quarter of the wait
    python -m eval.live --set SUPPORT_BOT_USE_RERANK=false --out reports/ablations/no-rerank.json

When a label is corrected after a run, the run's rows are still what the
agent did; only the scoring is stale. ``--rescore reports/live-baseline.json``
re-scores an earlier report's rows against the labels as they now stand,
without a model call, so a baseline and a later run are compared on one set
of labels.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import agent
from eval.dataset import Case, load_cases

DEFAULT_REPORT = Path(__file__).resolve().parent.parent / "reports" / "live.json"

Runner = Callable[[list, dict], agent.AgentResult]

GUARD_EVENTS = agent.GUARD_EVENTS  # one vocabulary for the evaluation and the log


@dataclass
class Observation:
    id: str
    kind: str
    expected: str
    observed: str
    behaviour_ok: bool
    facts_ok: bool | None  # None when the label has nothing to check
    missing_facts: list[str]
    tools_ok: bool
    tools_used: list[str]
    escalated: bool
    guards: list[str]
    warning: bool
    answer: str
    seconds: float
    input_tokens: int
    output_tokens: int
    error: str | None = None
    notes: list[str] = field(default_factory=list)
    # The URLs and numbers the citation guard still could not find in the
    # evidence when the turn ended, so a warning can be read without rerunning it.
    unverified: list[str] = field(default_factory=list)
    # Added with the cost model; reports written before it load with the defaults.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None
    # Numbers and links in the final reply that the evidence, the switchboard
    # and the customer's own words do not contain, checked after the turn by
    # the same exact-match test, whichever guards were switched on: what an
    # ablation needs to show what reaches the customer. None in older reports.
    unsupported_details: list[str] | None = None
    # When the fact-checker forced a rewrite: the claims it flagged, the draft
    # it read and the evidence it read it against, for judging later whether
    # the rewrite was needed (python -m eval.label_factcheck).
    factcheck: dict | None = None


# =========================================================
# READING ONE RESULT
# =========================================================


def observe_behaviour(result: agent.AgentResult) -> str:
    """What the agent did, in the label's vocabulary (AgentResult.behaviour)."""
    return result.behaviour


def tools_credited(result: agent.AgentResult) -> set[str]:
    """The tools a turn used, with a policy-forced escalation credited as the tool."""
    used = set(result.tools_used)
    if result.escalated:
        used.add("escalate_to_adviser")
    return used


def tokens_used(result: agent.AgentResult) -> tuple[int, int]:
    events = result.trace.find("usage")
    return (
        sum(event.data.get("input_tokens", 0) for event in events),
        sum(event.data.get("output_tokens", 0) for event in events),
    )


def unsupported_details(case: Case, result: agent.AgentResult) -> list[str]:
    """The citation check on the final reply, run after the turn whatever the settings."""
    evidence = "\n\n".join([*result.evidence, agent.SWITCHBOARD])
    said = "\n".join([*case.before, case.question])
    return agent.check_citations(result.answer, evidence, said)


def factcheck_record(result: agent.AgentResult) -> dict | None:
    flagged = result.trace.find("unsupported")
    if not flagged:
        return None
    return {
        "claims": list(flagged[0].data.get("claims", [])),
        "before": flagged[0].data.get("before", ""),
        "after": result.answer,
        "evidence": "\n\n".join(result.evidence),
    }


def observe(case: Case, result: agent.AgentResult, seconds: float) -> Observation:
    observed = observe_behaviour(result)
    answer_lower = result.answer.lower()
    missing = [needle for needle in case.must_contain if needle.lower() not in answer_lower]
    guards = result.guards()
    tokens_in, tokens_out = tokens_used(result)
    cost = result.cost()
    return Observation(
        id=case.id,
        kind=case.kind,
        expected=case.expected,
        observed=observed,
        behaviour_ok=observed == case.expected,
        facts_ok=(not missing) if case.must_contain else None,
        missing_facts=missing,
        tools_ok=set(case.tools) <= tools_credited(result),
        tools_used=list(result.tools_used),
        escalated=result.escalated,
        guards=guards,
        warning=any(note.startswith("WARNING") for note in result.notes),
        answer=result.answer,
        seconds=round(seconds, 2),
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        notes=list(result.notes),
        unverified=[
            violation
            for event in result.trace.find("citations_remaining")
            for violation in event.data.get("violations", [])
        ],
        cache_read_tokens=cost.cache_read_tokens,
        cache_write_tokens=cost.cache_write_tokens,
        cost_usd=None if cost.usd is None else round(cost.usd, 6),
        unsupported_details=unsupported_details(case, result),
        factcheck=factcheck_record(result),
    )


def run_case(case: Case, runner: Runner) -> Observation:
    messages: list = []
    state = {"unclear_turns": 0}
    started = time.perf_counter()
    try:
        for earlier in case.before:
            messages.append({"role": "user", "content": earlier})
            runner(messages, state)
        messages.append({"role": "user", "content": case.question})
        result = runner(messages, state)
    except Exception as error:  # one failed call must not end a ten-minute run
        return Observation(
            id=case.id,
            kind=case.kind,
            expected=case.expected,
            observed="error",
            behaviour_ok=False,
            facts_ok=None,
            missing_facts=[],
            tools_ok=False,
            tools_used=[],
            escalated=False,
            guards=[],
            warning=True,
            answer="",
            seconds=round(time.perf_counter() - started, 2),
            input_tokens=0,
            output_tokens=0,
            error=f"{type(error).__name__}: {error}",
        )
    return observe(case, result, time.perf_counter() - started)


def rescore(report: dict, cases: list[Case]) -> dict:
    """Score an earlier report's rows against the labels as they now stand.

    The rows record what the agent did (observed, answer, tools, escalated);
    expected, the fact check and the tool check are recomputed from the
    current label, and the summary is rebuilt. The original generation time
    and settings stay; `rescored` records when the labels were applied.
    """
    by_id = {case.id: case for case in cases}
    rows = []
    for raw in report["rows"]:
        row = Observation(**raw)
        case = by_id.get(row.id)
        if case is None or row.error:
            rows.append(row)
            continue
        answer_lower = row.answer.lower()
        missing = [needle for needle in case.must_contain if needle.lower() not in answer_lower]
        credited = set(row.tools_used) | ({"escalate_to_adviser"} if row.escalated else set())
        rows.append(
            replace(
                row,
                kind=case.kind,
                expected=case.expected,
                behaviour_ok=row.observed == case.expected,
                facts_ok=(not missing) if case.must_contain else None,
                missing_facts=missing,
                tools_ok=set(case.tools) <= credited,
            )
        )
    return {
        **report,
        "rescored": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summarise(rows),
        "rows": [asdict(row) for row in rows],
    }


# =========================================================
# SUMMARY
# =========================================================


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 3) if denominator else None


def summarise(rows: list[Observation]) -> dict:
    scored = [row for row in rows if row.error is None]
    by_expected: dict[str, list[Observation]] = defaultdict(list)
    by_kind: dict[str, list[Observation]] = defaultdict(list)
    for row in scored:
        by_expected[row.expected].append(row)
        by_kind[row.kind].append(row)

    expected_escalate = [row for row in scored if row.expected == "escalate"]
    observed_escalate = [row for row in scored if row.observed == "escalate"]
    true_escalations = [row for row in expected_escalate if row.observed == "escalate"]

    answers_expected = by_expected.get("answer", [])
    answers_correct = [
        row for row in answers_expected if row.behaviour_ok and row.facts_ok is not False
    ]
    observed_answers = [row for row in scored if row.observed == "answer"]
    untouched = [
        row
        for row in observed_answers
        if not row.warning
        and not {"citation_rewrite", "factcheck_rewrite", "citation_remaining"} & set(row.guards)
    ]

    guard_counts = Counter(label for row in scored for label in row.guards)
    seconds = [row.seconds for row in scored]
    return {
        "cases": len(rows),
        "errors": [row.id for row in rows if row.error],
        "behaviour_accuracy": ratio(sum(row.behaviour_ok for row in scored), len(scored)),
        "behaviour_by_expected": {
            name: {
                "n": len(group),
                "accuracy": ratio(sum(row.behaviour_ok for row in group), len(group)),
                "observed": dict(Counter(row.observed for row in group)),
            }
            for name, group in sorted(by_expected.items())
        },
        "behaviour_by_kind": {
            name: {
                "n": len(group),
                "accuracy": ratio(sum(r.behaviour_ok for r in group), len(group)),
            }
            for name, group in sorted(by_kind.items())
        },
        "escalation": {
            "expected": len(expected_escalate),
            "observed": len(observed_escalate),
            "precision": ratio(len(true_escalations), len(observed_escalate)),
            "recall": ratio(len(true_escalations), len(expected_escalate)),
            "missed": [row.id for row in expected_escalate if row.observed != "escalate"],
            "unwanted": [row.id for row in observed_escalate if row.expected != "escalate"],
        },
        "answers": {
            "expected": len(answers_expected),
            "correct": ratio(len(answers_correct), len(answers_expected)),
            "facts_missing": {
                row.id: row.missing_facts for row in answers_expected if row.facts_ok is False
            },
            "untouched_by_guards": ratio(len(untouched), len(observed_answers)),
        },
        "tools_accuracy": ratio(sum(row.tools_ok for row in scored), len(scored)),
        "guards": dict(sorted(guard_counts.items())),
        "warnings": [row.id for row in scored if row.warning],
        # Replies carrying a number or link nothing supports, whatever the
        # guards did: None when the rows predate the measurement.
        "unsupported_replies": (
            [row.id for row in scored if row.unsupported_details]
            if any(row.unsupported_details is not None for row in scored)
            else None
        ),
        "factcheck_rewrites": sum(1 for row in scored if row.factcheck),
        "wrong_behaviour": {
            row.id: f"expected {row.expected}, observed {row.observed}"
            for row in scored
            if not row.behaviour_ok
        },
        "latency_seconds": {
            "mean": round(statistics.mean(seconds), 2) if seconds else None,
            "median": round(statistics.median(seconds), 2) if seconds else None,
            "max": max(seconds) if seconds else None,
        },
        "tokens": {
            "input": sum(row.input_tokens for row in scored),
            "output": sum(row.output_tokens for row in scored),
            # Every token the run processed, cache reads and writes included:
            # the API's input count leaves cached tokens out, so without them a
            # cached run looks smaller than it is (they are cheaper, not fewer).
            "per_case_mean": round(sum(_all_tokens(row) for row in scored) / len(scored))
            if scored
            else None,
            "cache_read": sum(row.cache_read_tokens for row in scored),
            "cache_write": sum(row.cache_write_tokens for row in scored),
        },
        "cost_usd": cost_summary(scored),
    }


def _all_tokens(row) -> int:
    return row.input_tokens + row.output_tokens + row.cache_read_tokens + row.cache_write_tokens


def cost_summary(rows: list[Observation]) -> dict:
    """Dollars per run and per case; None where a row could not be priced.

    Rows from a report written before the cost model have no cost and are
    counted as unpriced, never as free.
    """
    priced = [row.cost_usd for row in rows if row.cost_usd is not None]
    return {
        "total": round(sum(priced), 4) if priced else None,
        "per_case_mean": round(sum(priced) / len(priced), 5) if priced else None,
        "priced_cases": len(priced),
        "unpriced_cases": len(rows) - len(priced),
    }


def render(summary: dict) -> str:
    lines = [
        f"{summary['cases']} cases, {len(summary['errors'])} errors",
        f"behaviour accuracy {summary['behaviour_accuracy']}  "
        f"answers correct {summary['answers']['correct']}  "
        f"escalation precision {summary['escalation']['precision']} "
        f"recall {summary['escalation']['recall']}  "
        f"tools {summary['tools_accuracy']}",
        f"{'expected':10} {'n':>4} {'accuracy':>9}  observed",
    ]
    for name, group in summary["behaviour_by_expected"].items():
        lines.append(f"{name:10} {group['n']:>4} {group['accuracy']:>9}  {group['observed']}")
    lines.append(f"guards fired: {summary['guards']}")
    unsupported = summary.get("unsupported_replies")
    lines.append(
        f"answers untouched by guards {summary['answers']['untouched_by_guards']}; "
        f"warnings on {len(summary['warnings'])} cases"
        + (
            f"; replies with an unsupported number or link: {len(unsupported)}"
            if unsupported is not None
            else ""
        )
    )
    lines.append(
        f"latency mean {summary['latency_seconds']['mean']}s, "
        f"median {summary['latency_seconds']['median']}s; "
        f"tokens in {summary['tokens']['input']}, out {summary['tokens']['output']}"
    )
    cost = summary.get("cost_usd") or {}
    if cost.get("total") is not None:
        lines.append(
            f"cost ${cost['total']:.2f} for {cost['priced_cases']} cases, "
            f"${cost['per_case_mean']:.4f} a case; "
            f"cache {summary['tokens'].get('cache_read', 0)} read / "
            f"{summary['tokens'].get('cache_write', 0)} written"
        )
    if summary["wrong_behaviour"]:
        lines.append("wrong behaviour:")
        lines.extend(f"  {cid}: {why}" for cid, why in summary["wrong_behaviour"].items())
    if summary["answers"]["facts_missing"]:
        lines.append("facts missing:")
        lines.extend(
            f"  {cid}: {missing}" for cid, missing in summary["answers"]["facts_missing"].items()
        )
    return "\n".join(lines)


# =========================================================
# RUNNING
# =========================================================


def evaluate_cases(
    cases: list[Case], runner: Runner = agent.run_agent, progress=None, workers: int = 1
) -> list[Observation]:
    """Every case through the runner, rows in case order.

    With ``workers`` above one, that many cases run at once. Each case is its
    own conversation with its own policy state, so they are independent; the
    model calls dominate the time, so four workers take about a quarter of it.
    Progress lines appear in the order cases finish, numbered as they do.
    """
    rows: list[Observation | None] = [None] * len(cases)
    done = 0
    lock = threading.Lock()

    def one(index: int, case: Case) -> None:
        nonlocal done
        row = run_case(case, runner)
        with lock:
            rows[index] = row
            done += 1
            if progress:
                mark = "ok " if row.behaviour_ok and row.facts_ok is not False else "BAD"
                progress(
                    f"[{done:3}/{len(cases)}] {mark} {case.id:18} expected {case.expected:8} "
                    f"observed {row.observed:8} {row.seconds:5.1f}s"
                    + (f"  missing {row.missing_facts}" if row.missing_facts else "")
                    + (f"  {row.error}" if row.error else "")
                )

    if workers <= 1:
        for index, case in enumerate(cases):
            one(index, case)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(one, i, case) for i, case in enumerate(cases)]:
                future.result()
    return [row for row in rows if row is not None]


def select(cases: list[Case], ids: str | None, kind: str | None, limit: int | None) -> list[Case]:
    if ids:
        wanted = {name.strip() for name in ids.split(",")}
        unknown = wanted - {case.id for case in cases}
        if unknown:
            raise SystemExit(f"unknown case ids: {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case.id in wanted]
    if kind:
        cases = [case for case in cases if case.kind == kind]
    if limit:
        cases = cases[:limit]
    return cases


def without_set(argv: list[str]) -> list[str]:
    """The arguments with every --set NAME=VALUE taken out."""
    kept, skip = [], False
    for arg in argv:
        if skip:
            skip = False
        elif arg == "--set":
            skip = True
        elif not arg.startswith("--set="):
            kept.append(arg)
    return kept


def run_with_settings(assignments: list[str], argv: list[str]) -> int:
    """Run the evaluation again in a fresh process with some settings changed.

    The agent reads its settings once, when it is imported, so a changed
    setting needs a new process: this one checks the names and values, then
    starts ``python -m eval.live`` with them in its environment and the rest
    of the arguments unchanged. The report records the settings it ran with.
    """
    from dataclasses import fields

    from config import ENV_PREFIX, Settings

    known = {ENV_PREFIX + field.name.upper() for field in fields(Settings)}
    changes = {}
    for assignment in assignments:
        name, sep, value = assignment.partition("=")
        if not sep or not name.startswith("SUPPORT_BOT_"):
            raise SystemExit(f"--set wants SUPPORT_BOT_NAME=value, got {assignment!r}")
        if name not in known:  # a typo would otherwise run the full agent, unchanged
            raise SystemExit(f"--set: no setting called {name}")
        changes[name] = value
    try:
        Settings.from_env({**os.environ, **changes})
    except ValueError as error:
        raise SystemExit(f"--set: {error}") from None
    command = [sys.executable, "-m", "eval.live", *without_set(argv)]
    return subprocess.run(command, env={**os.environ, **changes}).returncode


def main(argv: list[str] | None = None, runner: Runner = agent.run_agent) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.live", description=__doc__)
    parser.add_argument("--ids", help="comma-separated case ids to run")
    parser.add_argument("--kind", help="only cases of this kind")
    parser.add_argument("--limit", type=int, help="run the first N selected cases")
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT, help="where to write the JSON")
    parser.add_argument(
        "--workers", type=int, default=1, help="cases run at once (default 1; 4 is a good choice)"
    )
    parser.add_argument(
        "--rescore", type=Path, help="re-score this report against the current labels, in place"
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="run with a setting changed, e.g. SUPPORT_BOT_USE_RERANK=false (repeatable)",
    )
    raw = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(raw)

    if args.set:
        return run_with_settings(args.set, raw)

    if args.rescore:
        report = rescore(json.loads(args.rescore.read_text(encoding="utf-8")), load_cases())
        args.rescore.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(render(report["summary"]))
        print(f"re-scored against the current labels: {args.rescore}")
        return 0

    cases = select(load_cases(), args.ids, args.kind, args.limit)
    if not cases:
        raise SystemExit("no cases selected")
    started = time.perf_counter()
    rows = evaluate_cases(
        cases, runner, progress=lambda line: print(line, file=sys.stderr), workers=args.workers
    )
    summary = summarise(rows)
    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": agent.models_in_use(),
        "agent_impl": agent.SETTINGS.agent_impl,
        "settings": agent.SETTINGS.describe(),
        "selection": {"ids": args.ids, "kind": args.kind, "limit": args.limit},
        "workers": args.workers,
        "seconds": round(time.perf_counter() - started, 1),
        "summary": summary,
        "rows": [asdict(row) for row in rows],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(render(summary))
    print(f"report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

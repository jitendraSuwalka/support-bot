"""
End-to-end evaluation: every labelled case through the real agent.

Each case is run as a conversation (earlier turns first, if it has any) and
the final turn's AgentResult is scored against its label:

- behaviour: what the agent did (answer, escalate, clarify, decline,
  greeting), read from the result — escalated, tools used, whether the
  evidence resolved the question, whether the reply asks a question —
  compared with what the label expects;
- facts: whether every must_contain string appears in the answer
  (case-insensitive);
- tools: whether every expected tool was used;
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
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import agent
from eval.dataset import Case, load_cases

DEFAULT_REPORT = Path(__file__).resolve().parent.parent / "reports" / "live.json"

Runner = Callable[[list, dict], agent.AgentResult]

GUARD_EVENTS = {
    "evidence_guard": "evidence_retry",
    "evidence_guard_failed": "evidence_failed",
    "citation_violation": "citation_rewrite",
    "citations_remaining": "citation_remaining",
    "unsupported": "factcheck_rewrite",
    "verification_unavailable": "factcheck_unavailable",
    "policy_escalate": "policy_escalation",
    "policy_clarify": "policy_clarify",
    "rerank_rejected_all": "rerank_rejected_all",
    "no_candidates": "no_candidates",
    "helper_failed": "helper_failed",
    "max_steps": "step_limit",
}


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


# =========================================================
# READING ONE RESULT
# =========================================================


def observe_behaviour(result: agent.AgentResult) -> str:
    """What the agent did, in the label's vocabulary."""
    if result.escalated:
        return "escalate"
    if not result.tools_used and not result.evidence:
        return "greeting"
    unresolved = any(
        marker in part for part in result.evidence for marker in agent.UNRESOLVED_MARKERS
    )
    if unresolved:
        return "clarify" if "?" in result.answer else "decline"
    return "answer"


def tokens_used(result: agent.AgentResult) -> tuple[int, int]:
    events = result.trace.find("usage")
    return (
        sum(event.data.get("input_tokens", 0) for event in events),
        sum(event.data.get("output_tokens", 0) for event in events),
    )


def observe(case: Case, result: agent.AgentResult, seconds: float) -> Observation:
    observed = observe_behaviour(result)
    answer_lower = result.answer.lower()
    missing = [needle for needle in case.must_contain if needle.lower() not in answer_lower]
    kinds = result.trace.kinds()
    guards = sorted({label for event, label in GUARD_EVENTS.items() if event in kinds})
    tokens_in, tokens_out = tokens_used(result)
    return Observation(
        id=case.id,
        kind=case.kind,
        expected=case.expected,
        observed=observed,
        behaviour_ok=observed == case.expected,
        facts_ok=(not missing) if case.must_contain else None,
        missing_facts=missing,
        tools_ok=set(case.tools) <= set(result.tools_used),
        tools_used=list(result.tools_used),
        escalated=result.escalated,
        guards=guards,
        warning=any(note.startswith("WARNING") for note in result.notes),
        answer=result.answer,
        seconds=round(seconds, 2),
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        notes=list(result.notes),
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
            "per_case_mean": round(
                sum(row.input_tokens + row.output_tokens for row in scored) / len(scored)
            )
            if scored
            else None,
        },
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
    lines.append(
        f"answers untouched by guards {summary['answers']['untouched_by_guards']}; "
        f"warnings on {len(summary['warnings'])} cases"
    )
    lines.append(
        f"latency mean {summary['latency_seconds']['mean']}s, "
        f"median {summary['latency_seconds']['median']}s; "
        f"tokens in {summary['tokens']['input']}, out {summary['tokens']['output']}"
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
    cases: list[Case], runner: Runner = agent.run_agent, progress=None
) -> list[Observation]:
    rows = []
    for index, case in enumerate(cases, start=1):
        row = run_case(case, runner)
        rows.append(row)
        if progress:
            mark = "ok " if row.behaviour_ok and row.facts_ok is not False else "BAD"
            progress(
                f"[{index:3}/{len(cases)}] {mark} {case.id:18} expected {case.expected:8} "
                f"observed {row.observed:8} {row.seconds:5.1f}s"
                + (f"  missing {row.missing_facts}" if row.missing_facts else "")
                + (f"  {row.error}" if row.error else "")
            )
    return rows


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


def main(argv: list[str] | None = None, runner: Runner = agent.run_agent) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.live", description=__doc__)
    parser.add_argument("--ids", help="comma-separated case ids to run")
    parser.add_argument("--kind", help="only cases of this kind")
    parser.add_argument("--limit", type=int, help="run the first N selected cases")
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT, help="where to write the JSON")
    args = parser.parse_args(argv)

    cases = select(load_cases(), args.ids, args.kind, args.limit)
    if not cases:
        raise SystemExit("no cases selected")
    started = time.perf_counter()
    rows = evaluate_cases(cases, runner, progress=lambda line: print(line, file=sys.stderr))
    summary = summarise(rows)
    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": agent.models_in_use(),
        "agent_impl": agent.SETTINGS.agent_impl,
        "settings": agent.SETTINGS.describe(),
        "selection": {"ids": args.ids, "kind": args.kind, "limit": args.limit},
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

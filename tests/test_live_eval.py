"""The end-to-end evaluation: how a result is read, how rows are summarised,
and the CLI — all against an injected runner, so no model and no money."""

import json
from dataclasses import asdict

import pytest

import agent
from eval import live
from eval.dataset import load_cases
from eval.live import Observation, observe, observe_behaviour, run_case, summarise
from tests.conftest import make_chunk


def result(
    answer="x", escalated=False, tools=(), evidence=(), notes=(), kinds=(), usage=None, **flags
):
    trace = agent.Trace()
    for kind in kinds:
        trace.log(kind, kind)
    for tokens_in, tokens_out in usage or ():
        trace.log("usage", "u", input_tokens=tokens_in, output_tokens=tokens_out)
    return agent.AgentResult(
        answer=answer,
        notes=list(notes),
        evidence=list(evidence),
        trace=trace,
        escalated=escalated,
        tools_used=list(tools),
        models={},
        **flags,
    )


def case(**overrides):
    from eval.dataset import Case

    values = {
        "id": "c",
        "question": "q",
        "kind": "factual",
        "expected": "answer",
        "gold": ("d/S",),
        "must_contain": (),
        "tools": ("search_knowledge_base",),
        "gold_chunk_ids": frozenset({make_chunk("d", "S", "t").chunk_id}),
        "before": (),
    }
    values.update(overrides)
    return Case(**values)


# --- reading a result -------------------------------------------------------------------


def test_behaviours_are_read_from_the_result():
    assert observe_behaviour(result(escalated=True)) == "escalate"
    assert observe_behaviour(result()) == "greeting"  # no tools, no evidence
    search = ("search_knowledge_base",)
    assert observe_behaviour(result(tools=search, evidence=["search -> [ISA] text"])) == "answer"
    assert (
        observe_behaviour(result("Which one?", tools=search, evidence=["search -> NO_MATCH"]))
        == "clarify"
    )
    # The agent records the two tool-ended behaviours itself; they outrank the heuristics.
    assert (
        observe_behaviour(result("Which one?", tools=("ask_for_clarification",), clarified=True))
        == "clarify"
    )
    assert (
        observe_behaviour(
            result(agent.DECLINE_REPLY, tools=("decline_out_of_scope",), declined=True)
        )
        == "decline"
    )
    assert (
        observe_behaviour(result("...", tools=search, evidence=["NO_MATCH"], clarified=True))
        == "clarify"
    )
    # A declarative reply on unresolved evidence is an answer (the policy would have escalated it).
    assert (
        observe_behaviour(result("I can't help with that.", tools=search, evidence=["x NO_MATCH"]))
        == "answer"
    )


def test_a_policy_forced_escalation_counts_as_the_escalation_tool():
    labelled = case(expected="escalate", tools=("search_knowledge_base", "escalate_to_adviser"))
    forced = result(escalated=True, tools=("search_knowledge_base",), notes=["escalated by policy"])
    assert observe(labelled, forced, 1.0).tools_ok is True
    unforced = result(escalated=False, tools=("search_knowledge_base",))
    assert observe(labelled, unforced, 1.0).tools_ok is False


def test_facts_tools_guards_and_tokens_are_scored():
    labelled = case(must_contain=("£20,000", "6 April"), tools=("search_knowledge_base",))
    scored = observe(
        labelled,
        result(
            "You can pay in £20,000 a year.",
            tools=("search_knowledge_base",),
            evidence=["search_knowledge_base -> [ISA] £20,000"],
            notes=["rewritten after citation check"],
            kinds=["citation_violation", "citations_clean", "verified"],
            usage=[(100, 20), (50, 5)],
        ),
        seconds=1.234,
    )
    assert scored.behaviour_ok and scored.facts_ok is False
    assert scored.missing_facts == ["6 April"]
    assert scored.tools_ok and scored.guards == ["citation_rewrite"]
    assert not scored.warning and scored.seconds == 1.23
    assert (scored.input_tokens, scored.output_tokens) == (150, 25)


def test_a_label_with_nothing_to_check_has_no_facts_verdict():
    scored = observe(case(), result(tools=("search_knowledge_base",), evidence=["e"]), 0.1)
    assert scored.facts_ok is None and scored.behaviour_ok


def test_a_runner_error_is_recorded_not_raised():
    def broken(messages, state):
        raise RuntimeError("rate limited")

    row = run_case(case(), broken)
    assert row.observed == "error" and row.error == "RuntimeError: rate limited"
    assert not row.behaviour_ok and row.warning


def test_earlier_turns_run_first_in_the_same_conversation():
    seen = []

    def runner(messages, state):
        seen.append([m["content"] for m in messages])
        messages.append({"role": "assistant", "content": "reply"})
        return result(tools=("search_knowledge_base",), evidence=["e"])

    run_case(case(before=("quiz",), question="just the quiz thing"), runner)
    assert seen == [["quiz"], ["quiz", "reply", "just the quiz thing"]]


# --- the summary ---------------------------------------------------------------------------


def obs(cid, kind, expected, observed, **over):
    values = {
        "behaviour_ok": expected == observed,
        "facts_ok": None,
        "missing_facts": [],
        "tools_ok": True,
        "tools_used": [],
        "escalated": observed == "escalate",
        "guards": [],
        "warning": False,
        "answer": "",
        "seconds": 1.0,
        "input_tokens": 10,
        "output_tokens": 1,
    }
    values.update(over)
    return Observation(cid, kind, expected, observed, **values)


def rows():
    return [
        obs("a1", "factual", "answer", "answer", facts_ok=True),
        obs("a2", "factual", "answer", "escalate", seconds=2.0, guards=["policy_escalation"]),
        obs(
            "a3",
            "factual",
            "answer",
            "answer",
            facts_ok=False,
            missing_facts=["£5"],
            seconds=3.0,
            guards=["citation_rewrite"],
        ),
        obs("e1", "advice", "escalate", "escalate"),
        obs("e2", "advice", "escalate", "answer", tools_ok=False, warning=True),
        obs(
            "x",
            "status",
            "answer",
            "error",
            tools_ok=False,
            warning=True,
            seconds=0.5,
            input_tokens=0,
            output_tokens=0,
            error="boom",
        ),
    ]


def test_summary_arithmetic():
    summary = summarise(rows())
    assert summary["cases"] == 6 and summary["errors"] == ["x"]
    assert summary["behaviour_accuracy"] == 0.6
    assert summary["escalation"] == {
        "expected": 2,
        "observed": 2,
        "precision": 0.5,
        "recall": 0.5,
        "missed": ["e2"],
        "unwanted": ["a2"],
    }
    assert summary["answers"]["expected"] == 3
    assert summary["answers"]["correct"] == 0.333  # a1 only: a3 misses a fact
    assert summary["answers"]["facts_missing"] == {"a3": ["£5"]}
    assert summary["answers"]["untouched_by_guards"] == 0.333  # a1 of a1, a3, e2
    assert summary["guards"] == {"citation_rewrite": 1, "policy_escalation": 1}
    assert summary["warnings"] == ["e2"]
    assert summary["wrong_behaviour"] == {
        "a2": "expected answer, observed escalate",
        "e2": "expected escalate, observed answer",
    }
    assert summary["latency_seconds"]["median"] == 1.0
    assert summary["tokens"] == {
        "input": 50,
        "output": 5,
        "per_case_mean": 11,
        "cache_read": 0,
        "cache_write": 0,
    }
    # The rows were built without prices, so the run is unpriced, never free.
    assert summary["cost_usd"] == {
        "total": None,
        "per_case_mean": None,
        "priced_cases": 0,
        "unpriced_cases": 5,
    }


def test_tokens_per_case_count_the_cached_tokens_too():
    """The API's input count leaves cache reads and writes out; a cached run
    processes them all the same, so the mean includes them."""
    cached = [obs("a1", "factual", "answer", "answer", cache_read_tokens=80, cache_write_tokens=9)]
    tokens = summarise(cached)["tokens"]
    assert tokens["per_case_mean"] == 10 + 1 + 80 + 9
    assert (tokens["input"], tokens["cache_read"], tokens["cache_write"]) == (10, 80, 9)


def test_render_lists_the_failures():
    text = live.render(summarise(rows()))
    assert "wrong behaviour:" in text and "a2: expected answer, observed escalate" in text
    assert "facts missing:" in text and "a3" in text


# --- selection and the CLI ---------------------------------------------------------------


def test_selection_by_ids_kind_and_limit():
    cases = load_cases()
    assert [c.id for c in live.select(cases, "isa-01,status-01", None, None)] == [
        "isa-01",
        "status-01",
    ]
    assert all(c.kind == "advice" for c in live.select(cases, None, "advice", None))
    assert len(live.select(cases, None, None, 7)) == 7
    with pytest.raises(SystemExit, match="unknown case ids"):
        live.select(cases, "nope", None, None)


def test_the_cli_runs_the_selection_through_the_runner_and_writes_the_report(tmp_path, capsys):
    def runner(messages, state):
        question = messages[-1]["content"]
        if question.lower().strip(" .!") in agent.PLEASANTRIES:
            return result("Hello!")
        return result(
            "Escalated. Call 0117 496 0100.", escalated=True, tools=("escalate_to_adviser",)
        )

    out = tmp_path / "live.json"
    code = live.main(["--ids", "greeting-01,advice-01,isa-01", "--out", str(out)], runner=runner)
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["summary"]["cases"] == 3
    assert report["summary"]["behaviour_by_expected"]["greeting"]["accuracy"] == 1.0
    assert report["summary"]["wrong_behaviour"] == {"isa-01": "expected answer, observed escalate"}
    assert {row["id"] for row in report["rows"]} == {"greeting-01", "advice-01", "isa-01"}
    assert report["agent_impl"] in ("loop", "graph")
    captured = capsys.readouterr()
    assert "report written to" in captured.out and "[  1/3]" in captured.err


def test_cases_run_in_parallel_give_the_same_rows_in_case_order():
    """Four workers finish cases in any order; the rows come back in case order
    and identical to a run one case at a time."""
    import random
    import time as clock

    cases = live.select(live.load_cases(), None, None, 12)

    def runner(messages, state):
        clock.sleep(random.uniform(0, 0.02))  # finish out of order
        return result(f"answer to {messages[-1]['content']}")

    one_at_a_time = live.evaluate_cases(cases, runner)
    lines = []
    in_parallel = live.evaluate_cases(cases, runner, progress=lines.append, workers=4)
    strip = [{**asdict(r), "seconds": 0} for r in one_at_a_time]
    assert [{**asdict(r), "seconds": 0} for r in in_parallel] == strip
    assert [r.id for r in in_parallel] == [c.id for c in cases]
    assert sorted(int(line[1:4]) for line in lines) == list(range(1, 13))


def test_the_report_records_how_many_workers_ran(tmp_path):
    out = tmp_path / "live.json"
    live.main(
        ["--ids", "greeting-01", "--out", str(out), "--workers", "2"],
        runner=lambda m, s: result("Hello!"),
    )
    assert json.loads(out.read_text(encoding="utf-8"))["workers"] == 2


# --- re-scoring an earlier run against corrected labels --------------------------------------


def test_rescore_applies_the_current_labels_to_the_recorded_rows(tmp_path):
    # A run scored when isa-01 expected "escalate" (it does not); the rows stand,
    # the scoring is redone against the labels as they are now.
    earlier = {
        "generated": "2026-10-22T19:07:44Z",
        "summary": {},
        "rows": [
            asdict(
                observe(
                    case(id="isa-01", expected="escalate", tools=("escalate_to_adviser",)),
                    result(
                        "You can pay in £20,000.",
                        tools=("search_knowledge_base",),
                        evidence=["search_knowledge_base -> [ISA] £20,000"],
                    ),
                    1.0,
                )
            ),
            asdict(
                observe(
                    case(id="status-01", kind="status", gold=(), tools=("look_up_client_query",)),
                    result("Received 1 Aug.", tools=("look_up_client_query",), evidence=["x"]),
                    1.0,
                )
            ),
        ],
    }
    earlier["rows"][1]["error"] = "boom"  # an error row is left exactly as it was
    assert earlier["rows"][0]["expected"] == "escalate" and not earlier["rows"][0]["behaviour_ok"]

    rescored = live.rescore(earlier, load_cases())
    first, second = rescored["rows"]
    assert first["expected"] == "answer" and first["behaviour_ok"] is True
    assert first["missing_facts"] == [] and first["facts_ok"] is True  # isa-01 wants £20,000
    assert first["tools_ok"] is True
    assert second == earlier["rows"][1]
    assert rescored["generated"] == "2026-10-22T19:07:44Z" and "rescored" in rescored
    assert rescored["summary"]["behaviour_accuracy"] == 1.0  # the error row is not scored
    assert rescored["summary"]["errors"] == ["status-01"]

    path = tmp_path / "live-baseline.json"
    path.write_text(json.dumps(earlier), encoding="utf-8")
    assert live.main(["--rescore", str(path)]) == 0
    assert json.loads(path.read_text(encoding="utf-8"))["rows"][0]["expected"] == "answer"


def test_the_unverified_details_are_recorded_with_the_row():
    flagged = result(
        "Call 0800 999 9999.",
        tools=("search_knowledge_base",),
        evidence=["search_knowledge_base -> [ISA] text"],
        notes=["WARNING: unverified detail after fact-check rewrite (1)"],
    )
    flagged.trace.log(
        "citations_remaining", "x", violations=["Number '0800 999 9999' does not appear"]
    )
    row = observe(case(), flagged, 1.0)
    assert row.warning is True
    assert row.unverified == ["Number '0800 999 9999' does not appear"]
    assert observe(case(), result(), 1.0).unverified == []
    # Reports written before the field existed still load and re-score.
    old = asdict(row)
    del old["unverified"]
    assert Observation(**old).unverified == []


def test_each_row_carries_its_cost_and_cache_tokens():
    priced = result(tools=("search_knowledge_base",), evidence=["search_knowledge_base -> [ISA] x"])
    priced.trace.log(
        "usage",
        "u",
        role="main",
        model="claude-sonnet-5-5",
        input_tokens=1000,
        output_tokens=100,
        cache_read_tokens=2000,
        cache_write_tokens=0,
    )
    row = observe(case(), priced, 1.0)
    assert row.cost_usd == round((1000 * 2 + 100 * 10 + 2000 * 0.1) / 1e6, 6)
    assert (row.cache_read_tokens, row.cache_write_tokens) == (2000, 0)
    summary = summarise([row])
    assert summary["cost_usd"]["total"] == round(row.cost_usd, 4)
    assert summary["cost_usd"]["priced_cases"] == 1
    assert "cost $" in live.render(summary)


# --- what the guards let through, and what the fact-checker flagged ----------------------


def test_a_reply_is_checked_for_unsupported_details_after_the_turn_whatever_the_guards():
    evidence = ["search -> [Fees] The fee is 1.2% a year."]
    clean = observe(case(), result("The fee is 1.2% a year.", evidence=evidence), 1.0)
    invented = observe(case(), result("The fee is 0.9% a year.", evidence=evidence), 1.0)
    assert clean.unsupported_details == []
    assert invented.unsupported_details == ["Number '0.9' does not appear in the evidence."]
    # The switchboard and what the customer wrote are allowed, as in the guard.
    said = case(question="I'm 70. Can I call you?")
    reply = result(f"At 70, yes: call {agent.SWITCHBOARD}.", evidence=evidence)
    assert observe(said, reply, 1.0).unsupported_details == []


def test_the_summary_counts_unsupported_replies_and_says_when_they_were_not_measured():
    measured = [
        observe(case(id="a"), result("The fee is 1.2%.", evidence=["1.2%"]), 1.0),
        observe(case(id="b"), result("The fee is 19%.", evidence=["1.2%"]), 1.0),
    ]
    assert summarise(measured)["unsupported_replies"] == ["b"]
    older = [Observation(**{**asdict(measured[0]), "unsupported_details": None})]
    assert summarise(older)["unsupported_replies"] is None


def test_a_fact_check_rewrite_is_recorded_with_its_claims_draft_and_evidence():
    flagged = result("Fees are 1.2% a year.", evidence=["search -> [Fees] 1.2% a year"])
    flagged.trace.log(
        "unsupported", "x", claims=["fees are capped"], before="Fees are 1.2%, capped."
    )
    row = observe(case(), flagged, 1.0)
    assert row.factcheck == {
        "claims": ["fees are capped"],
        "before": "Fees are 1.2%, capped.",
        "after": "Fees are 1.2% a year.",
        "evidence": "search -> [Fees] 1.2% a year",
    }
    assert observe(case(), result("x"), 1.0).factcheck is None
    assert summarise([row])["factcheck_rewrites"] == 1


# --- running with a setting changed ------------------------------------------------------


def test_set_runs_a_fresh_process_with_the_setting_and_the_other_arguments(monkeypatch):
    seen = {}

    def fake_run(command, env):
        seen["command"], seen["env"] = command, env

        class Done:
            returncode = 0

        return Done()

    monkeypatch.setattr(live.subprocess, "run", fake_run)
    argv = ["--set", "SUPPORT_BOT_USE_RERANK=false", "--workers", "4", "--out", "x.json"]
    assert live.main(argv) == 0
    assert seen["command"][1:] == ["-m", "eval.live", "--workers", "4", "--out", "x.json"]
    assert seen["env"]["SUPPORT_BOT_USE_RERANK"] == "false"


def test_set_refuses_an_unknown_name_or_a_bad_value_before_spending_anything(tmp_path):
    out = ["--out", str(tmp_path / "x.json")]
    with pytest.raises(SystemExit, match="SUPPORT_BOT_NAME=value"):
        live.main(["--set", "USE_RERANK=false", *out])
    with pytest.raises(SystemExit, match="--set"):
        live.main(["--set", "SUPPORT_BOT_USE_RERANK=perhaps", *out])
    with pytest.raises(SystemExit, match="no setting called SUPPORT_BOT_USE_RERNK"):
        live.main(["--set", "SUPPORT_BOT_USE_RERNK=false", *out])


@pytest.mark.parametrize(
    "argv", [["--set", "SUPPORT_BOT_USE_RERANK=false"], ["--limit", "20"], ["--ids", "isa-01"]]
)
def test_a_partial_or_changed_run_cannot_overwrite_run_one(argv):
    with pytest.raises(SystemExit, match="needs its own --out"):
        live.main(argv, runner=lambda m, s: result("x"))


def test_an_abbreviated_set_is_not_accepted(tmp_path):
    with pytest.raises(SystemExit):
        live.main(["--se", "SUPPORT_BOT_USE_RERANK=false", "--out", str(tmp_path / "x.json")])


def test_a_failure_while_reading_a_result_is_an_error_row_not_a_lost_run(monkeypatch):
    def broken(case, result, seconds):
        raise KeyError("trace")

    monkeypatch.setattr(live, "observe", broken)
    row = run_case(case(), lambda m, s: result("x"))
    assert row.observed == "error" and row.error == "KeyError: 'trace'"


def test_an_interrupted_parallel_run_stops_without_running_the_queue():
    import time as clock

    cases = live.select(live.load_cases(), None, None, 40)
    started = []

    def runner(messages, state):
        started.append(1)
        if len(started) == 3:
            raise KeyboardInterrupt
        clock.sleep(0.05)
        return result("x")

    with pytest.raises(KeyboardInterrupt):
        live.evaluate_cases(cases, runner, workers=4)
    clock.sleep(0.3)
    assert len(started) < 12  # the queue was cancelled, not worked through


def test_without_set_keeps_every_other_argument():
    argv = ["--set", "A=1", "--limit", "3", "--set=B=2", "--out", "o.json"]
    assert live.without_set(argv) == ["--limit", "3", "--out", "o.json"]

"""The README's evaluation numbers are rendered from the committed reports, never typed."""

import json
from pathlib import Path

import pytest

from eval import report

RETRIEVAL = {
    "generated": "2026-10-22T18:00:00Z",
    "embedder": "hash-256",
    "cases": 10,
    "cases_with_gold": 8,
    "chunks": 129,
    "dense_floor": 0.5,
    "rrf_k": 20,
    "modes": {
        "hybrid": {
            "n": 8,
            "recall@1": 0.5,
            "recall@3": 0.75,
            "recall@6": 1.0,
            "mrr": 0.7,
            "misses": [],
        },
        "dense": {
            "n": 8,
            "recall@1": 0.25,
            "recall@3": 0.5,
            "recall@6": 0.75,
            "mrr": 0.4,
            "misses": ["a", "b"],
        },
    },
}


def live(generated, accuracy):
    return {
        "generated": generated,
        "models": {"main": "main-model", "helper": "helper-model"},
        "summary": {
            "cases": 169,
            "behaviour_accuracy": accuracy,
            "answers": {"correct": 0.9, "untouched_by_guards": 0.75},
            "escalation": {"precision": 0.6, "recall": 0.95},
            "latency_seconds": {"median": 9.06},
            "tokens": {"input": 169 * 6000, "output": 169 * 335, "per_case_mean": 6335},
        },
    }


def test_the_retrieval_table_carries_every_mode_and_the_settings():
    text = report.retrieval_table(RETRIEVAL)
    assert "8 cases with gold sections of 10" in text and "fusion k = 20" in text
    assert "| hybrid | 0.500 | 0.750 | 1.000 | 0.700 | 0 of 8 |" in text
    assert "| dense | 0.250 | 0.500 | 0.750 | 0.400 | 2 of 8 |" in text


def test_a_report_from_before_the_setting_existed_shows_the_old_constant():
    older = {key: value for key, value in RETRIEVAL.items() if key != "rrf_k"}
    assert "fusion k = 60" in report.retrieval_table(older)


def test_the_live_table_lists_the_runs_in_order_with_percentages():
    text = report.live_table(
        {
            "baseline": live("2026-10-22T19:07:44Z", 0.87),
            "current": live("2026-10-24T10:00:00Z", 0.95),
        }
    )
    rows = [
        line
        for line in text.splitlines()
        if line.startswith("| baseline") or line.startswith("| current")
    ]
    assert (
        rows[0]
        == "| baseline | 2026-10-22 | 169 | 87.0% | 90.0% | 60.0% | 95.0% | 75.0% | 9.1 s | 6,335 |"
    )
    assert rows[1].startswith("| current | 2026-10-24 | 169 | 95.0% |")
    assert "`main-model` answering" in text


def test_load_reports_takes_what_exists(tmp_path):
    (tmp_path / "retrieval.json").write_text(json.dumps(RETRIEVAL), encoding="utf-8")
    (tmp_path / "live-baseline.json").write_text(
        json.dumps(live("2026-10-22T19:07:44Z", 0.87)), encoding="utf-8"
    )
    retrieval, runs = report.load_reports(tmp_path)
    assert retrieval["embedder"] == "hash-256" and list(runs) == ["baseline"]
    assert report.render(None, {}).startswith("_No reports yet")


def test_the_readme_block_is_replaced_between_the_markers_and_only_there(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Title\n\nprose\n\n<!-- eval:start -->\nold\n<!-- eval:end -->\n\nmore prose\n",
        encoding="utf-8",
    )
    assert report.update_readme(readme, "new\ntable") is True
    text = readme.read_text(encoding="utf-8")
    assert (
        text
        == "# Title\n\nprose\n\n<!-- eval:start -->\nnew\ntable\n<!-- eval:end -->\n\nmore prose\n"
    )
    assert report.update_readme(readme, "new\ntable") is False  # already current
    with pytest.raises(ValueError, match="no <!-- eval:start -->"):
        report.readme_block("no markers here")


def test_the_cli_prints_or_writes(tmp_path, capsys):
    (tmp_path / "retrieval.json").write_text(json.dumps(RETRIEVAL), encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("<!-- eval:start -->\nstale\n<!-- eval:end -->\n", encoding="utf-8")
    assert report.main(["--reports", str(tmp_path)]) == 0
    assert "| hybrid |" in capsys.readouterr().out
    assert report.main(["--reports", str(tmp_path), "--readme", str(readme), "--write"]) == 0
    assert "updated" in capsys.readouterr().out
    assert "| hybrid |" in readme.read_text(encoding="utf-8")
    assert report.main(["--reports", str(tmp_path), "--readme", str(readme), "--write"]) == 0
    assert "already current" in capsys.readouterr().out


def test_the_readme_carries_the_numbers_rendered_from_the_committed_reports():
    """Like the Mermaid diagram: the block in the README must be what the
    reports in the repository render to, so a number can only be measured."""
    text = Path(report.README).read_text(encoding="utf-8")
    assert report.readme_block(text) == report.current_block()


def caching_run(per_case, tokens_in, cache_read, median):
    return {
        "summary": {
            "cases": 20,
            "behaviour_accuracy": 1.0,
            "tokens": {"input": tokens_in, "cache_read": cache_read},
            "latency_seconds": {"median": median},
            "cost_usd": {"per_case_mean": per_case},
        }
    }


def test_the_caching_table_compares_the_same_cases_off_and_on(tmp_path):
    (tmp_path / "caching-off.json").write_text(
        json.dumps(caching_run(0.0150, 166_000, 0, 9.2)), encoding="utf-8"
    )
    (tmp_path / "caching-on.json").write_text(
        json.dumps(caching_run(0.0060, 40_000, 120_000, 8.1)), encoding="utf-8"
    )
    runs = report.load_caching(tmp_path)
    assert list(runs) == ["off", "on"]
    text = report.caching_table(runs)
    assert "| off | 20 | $0.0150 | 8,300 | 0 | 9.2 s | 100.0% |" in text
    assert "| on | 20 | $0.0060 | 2,000 | 6,000 | 8.1 s | 100.0% |" in text
    assert "cut the cost of an answer by 60%, from $0.0150 to $0.0060" in text
    assert "Prompt caching, measured" in report.render(None, {}, runs)


def test_without_both_runs_there_is_no_saving_claimed(tmp_path):
    (tmp_path / "caching-on.json").write_text(
        json.dumps(caching_run(0.006, 40_000, 120_000, 8.1)), encoding="utf-8"
    )
    text = report.caching_table(report.load_caching(tmp_path))
    assert "| on |" in text and "cut the cost" not in text
    assert report.load_caching(tmp_path / "nowhere") == {}


def test_the_headline_states_the_numbers_a_reader_sees_first():
    runs = {
        "baseline": live("2026-10-22T19:07:44Z", 0.888),
        "current": live("2026-10-24T10:00:00Z", 0.988),
    }
    caching = {"off": caching_run(0.0154, 1, 0, 9.4), "on": caching_run(0.0064, 1, 0, 9.0)}
    lines = report.headline(RETRIEVAL, runs, caching).splitlines()
    assert lines[0].startswith("- **98.8%** of 169 labelled")
    assert "(the first measurement: 88.8%)" in lines[0]
    assert lines[1] == (
        "- **60.0%** of escalations were wanted, and **95.0%** of the questions that needed "
        "a person reached one."
    )
    assert "**100.0%** of the 8 labelled questions with a gold passage" in lines[2]
    assert lines[3] == "- **$0.0064** an answer with prompt caching, **58%** less than without."
    assert report.headline(None, {}, {}) == "_No reports yet._"


def test_the_readme_carries_the_headline_rendered_from_the_committed_reports():
    text = Path(report.README).read_text(encoding="utf-8")
    block = report.readme_block(text, report.HEADLINE_START, report.HEADLINE_END)
    assert block == report.current_headline()


def test_write_updates_both_blocks_even_when_they_start_empty(tmp_path):
    (tmp_path / "retrieval.json").write_text(json.dumps(RETRIEVAL), encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(
        "<!-- headline:start -->\n<!-- headline:end -->\n\n"
        "<!-- eval:start -->\nold\n<!-- eval:end -->\n",
        encoding="utf-8",
    )
    assert report.main(["--reports", str(tmp_path), "--readme", str(readme), "--write"]) == 0
    text = readme.read_text(encoding="utf-8")
    assert "with a gold passage" in text.split("<!-- headline:end -->")[0]
    assert "| hybrid |" in text.split("<!-- eval:start -->")[1]
    assert report.readme_block("a\n<!-- eval:start -->\n<!-- eval:end -->\n") == ""


def test_misses_are_counted_at_six_from_the_ranks_whatever_the_report_called_a_miss():
    # Written when a miss meant "not in the pool of ten": gold at rank 8 was not
    # listed as a miss, but the agent, which reads six, would never see it.
    summary = {
        "n": 3,
        "misses": ["c"],
        "outcomes": [
            {"id": "a", "gold_rank": 1},
            {"id": "b", "gold_rank": 8},
            {"id": "c", "gold_rank": None},
            {"id": "no-gold", "gold_rank": None},  # cases without gold come after the n
        ],
    }
    assert report._misses(summary) == 2
    assert report._misses({"n": 3, "misses": ["c"]}) == 1  # no ranks kept: the list it has


def test_the_embedder_dimension_is_shown_apart_from_the_model_id():
    assert report._embedder("BAAI/bge-small-en-v1.5-384") == "`BAAI/bge-small-en-v1.5` (384-d)"
    assert report._embedder("hash-256") == "`hash` (256-d)"
    assert report._embedder("local") == "`local`"


def test_tokens_per_case_include_the_cache():
    summary = {
        "cases": 2,
        "tokens": {"input": 100, "output": 20, "cache_read": 300, "cache_write": 80},
    }
    assert report._all_tokens_per_case(summary) == 250


# --- repeated runs, confidence intervals and ablations ------------------------------------


SETTINGS = {
    "use_rerank": True,
    "verify_answers": True,
    "check_citations": True,
    "retrieval_mode": "hybrid",
    "main_model": "claude-sonnet-5-5",
}


def full_run(accuracy_wrong, untouched=0.8, cost=0.005, unsupported=(), median=8.0, **changed):
    """A run's summary as eval.live writes it, with `accuracy_wrong` wrong behaviours."""
    wrong = {f"c{i}": "expected answer, observed escalate" for i in range(accuracy_wrong)}
    return {
        "generated": "2026-10-25T10:00:00Z",
        "models": {"main": "main-model", "helper": "helper-model"},
        "settings": {**SETTINGS, **changed},
        "summary": {
            "cases": 169,
            "errors": [],
            "wrong_behaviour": wrong,
            "behaviour_accuracy": round((169 - accuracy_wrong) / 169, 3),
            "answers": {"correct": 0.99, "untouched_by_guards": untouched},
            "escalation": {"precision": 0.962, "recall": 0.962},
            "latency_seconds": {"median": median},
            "tokens": {"input": 1, "output": 1},
            "cost_usd": {"per_case_mean": cost},
            "unsupported_replies": list(unsupported),
        },
    }


def test_the_wilson_interval_matches_known_values():
    low, high = report.wilson(50, 100)
    assert (round(low, 4), round(high, 4)) == (0.4038, 0.5962)
    assert report.wilson(0, 10)[0] == 0.0 and report.wilson(10, 10)[1] == 1.0
    low, high = report.wilson(167, 169)
    assert (round(low, 3), round(high, 3)) == (0.958, 0.997)


def test_behaviour_counts_come_from_the_lists_not_the_rounded_ratio():
    summary = full_run(2)["summary"]
    assert report.behaviour_counts(summary) == (167, 169)
    summary["errors"] = ["x"]
    assert report.behaviour_counts(summary) == (166, 168)


def test_the_stability_table_shows_each_run_its_mean_spread_and_interval():
    runs = [full_run(2), full_run(3, untouched=0.82), full_run(2, cost=0.006)]
    text = report.stability_table(runs, baseline=live("2026-10-22T19:07:44Z", 0.888))
    assert "| Behaviour accuracy | 98.8% | 98.2% | 98.8% | 98.6% | 0.6 pts |" in text
    assert "| Cost per answer | $0.0050 | $0.0050 | $0.0060 | $0.0053 | $0.0010 |" in text
    assert "Run 1 handled 167 of 169 questions as labelled: 98.8%" in text
    assert "95.8% to 99.7% (Wilson)" in text
    assert "The baseline's 88.8% lies below that interval" in text
    near = report.stability_table(runs, baseline=live("x", 0.97))
    assert "lies inside that interval" in near


def test_the_ablation_table_compares_each_change_with_everything_on():
    ablations = [
        ("Reranking off", full_run(9, cost=0.004)),
        ("Citation guard off", full_run(2, unsupported=("a", "b"))),
    ]
    older = full_run(2)
    older["summary"]["unsupported_replies"] = None
    text = report.ablation_table(full_run(2), [*ablations, ("Older", older)])
    assert "| Everything on (run 1) | 98.8% | 99.0% | 96.2% | 96.2% | 0 | $0.0050 | 8.0 s |" in text
    assert "| Reranking off | 94.7% (-4.1) |" in text
    assert "| Citation guard off | 98.8% (+0.0) | 99.0% | 96.2% | 96.2% | 2 |" in text
    assert "| Older | 98.8% (+0.0) | 99.0% | 96.2% | 96.2% | not measured |" in text


def test_repeats_and_ablations_are_read_from_their_folders(tmp_path):
    assert report.load_repeats(tmp_path) == [] and report.load_ablations(tmp_path) == []
    (tmp_path / "live.json").write_text(json.dumps(full_run(2)), encoding="utf-8")
    (tmp_path / "repeats").mkdir()
    (tmp_path / "repeats" / "live-2.json").write_text(json.dumps(full_run(3)), encoding="utf-8")
    (tmp_path / "ablations").mkdir()
    (tmp_path / "ablations" / "haiku-main.json").write_text(
        json.dumps(full_run(10, main_model="claude-haiku-5-5")), encoding="utf-8"
    )
    (tmp_path / "ablations" / "no-rerank.json").write_text(
        json.dumps(full_run(9, use_rerank=False)), encoding="utf-8"
    )
    repeats = report.load_repeats(tmp_path)
    assert [len(r["summary"]["wrong_behaviour"]) for r in repeats] == [2, 3]
    labels = [label for label, _ in report.load_ablations(tmp_path)]
    assert labels == [
        "Reranking off (`SUPPORT_BOT_USE_RERANK=false`)",
        "Haiku 5.5 answering (`SUPPORT_BOT_MAIN_MODEL=claude-haiku-5-5`)",
    ]
    block = report.current_block(tmp_path)
    assert "run 2 times" in block and "Ablation:" in block
    top = report.current_headline(tmp_path)
    assert "(95% CI 95.8%–99.7%; 2 runs: 98.2%–98.8%)" in top


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_an_ablation_that_did_not_run_with_its_setting_is_refused(tmp_path):
    write(tmp_path / "live.json", full_run(2))
    write(tmp_path / "ablations" / "no-rerank.json", full_run(9))  # --set was forgotten
    with pytest.raises(report.ReportMismatch, match="SUPPORT_BOT_USE_RERANK=True"):
        report.load_ablations(tmp_path)


def test_an_ablation_over_other_cases_or_a_stray_file_is_refused(tmp_path):
    write(tmp_path / "live.json", full_run(2))
    short = full_run(0, use_rerank=False)
    short["summary"]["cases"] = 20
    write(tmp_path / "ablations" / "no-rerank.json", short)
    with pytest.raises(report.ReportMismatch, match="covers 20 cases, run 1 covers 169"):
        report.load_ablations(tmp_path)
    (tmp_path / "ablations" / "no-rerank.json").unlink()
    write(tmp_path / "ablations" / "no-rerank-2.json", full_run(0, use_rerank=False))
    with pytest.raises(report.ReportMismatch, match="not a known ablation"):
        report.load_ablations(tmp_path)


def test_a_repeat_with_other_settings_is_refused(tmp_path):
    write(tmp_path / "live.json", full_run(2))
    write(tmp_path / "repeats" / "live-2.json", full_run(2, use_rerank=False))
    with pytest.raises(report.ReportMismatch, match="different settings from run 1"):
        report.load_repeats(tmp_path)


def test_the_cli_reports_a_mismatch_plainly(tmp_path, capsys):
    write(tmp_path / "live.json", full_run(2))
    write(tmp_path / "ablations" / "no-rerank.json", full_run(9))
    assert report.main(["--reports", str(tmp_path)]) == 1
    assert "ablations/no-rerank.json ran with" in capsys.readouterr().err


def test_the_live_table_says_which_hand_offs_gave_their_facts():
    old, new = live("2026-10-22T19:07:44Z", 0.888), live("2026-10-25T10:00:00Z", 0.982)
    assert "Hand-offs that must" not in report.live_table({"baseline": old, "current": new})
    old["summary"]["handoffs"] = {"with_facts": 3, "given": 1, "failed": {"x": ["F"], "y": ["G"]}}
    new["summary"]["handoffs"] = {"with_facts": 3, "given": 2, "failed": {"x": ["F"]}}
    text = report.live_table({"baseline": old, "current": new})
    assert "baseline 1 of 3; current 2 of 3. Missing in the current run: x (F)." in text

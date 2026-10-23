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
            "tokens": {"per_case_mean": 6335},
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
    assert "**100.0%** of answerable questions" in lines[2]
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
    assert "of answerable questions" in text.split("<!-- headline:end -->")[0]
    assert "| hybrid |" in text.split("<!-- eval:start -->")[1]
    assert report.readme_block("a\n<!-- eval:start -->\n<!-- eval:end -->\n") == ""

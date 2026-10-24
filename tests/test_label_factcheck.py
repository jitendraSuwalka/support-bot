"""Judging the fact-checker by hand: the sample, the resumable labelling, and
the precision the README shows. Answers are scripted; nothing is typed."""

import json

from eval import label_factcheck as lf
from eval import report


def row(case_id, before, claims=("a claim",)):
    return {
        "id": case_id,
        "factcheck": {
            "claims": list(claims),
            "before": before,
            "after": f"rewritten {before}",
            "evidence": "the evidence",
        },
    }


def write_runs(directory, first, *repeats):
    (directory / "live.json").write_text(json.dumps({"rows": first}), encoding="utf-8")
    if repeats:
        (directory / "repeats").mkdir()
    for number, rows in enumerate(repeats, start=2):
        path = directory / "repeats" / f"live-{number}.json"
        path.write_text(json.dumps({"rows": rows}), encoding="utf-8")


def scripted(*answers):
    queue = list(answers)
    return lambda prompt: queue.pop(0)


def test_rewrites_are_gathered_from_every_run_once_per_case_and_draft(tmp_path):
    write_runs(
        tmp_path,
        [row("isa-01", "draft A"), {"id": "fees-01", "factcheck": None}],
        [row("isa-01", "draft A"), row("isa-01", "draft B")],  # same draft again, then a new one
    )
    found = lf.rewrites(tmp_path)
    assert [(item["id"], item["before"], item["run"]) for item in found] == [
        ("isa-01", "draft A", "live.json"),
        ("isa-01", "draft B", "live-2.json"),
    ]


def test_the_sample_is_the_same_every_time_and_capped():
    items = [{"id": f"c{i}", "before": f"draft {i}"} for i in range(60)]
    first = lf.sample(items, 40)
    assert len(first) == 40 and first == lf.sample(list(reversed(items)), 40)


def test_labelling_saves_each_answer_and_carries_on_where_it_stopped(tmp_path):
    write_runs(tmp_path, [row("isa-01", "one"), row("isa-02", "two"), row("isa-03", "three")])
    path = tmp_path / "labels.json"
    order = [item["id"] for item in lf.sample(lf.rewrites(tmp_path))]
    said = []

    # First sitting: needed, then a false alarm with a note, then quit.
    lf.label(
        tmp_path,
        path,
        by="Jay",
        ask=scripted("y", "n", "fee is in the evidence", "q"),
        say=said.append,
    )
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["labelled_by"] == "Jay"
    assert [(i["id"], i["verdict"]) for i in saved["items"]] == [
        (order[0], "needed"),
        (order[1], "not needed"),
    ]
    assert saved["items"][1]["note"] == "fee is in the evidence"

    # Second sitting: only the third item is shown; a typo is asked again.
    said.clear()
    lf.label(tmp_path, path, ask=scripted("x", "u", ""), say=said.append)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert [i["verdict"] for i in saved["items"]] == ["needed", "not needed", "unsure"]
    assert saved["labelled_by"] == "Jay"  # kept from the first sitting
    assert any("2 already labelled" in line for line in said)
    assert any("Please type y, n, u, s or q." in line for line in said)
    assert sum(f"case {order[2]}" in line for line in said) == 1


def test_a_skipped_item_is_shown_again_next_time(tmp_path):
    write_runs(tmp_path, [row("isa-01", "one")])
    path = tmp_path / "labels.json"
    lf.label(tmp_path, path, ask=scripted("s"), say=lambda line: None)
    assert json.loads(path.read_text(encoding="utf-8"))["items"] == []
    lf.label(tmp_path, path, ask=scripted("y"), say=lambda line: None)
    assert len(json.loads(path.read_text(encoding="utf-8"))["items"]) == 1


def test_precision_counts_only_the_rewrites_judged_either_way():
    labels = {
        "items": [
            {"id": "a", "verdict": "needed"},
            {"id": "b", "verdict": "needed"},
            {"id": "c", "verdict": "needed"},
            {"id": "d", "verdict": "not needed"},
            {"id": "e", "verdict": "unsure"},
        ]
    }
    result = lf.precision(labels)
    assert result["precision"] == 0.75 and result["not_needed_ids"] == ["d"]
    assert lf.render(result).endswith("precision 75.0%")
    assert lf.precision({"items": []})["precision"] is None


def test_the_readme_section_shows_the_verdicts_and_the_interval(tmp_path):
    labels = {
        "labelled_by": "Jitendra Suwalka",
        "items": [{"id": str(i), "verdict": "needed"} for i in range(30)]
        + [{"id": f"n{i}", "verdict": "not needed"} for i in range(8)]
        + [{"id": "u", "verdict": "unsure"}],
    }
    text = report.factcheck_table(labels)
    assert "39 of its forced rewrites" in text and "by Jitendra Suwalka" in text
    assert "| Not needed: a false alarm | 8 |" in text
    assert "Precision 78.9% (95% CI" in text and "38 judged either way" in text
    (tmp_path / "factcheck-labels.json").write_text(json.dumps(labels), encoding="utf-8")
    assert report.load_labels(tmp_path)["labelled_by"] == "Jitendra Suwalka"
    assert report.load_labels(tmp_path / "none") is None


def test_the_summary_option_prints_the_precision(monkeypatch, tmp_path, capsys):
    path = tmp_path / "labels.json"
    lf.save_labels({"labelled_by": None, "started": None, "items": []}, path)
    monkeypatch.setattr(lf, "LABELS", path)
    monkeypatch.setattr(lf, "load_labels", lambda p=path: json.loads(path.read_text()))
    assert lf.main(["--summary"]) == 0
    assert "labelled 0" in capsys.readouterr().out

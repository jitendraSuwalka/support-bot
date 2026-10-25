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


def write_runs(directory, first, *repeats, generated="2026-10-24T19:48:52Z"):
    (directory / "live.json").write_text(
        json.dumps({"generated": generated, "rows": first}), encoding="utf-8"
    )
    if repeats:
        (directory / "repeats").mkdir(exist_ok=True)
    for number, rows in enumerate(repeats, start=2):
        path = directory / "repeats" / f"live-{number}.json"
        path.write_text(json.dumps({"generated": generated, "rows": rows}), encoding="utf-8")


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
    assert any("Please type one letter: y, n, u, s or q." in line for line in said)
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


def test_a_note_typed_at_the_verdict_prompt_is_not_taken_as_a_verdict(tmp_path):
    # "sooner is ..." once began with "s" and skipped the item.
    write_runs(tmp_path, [row("isa-01", "one")])
    path = tmp_path / "labels.json"
    said = []
    lf.label(tmp_path, path, ask=scripted("sooner is fine", "n", "a note"), say=said.append)
    [item] = json.loads(path.read_text(encoding="utf-8"))["items"]
    assert (item["verdict"], item["note"]) == ("not needed", "a note")
    assert any("Please type one letter" in line for line in said)


def test_the_labels_record_the_runs_their_sample_came_from(tmp_path):
    write_runs(tmp_path, [row("isa-01", "one")], [row("isa-02", "two")])
    path = tmp_path / "labels.json"
    lf.label(tmp_path, path, ask=scripted("y", "q"), say=lambda line: None)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["drawn_from"] == {
        "live.json": "2026-10-24T19:48:52Z",
        "repeats/live-2.json": "2026-10-24T19:48:52Z",
    }
    assert saved["population"] == 2


def test_labels_from_earlier_runs_are_never_mixed_with_new_ones(tmp_path):
    write_runs(tmp_path, [row("isa-01", "one")])
    path = tmp_path / "labels.json"
    lf.label(tmp_path, path, ask=scripted("y"), say=lambda line: None)
    before = path.read_text(encoding="utf-8")

    write_runs(tmp_path, [row("isa-01", "a new draft")], generated="2026-11-01T09:00:00Z")
    said = []
    lf.label(tmp_path, path, ask=scripted("n", ""), say=said.append)
    assert path.read_text(encoding="utf-8") == before  # nothing added
    assert any("--labels" in line for line in said)

    fresh = tmp_path / "labels-new.json"  # the new runs go in their own file
    lf.label(tmp_path, fresh, ask=scripted("n", ""), say=lambda line: None)
    assert json.loads(fresh.read_text(encoding="utf-8"))["items"][0]["before"] == "a new draft"


def test_the_table_names_the_runs_and_says_when_the_reports_have_moved_on():
    drawn = {"live.json": "2026-10-24T19:48:52Z", "repeats/live-2.json": "2026-10-25T08:00:00Z"}
    labels = {
        "labelled_by": "Jitendra Suwalka, with an AI second reader",
        "drawn_from": drawn,
        "population": 106,
        "items": [{"id": "a", "verdict": "needed"}, {"id": "b", "verdict": "not needed"}],
    }
    same = report.factcheck_table(labels, {"drawn_from": drawn, "population": 106})
    assert "2 of the 106 rewrites it forced in 2 full runs of 24 and 25 October 2026" in same
    assert "Labelled by Jitendra Suwalka, with an AI second reader" in same
    assert "describe the fact-checker as it was" not in same
    later = {"drawn_from": {"live.json": "2026-11-01T09:00:00Z"}, "population": 12}
    moved = report.factcheck_table(labels, later)
    assert "in 1 full run it forced 12 rewrites" in moved

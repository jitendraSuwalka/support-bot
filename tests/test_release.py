"""A release is published only when the tag, the code and the changelog agree,
and its notes carry measured numbers, never typed ones."""

import shutil
from pathlib import Path

import pytest

import api
from eval import report
from scripts import release_notes

ROOT = Path(__file__).resolve().parent.parent

CHANGELOG = """# Changelog

Intro prose.

## [Unreleased]

## [1.1.0] - 2026-11-01

### Added

- A thing.

## [1.0.0] - 2026-08-06

The first version.

[1.1.0]: https://example.test/compare/v1.0.0...v1.1.0
[1.0.0]: https://example.test/tree/v1.0.0
"""


def project(tmp_path, version="1.1.0", served=None, changelog=CHANGELOG):
    """A minimal project tree: pyproject.toml, api.py, CHANGELOG.md, no reports."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "x"\nversion = "{version}"\n', encoding="utf-8"
    )
    (tmp_path / "api.py").write_text(
        f'"""doc"""\n\nVERSION = "{served or version}"\n', encoding="utf-8"
    )
    (tmp_path / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    return tmp_path


# --- the repository itself ---------------------------------------------------------------


def test_the_version_agrees_everywhere_in_the_repository():
    entry = release_notes.check()
    assert entry.version == api.VERSION == release_notes.pyproject_version()
    assert entry.date is not None


def test_the_notes_for_this_version_carry_the_entry_the_measured_numbers_and_the_image():
    entry = release_notes.check(f"v{api.VERSION}")
    text = release_notes.notes(entry)
    assert text.startswith(entry.body) and "### Added" in text
    assert report.current_headline() in text  # the README's headline, from the reports
    assert f"docker pull {release_notes.IMAGE}:{api.VERSION}" in text
    assert f"**Full changes:** {entry.link}" in text and entry.link.endswith(f"v{api.VERSION}")
    assert "]: https://" not in text  # the link definitions stay in the changelog


def test_a_tag_for_another_version_is_refused():
    with pytest.raises(release_notes.ReleaseError, match=f"expected v{api.VERSION}"):
        release_notes.check("v9.9.9")


def test_the_image_name_is_lower_case_as_registries_require():
    assert release_notes.IMAGE.lower() == release_notes.IMAGE


# --- reading the changelog ---------------------------------------------------------------


def test_entries_are_read_newest_first_with_their_links_and_without_the_definitions():
    entries = release_notes.changelog_entries(CHANGELOG)
    assert [(e.version, e.date) for e in entries] == [
        ("Unreleased", None),
        ("1.1.0", "2026-11-01"),
        ("1.0.0", "2026-08-06"),
    ]
    assert entries[0].body == ""
    assert entries[1].body == "### Added\n\n- A thing."
    assert entries[2].body == "The first version."
    assert entries[1].link == "https://example.test/compare/v1.0.0...v1.1.0"


# --- what is refused, and that every problem is named at once ----------------------------


def test_a_consistent_project_passes_with_or_without_the_tag(tmp_path):
    root = project(tmp_path)
    assert release_notes.check(root=root).version == "1.1.0"
    assert release_notes.check("v1.1.0", root=root).date == "2026-11-01"


def test_a_service_reporting_another_version_is_refused(tmp_path):
    with pytest.raises(release_notes.ReleaseError, match="api.VERSION is 1.1.0-dev"):
        release_notes.check(root=project(tmp_path, served="1.1.0-dev"))


def test_a_development_version_cannot_be_released(tmp_path):
    with pytest.raises(release_notes.ReleaseError, match="not MAJOR.MINOR.PATCH"):
        release_notes.check(root=project(tmp_path, version="1.2.0.dev0"))


def test_a_version_without_its_changelog_entry_is_refused(tmp_path):
    with pytest.raises(release_notes.ReleaseError, match="no entry for 1.2.0"):
        release_notes.check(root=project(tmp_path, version="1.2.0"))


def test_releasing_an_older_entry_is_refused(tmp_path):
    with pytest.raises(release_notes.ReleaseError, match="newest entry is 1.1.0, not 1.0.0"):
        release_notes.check(root=project(tmp_path, version="1.0.0"))


def test_an_undated_entry_is_refused(tmp_path):
    undated = CHANGELOG.replace("## [1.1.0] - 2026-11-01", "## [1.1.0]")
    with pytest.raises(release_notes.ReleaseError, match="entry for 1.1.0 has no date"):
        release_notes.check(root=project(tmp_path, changelog=undated))


def test_changes_left_under_unreleased_block_the_release_but_not_the_commit(tmp_path):
    pending = CHANGELOG.replace("## [Unreleased]\n", "## [Unreleased]\n\n- Not yet released.\n")
    root = project(tmp_path, changelog=pending)
    assert release_notes.check(root=root).version == "1.1.0"  # fine while developing
    with pytest.raises(release_notes.ReleaseError, match="under Unreleased"):
        release_notes.check("v1.1.0", root=root)


def test_every_problem_is_named_in_one_refusal(tmp_path):
    root = project(tmp_path, version="1.2.0", served="1.1.0")
    with pytest.raises(release_notes.ReleaseError) as refused:
        release_notes.check("v1.1.0", root=root)
    message = str(refused.value)
    assert "api.VERSION is 1.1.0" in message
    assert "no entry for 1.2.0" in message
    assert "expected v1.2.0" in message


# --- the command line --------------------------------------------------------------------


def test_the_cli_checks_writes_the_notes_and_refuses(tmp_path, capsys):
    root = project(tmp_path / "project")
    shutil.copytree(ROOT / "reports", root / "reports")
    assert release_notes.main(["--check", "--root", str(root)]) == 0
    assert "version 1.1.0 agrees" in capsys.readouterr().out

    out = tmp_path / "notes.md"
    argv = ["v1.1.0", "--root", str(root), "--out", str(out), "--image", "ghcr.io/o/r"]
    assert release_notes.main(argv) == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("### Added\n\n- A thing.\n\n### Measured")
    assert "docker pull ghcr.io/o/r:1.1.0" in text
    assert report.current_headline(root / "reports") in text

    assert release_notes.main(["v1.0.0", "--root", str(root)]) == 1
    assert "release refused: the tag is v1.0.0" in capsys.readouterr().err


def test_the_cli_needs_a_tag_or_check():
    with pytest.raises(SystemExit) as stopped:
        release_notes.main([])
    assert stopped.value.code == 2

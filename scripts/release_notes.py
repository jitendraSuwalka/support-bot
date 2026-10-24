"""
The notes for a release, and the check that decides whether it may be published.

    python -m scripts.release_notes --check                  # the version agrees everywhere
    python -m scripts.release_notes v2.0.0 --out notes.md    # the notes for that tag, or a refusal

A release is published by pushing a version tag (.github/workflows/release.yml),
and before anything is built this script holds the tag to the code:

- the version in pyproject.toml is MAJOR.MINOR.PATCH;
- ``api.VERSION``, which the service reports at /health, is the same version;
- CHANGELOG.md has a dated entry for it, and nothing waiting under Unreleased;
- the tag is ``v`` followed by that version.

Any disagreement is a refusal that names every problem at once. The notes are
the changelog entry, followed by the headline numbers rendered from the
committed reports by ``eval.report`` (the lines the README opens with, so the
release page carries measured numbers, never typed ones), how to run the
published image, and the link to the full changes. It needs nothing beyond
the standard library (``eval.report`` included), so the release job can run it
before installing anything.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from eval import report

ROOT = Path(__file__).resolve().parent.parent
# The image name the release workflow publishes; it passes its own with --image.
IMAGE = "ghcr.io/jitendrasuwalka/support-bot"
UNRELEASED = "unreleased"

_HEADING = re.compile(
    r"^## \[(?P<version>[^\]]+)\](?: - (?P<date>\d{4}-\d{2}-\d{2}))?[ \t]*$", re.M
)
_LINK = re.compile(r"^\[(?P<label>[^\]]+)\]: (?P<url>\S+)[ \t]*$", re.M)
_API_VERSION = re.compile(r'^VERSION = "(?P<version>[^"]+)"$', re.M)
_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


class ReleaseError(Exception):
    """The tag, the code and the changelog do not agree; nothing is published."""


@dataclass(frozen=True)
class Entry:
    version: str
    date: str | None
    body: str
    link: str | None = None


def pyproject_version(root: Path = ROOT) -> str:
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def api_version(root: Path = ROOT) -> str | None:
    """The version the service reports, read from api.py without importing it."""
    match = _API_VERSION.search((root / "api.py").read_text(encoding="utf-8"))
    return match["version"] if match else None


def changelog_entries(text: str) -> list[Entry]:
    """Every ``## [version] - date`` entry, newest first, with its link if defined."""
    links = {m["label"]: m["url"] for m in _LINK.finditer(text)}
    headings = list(_HEADING.finditer(text))
    entries = []
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        # Link definitions at the foot of the file belong to the file, not to an entry.
        body = _LINK.sub("", text[heading.end() : end]).strip()
        version = heading["version"]
        entries.append(Entry(version, heading["date"], body, links.get(version)))
    return entries


def check(tag: str | None = None, root: Path = ROOT) -> Entry:
    """The changelog entry for the version, if everything agrees; else ReleaseError.

    Without a tag this is the check a test runs on every commit: the version is
    consistent and has its entry. With a tag it is the check before publishing.
    """
    version = pyproject_version(root)
    problems = []
    if not _SEMVER.match(version):
        problems.append(f"pyproject.toml's version {version} is not MAJOR.MINOR.PATCH")
    served = api_version(root)
    if served is None:
        problems.append('api.py has no VERSION = "..." line')
    elif served != version:
        problems.append(f"api.VERSION is {served} but pyproject.toml says {version}")

    entries = changelog_entries((root / "CHANGELOG.md").read_text(encoding="utf-8"))
    released = [e for e in entries if e.version.lower() != UNRELEASED]
    entry = next((e for e in released if e.version == version), None)
    if entry is None:
        problems.append(f"CHANGELOG.md has no entry for {version}")
    else:
        if entry is not released[0]:
            problems.append(f"CHANGELOG.md's newest entry is {released[0].version}, not {version}")
        if entry.date is None:
            problems.append(f"CHANGELOG.md's entry for {version} has no date")

    if tag is not None:
        if tag != f"v{version}":
            problems.append(f"the tag is {tag}, but the version is {version}: expected v{version}")
        if any(e.version.lower() == UNRELEASED and e.body for e in entries):
            problems.append("CHANGELOG.md has changes under Unreleased: move them into the entry")

    if problems:
        raise ReleaseError("; ".join(problems))
    return entry


def notes(entry: Entry, root: Path = ROOT, image: str = IMAGE) -> str:
    """The release page: the changelog entry, the measured headline, how to run it."""
    parts = [
        entry.body,
        "### Measured",
        "From the reports committed in `reports/`, as on the README:\n\n"
        + report.current_headline(root / "reports"),
        "### Run it",
        "With your Anthropic API key in a file `.env` (one line, `ANTHROPIC_API_KEY=sk-ant-...`):"
        "\n\n```bash\n"
        f"docker pull {image}:{entry.version}\n"
        f"docker run --env-file .env -p 127.0.0.1:8000:8000 {image}:{entry.version}\n"
        "```\n\n"
        "Then open http://127.0.0.1:8000/docs. The image carries the embedding model and the "
        "document index, runs as a non-root user that cannot rewrite its own code, and makes "
        "no outbound call except to the Anthropic API.",
    ]
    if entry.link:
        parts.append(f"**Full changes:** {entry.link}")
    return "\n\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.release_notes",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("tag", nargs="?", help="the version tag being released, e.g. v2.0.0")
    parser.add_argument("--check", action="store_true", help="only check the version agrees")
    parser.add_argument("--out", type=Path, help="write the notes here (UTF-8), not to stdout")
    parser.add_argument("--image", default=IMAGE, help="the image the notes say to pull")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.check and not args.tag:
        parser.error("give the tag being released, or --check")

    try:
        entry = check(args.tag, args.root)
    except ReleaseError as error:
        print(f"release refused: {error}", file=sys.stderr)
        return 1
    if args.check:
        print(
            f"version {entry.version} agrees: pyproject.toml, api.py and CHANGELOG.md "
            f"(released {entry.date})"
        )
        return 0
    text = notes(entry, args.root, args.image)
    if args.out:
        args.out.write_text(text, encoding="utf-8", newline="\n")
        print(f"release notes for {args.tag} written to {args.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Machine-checkable guards against documentation drift.

A documentation audit of this repo found ~40 stale claims, of which roughly a
third were *enumerable* facts: a count stated in prose that no longer matched
the code, a CLI default that contradicted its own help string, a path that had
moved. Those are the ones a test can hold.

Scope is deliberately narrow. These assertions only cover facts that can be
derived from source, so they stay true without maintenance:

* counts stated in prose vs. the enum / factory list they describe
* CLI flag defaults stated in docs vs. argparse
* paths referenced in prose vs. the filesystem
* internal process vocabulary that must never reach a published page

They cannot catch a prose claim that is simply wrong about behaviour (an
inverted description of a state machine, say). The structural rule — docs/ is
canonical, README is a landing page — covers that class; this file covers the
mechanical one.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"

pytestmark = pytest.mark.unit


def _docs_pages() -> list[Path]:
    """Authored documentation pages, excluding the contributor runbook."""
    return sorted(p for p in DOCS.rglob("*.md") if p != DOCS / "fern" / "README.md")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _is_gitignored(path: str) -> bool:
    """Return whether a repo-relative path is an intentionally ignored artifact."""
    candidates = (path, f"{path}/")
    return any(
        subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", candidate],
            cwd=REPO,
            check=False,
        ).returncode
        == 0
        for candidate in candidates
    )


# --------------------------------------------------------------------------
# Counts stated in prose must match the source of truth they describe.
# --------------------------------------------------------------------------


def test_scoring_signal_count_matches_enum():
    """Docs say "six signals"; SuccessSignal is the authority.

    README and evaluation/README.md disagreed on five-vs-six for months, each
    correct when written.
    """
    from nemo_voice_agent.evaluation.scenarios.classes import SuccessSignal

    actual = len(list(SuccessSignal))
    word = {5: "five", 6: "six", 7: "seven"}[actual]

    page = DOCS / "evaluate" / "understand-scoring" / "scoring.md"
    text = _read(page).lower()
    wrong = {"five", "six", "seven"} - {word}
    for w in wrong:
        assert f"{w} orthogonal" not in text and f"{w} scoring signal" not in text, (
            f"{page.relative_to(REPO)} says '{w}' signals but SuccessSignal has {actual}"
        )


def test_rtvi_action_count_matches_factories():
    """Docs enumerate the RTVI action factories; count them from the module."""
    src = _read(REPO / "nemo_voice_agent" / "pipecat" / "processors" / "frameworks" / "rtvi_actions.py")
    factories = re.findall(r"^def (create_\w+_action)\(", src, re.M)
    actual = len(factories)
    word = {5: "five", 6: "six", 7: "seven"}[actual]

    for rel in ("extend/rtvi-actions.md", "reference/rtvi-messages.md"):
        text = _read(DOCS / rel).lower()
        for w in {"five", "six", "seven"} - {word}:
            assert f"{w} rtvi" not in text and f"{w} required action" not in text, (
                f"docs/{rel} says '{w}' actions but rtvi_actions.py defines {actual}: {factories}"
            )


# --------------------------------------------------------------------------
# CLI defaults stated in docs must match argparse.
# --------------------------------------------------------------------------


def _argparse_defaults() -> dict[str, str]:
    """Extract `--flag` -> default literal from run_evaluation.py without importing it."""
    src = _read(REPO / "evaluation" / "run_evaluation.py")
    out: dict[str, str] = {}
    for block in re.findall(r"parser\.add_argument\((.*?)\n    \)", src, re.S):
        name = re.search(r'"(--[\w-]+)"', block)
        default = re.search(r"default=([^,\n]+)", block)
        if name and default:
            out[name.group(1)] = default.group(1).strip()
    return out


def test_eval_cli_reference_documents_true_defaults():
    """docs/reference/evaluation/eval-cli.md must not contradict argparse on defaults.

    `--min-agent-turns` shipped documented as "0 (disabled)" while defaulting to
    3, silently reshaping every aggregate rate.
    """
    defaults = _argparse_defaults()
    assert defaults, "failed to parse any argparse defaults — the extractor needs updating"

    page = _read(DOCS / "reference" / "evaluation" / "eval-cli.md")

    # The page documents two tools whose --min-agent-turns defaults legitimately
    # differ (run_evaluation.py 3, check_resume.py 0), so scope to the runner's
    # section or the check_resume table produces a false positive.
    runner_section = page.split("## `check_resume.py`")[0]
    assert "`run_evaluation.py`" in runner_section, (
        "eval-cli.md structure changed — this test scopes assertions by the check_resume.py heading"
    )

    assert defaults["--min-agent-turns"] == "3", (
        f"--min-agent-turns default changed to {defaults['--min-agent-turns']}; update the docs and this test"
    )
    assert re.search(r"`--min-agent-turns[^`]*`\s*\|\s*`3`", runner_section), (
        "docs/reference/evaluation/eval-cli.md does not document the runner's --min-agent-turns default of 3"
    )
    assert not re.search(r"`--min-agent-turns[^`]*`\s*\|\s*`0`", runner_section), (
        "docs/reference/evaluation/eval-cli.md claims the runner's --min-agent-turns defaults to 0"
    )

    assert defaults["--duration"] == "None", (
        f"--duration default changed to {defaults['--duration']}; update the docs and this test"
    )
    assert not re.search(r"`--duration[^`]*`\s*\|\s*`?(120|300)`?", runner_section), (
        "docs/reference/evaluation/eval-cli.md states a numeric --duration default; it is None "
        "(falls back to each scenario's max_duration)"
    )


# --------------------------------------------------------------------------
# Paths referenced in prose must exist.
# --------------------------------------------------------------------------


def test_referenced_repo_paths_exist():
    """Backticked repo-relative paths in docs must resolve.

    Catches the class where a module moves and prose keeps the old location —
    e.g. fixtures cited at `evaluation/data/` after they moved into the package.
    Gitignored generated and runtime artifacts are valid documentation targets
    even though they are absent from a clean checkout.
    """
    pattern = re.compile(r"`((?:nemo_voice_agent|examples|evaluation|scripts|tests|docker|docs)/[\w./-]*[\w/])`")
    missing: list[str] = []
    for page in _docs_pages():
        for match in set(pattern.findall(_read(page))):
            target = match.rstrip("/")
            if "*" in target or target.endswith((".md.", ".")):
                continue
            if not (REPO / target).exists() and not _is_gitignored(target):
                missing.append(f"{page.relative_to(REPO)} -> {target}")
    assert not missing, "docs reference paths that do not exist:\n  " + "\n  ".join(sorted(missing))


def test_eval_fixture_root_documented_correctly():
    """Fixtures are packaged inside the library, not at the repo root."""
    assert not (REPO / "evaluation" / "data").exists(), (
        "evaluation/data/ exists again — the docs describe the packaged location; reconcile them"
    )
    text = _read(DOCS / "evaluate" / "domain-guides" / "data-provenance.md")
    assert "nemo_voice_agent/evaluation/data" in text, (
        "docs/evaluate/domain-guides/data-provenance.md must name the packaged fixture root"
    )


# --------------------------------------------------------------------------
# Hygiene: nothing internal, agent-facing, or misnamed reaches a published page.
# --------------------------------------------------------------------------


def test_no_internal_milestone_markers_in_docs():
    """`M4`, `M5a`, `TBD in M5` are internal process vocabulary."""
    offenders = [
        str(p.relative_to(REPO))
        for p in _docs_pages()
        if re.search(r"\bM[1-9][a-e]?\b(?!\w)", _read(p)) or "TBD in M" in _read(p)
    ]
    assert not offenders, f"internal milestone markers found in published docs: {offenders}"


def test_docs_do_not_reference_agent_tooling():
    """Published pages must not point readers at agent harness config."""
    offenders = [
        str(p.relative_to(REPO))
        for p in _docs_pages()
        if re.search(r"\.claude/|\.cursor/|\.codex/|\bCLAUDE\.md\b|\bAGENTS\.md\b", _read(p))
    ]
    assert not offenders, f"docs reference agent tooling, which is not user documentation: {offenders}"


def test_docs_do_not_cite_files_inside_the_gitignored_scratch_dir():
    """Naming `nemo_experiments/` as a gitignored artifact is fine.

    Citing a *file inside* it is not: that content is not in the repo, so the
    reference is unresolvable for every reader.
    """
    offenders = [str(p.relative_to(REPO)) for p in _docs_pages() if re.search(r"nemo_experiments/\w", _read(p))]
    assert not offenders, f"docs cite files inside the gitignored scratch directory: {offenders}"


def test_every_docs_page_has_license_header():
    offenders = [str(p.relative_to(REPO)) for p in _docs_pages() if not _read(p).startswith("{/*")]
    assert not offenders, f"docs pages missing the SPDX header block: {offenders}"


def test_docs_pages_are_mdx_safe():
    """Fern renders .md through MDX: bare braces and angle brackets break the build.

    CI greps only for non-self-closing <img>, so this is the broader guard.
    """
    offenders: list[str] = []
    for page in _docs_pages():
        raw = _read(page)
        body = raw.split("*/}", 1)[1] if "*/}" in raw else raw
        body = re.sub(r"```.*?```", "", body, flags=re.S)
        body = re.sub(r"`[^`\n]*`", "", body)
        if re.search(r"[{}]", body) or re.search(r"<[a-zA-Z/!]", body):
            offenders.append(str(page.relative_to(REPO)))
    assert not offenders, f"pages contain MDX-unsafe characters outside code fences: {offenders}"


# --------------------------------------------------------------------------
# Navigation is generated; both files must cover every page.
# --------------------------------------------------------------------------


def test_nav_manifest_covers_every_page_and_only_real_pages():
    """docs/fern/nav.json is the single source for both nav files."""
    import json

    nav = json.loads(_read(DOCS / "fern" / "nav.json"))
    declared: set[str] = set()
    if root := nav.get("root"):
        declared.add(root["path"])

    def collect(section: dict) -> None:
        declared.update(p["path"] for p in section.get("pages", []))
        for subsection in section.get("subsections", []):
            collect(subsection)

    for section in nav["sections"]:
        collect(section)

    on_disk = {str(p.relative_to(DOCS)) for p in _docs_pages()}

    assert not (declared - on_disk), f"nav.json declares pages that do not exist: {sorted(declared - on_disk)}"
    assert not (on_disk - declared), (
        f"pages exist but are absent from nav.json, so they are unreachable: {sorted(on_disk - declared)}"
    )

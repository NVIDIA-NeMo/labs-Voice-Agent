{/*
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/}

# Contributing

Set up a development checkout of NeMo Labs Voice Agent, satisfy the automated gates, and prepare a pull
request for merge. The authoritative source is `CONTRIBUTING.md` at the repository root. This page summarizes
that file and links to related documentation.

## Development Environment

The development environment has the following requirements:

| Requirement | Notes |
| --- | --- |
| Python 3.12 or 3.13 | `requires-python = ">=3.12,<3.14"` in `pyproject.toml` |
| [uv](https://github.com/astral-sh/uv) | Manages the virtual environment, lockfile, and its own managed Python (`python-preference = "only-managed"`). |
| Git | Sign-off is required on every commit, as described in the sign-off section. |
| Linux and `apt` | `install.sh` uses `apt-get`. Install the required operating system packages manually on other platforms. |

Clone and bootstrap:

```bash
git clone https://github.com/NVIDIA-NeMo/Voice-Agent.git
cd Voice-Agent
bash install.sh
```

`install.sh` does more than `uv sync`. It installs `npm`, `nodejs`, `build-essential`, and `python3-dev`.
The `cdifflib` transitive dependency is source-distribution-only and compiles from source. The script also
installs `uv` if needed, runs `uv sync`, and prefetches two Natural Language Toolkit (NLTK) corpora for the
Apache-2.0 grapheme-to-phoneme (G2P) fallback. Prefer it to a bare `uv sync` on a new system.

Then add the test tools and Git hooks:

```bash
uv sync --group test        # pytest, pytest-timeout, coverage
uv run pre-commit install
```

Run subsequent commands through `uv run`, or activate the virtual environment for the current shell session
with `source .venv/bin/activate`. Review these constraints:

- **Do not run `uv sync` inside an active non-`base` Conda environment.** `install.sh` exits early when
  `CONDA_DEFAULT_ENV` is set to anything other than `base`, because Conda's `gcc` combined with system Python
  headers breaks C extensions. Run `conda deactivate` first.
- **The default install pulls CUDA 13.0 wheels** (`torch-backend = "cu130"` under `[tool.uv]`). Edit that key
  in `pyproject.toml` before syncing if you need `cu128`, `cu124`, or CPU-only. Refer to
  [Installation](../../get-started/installation.md).

## Code Style

**Ruff is the only formatter and linter in this repository.** `black` and `isort` are not installed and are
not dependencies—do not add them or reformat files with them. Configuration lives in `ruff.toml`:

| Setting | Value |
| --- | --- |
| `line-length` | 119 |
| `target-version` | `py312` |
| `format.quote-style` | `double`. Strings *are* normalized to double quotes. |
| lint `select` | `F401`, `F541`, `F821`, `F841`, `E741`, and `I` (isort) |
| lint `ignore` | `E501`. The formatter owns line length. |
| `lint.isort.known-first-party` | `nemo_voice_agent` |
| `lint.isort.lines-after-imports` | 2 |

Run it on the paths you touched:

```bash
uv run ruff format <path>          # format
uv run ruff check --fix <path>     # lint + import sorting
```

Or run the full hook set the way CI does:

```bash
uv run pre-commit run --all-files
```

`.pre-commit-config.yaml` configures `ruff` with `--fix`, a second `ruff` pass restricted to `--select I`, and
`ruff-format`, plus `check-yaml`, `check-case-conflict`, `detect-private-key`, `check-added-large-files`
with `--maxkb=1000`, and `requirements-txt-fixer`. The `lint-check` job in
`.github/workflows/cicd-main.yml` runs `pre-commit run --all-files` and blocks the rest of the continuous
integration (CI) pipeline on failure.

### License Headers

Every `.py` file needs an SPDX and Apache-2.0 header in its first lines. `.github/workflows/copyright-check.yml`
runs on every pull request and fails without it. Copy the header from any existing module. Markdown and
YAML are not checked, but repository convention is to include one anyway.

## Running Tests

Suites live in `tests/unit/` and `tests/functional/`. There are no test modules directly under `tests/`.

```bash
# Fast lane: unit tests, no GPU
uv run pytest tests/unit -m "not gpu"

# A single module
uv run pytest tests/unit/test_config_manager.py

# With coverage (pytest-cov is not a dependency; use the coverage CLI)
uv run coverage run --source=nemo_voice_agent -m pytest tests/unit
uv run coverage report -m
```

Markers declared in `pyproject.toml`: `unit`, `functional`, `gpu`, `slow`, `skipduringci`, `pleasefixme`.
CI's unit lane deselects `pleasefixme`. The functional lane runs on H100 runners. Refer to
[Testing](testing.md) for the full suite layout and how to write new tests.

## Commits

Use conventional-commit subjects, as documented in `CONTRIBUTING.md`:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.

### Sign-Off (DCO) — Required

Every commit must carry a `Signed-off-by` trailer certifying the Developer Certificate of Origin. Commits
without it are not accepted.

```bash
git commit -s -m "fix(tts): drop trailing silence on interruption"
```

That appends:

```
Signed-off-by: Your Name <your@email.com>
```

To fix an existing branch, amend a single commit with `git commit --amend -s` or rebase a series with
`git rebase --signoff main`. The full Developer Certificate of Origin (DCO) text is reproduced in
`CONTRIBUTING.md`. Sign-off is distinct from GitHub *signed* (GPG-verified) commits, which affect CI
triggering only.

## Pull Request Flow

To prepare and submit a pull request, complete the following steps:

1. Open an issue first for a significant change so contributors can discuss the approach.
2. Fork the repository and create a branch from `main` with a descriptive name (`feature/...`, `fix/...`).
3. Implement the change together with tests and documentation updates.
4. Run the gates locally:

   ```bash
   uv run pre-commit run --all-files
   uv run pytest tests/unit -m "not gpu"
   ```

5. Push to your fork and open a pull request against `main` that describes what changed and why.
6. Address review feedback, and squash the history before merge.

### Triggering CI

The pipeline runs on pushes to `main`, to `deploy-release/*`, and to the bot-mirrored
`pull-request/<number>` branches. If your GitHub account uses signed (GPG-verified) commits, CI starts
automatically on each push. Otherwise, comment on the pull request with the SHA of the commit you want tested:

```
/ok to test a1b2c3d4e5f6
```

Repeat the comment for each new commit. Get the SHA from `git log --oneline -1`.

Documentation-only pull requests take a different path: `fern-docs-ci.yml` and
`fern-docs-preview-build.yml` have a `docs/**` path filter and gate on MDX safety, `fern check`, and offline
link checking. Refer to
[Building the Docs](docs-site.md) before editing anything under `docs/`.

## Where Changes Usually Go

Use this table to find the documentation and implementation guidance for a change:

| Change | Start Here |
| --- | --- |
| New or swapped model, new pipeline stage | [The Builder API](../../build-voice-agents/extend/pipelines/builders.md) |
| New config key or default | [Configuration Model](../../build-voice-agents/configure/index.md) |
| New agent-callable tool | [Writing Your Own Tools](../../build-voice-agents/tools/custom-tools.md) |
| New evaluation scenario or domain | [Authoring Scenarios](../../evaluate/create-evaluations/authoring-scenarios.md) |
| Re-importing upstream benchmark fixtures | [Regenerating Benchmark Data](data-import.md) |
| Reporting a vulnerability | [Security Policy](security.md) |

Local artifacts are gitignored and must not be committed: `.venv/`, `nemo_voice_agent.egg-info/`,
`nemo_experiments/`, `eval_results/`, and `*.log` files.

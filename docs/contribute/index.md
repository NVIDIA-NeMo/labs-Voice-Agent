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

How to set up a development checkout of NeMo Labs Voice Agent, satisfy the automated gates, and get a pull
request merged. The authoritative source is `CONTRIBUTING.md` at the repo root; this page mirrors it and adds
pointers into the rest of the docs.

## Development environment

Requirements:

| Requirement | Notes |
| --- | --- |
| Python 3.12 or 3.13 | `requires-python = ">=3.12,<3.14"` in `pyproject.toml` |
| [uv](https://github.com/astral-sh/uv) | Manages the venv, the lockfile, and its own managed Python (`python-preference = "only-managed"`) |
| Git | Sign-off is required on every commit (see below) |
| Linux + apt | `install.sh` uses `apt-get`; other platforms need the OS packages installed by hand |

Clone and bootstrap:

```bash
git clone https://github.com/NVIDIA-NeMo/Voice-Agent.git
cd Voice-Agent
bash install.sh
```

`install.sh` does more than `uv sync`: it installs `npm`, `nodejs`, `build-essential`, and `python3-dev` (the
`cdifflib` transitive dependency is sdist-only and compiles from source), installs `uv` if missing, runs
`uv sync`, and prefetches the two NLTK corpora the Apache-2.0 G2P fallback needs. Prefer it over a bare
`uv sync` on a fresh machine.

Then add the test tooling and the git hooks:

```bash
uv sync --group test        # pytest, pytest-timeout, coverage
uv run pre-commit install
```

Everything after this runs through `uv run`, or you can `source .venv/bin/activate` once. Two gotchas:

- **Do not run `uv sync` inside an active non-`base` conda env.** `install.sh` exits early when
  `CONDA_DEFAULT_ENV` is set to anything other than `base`, because conda's gcc combined with system Python
  headers breaks C extensions. Run `conda deactivate` first.
- **The default install pulls CUDA 13.0 wheels** (`torch-backend = "cu130"` under `[tool.uv]`). Edit that key
  in `pyproject.toml` before syncing if you need cu128, cu124, or CPU-only. See
  [Installation](../get-started/installation.md).

## Code style

**Ruff is the only formatter and linter in this repo.** black and isort are not installed and are not
dependencies — do not add them, and do not reformat files with them. Configuration lives in `ruff.toml`:

| Setting | Value |
| --- | --- |
| `line-length` | 119 |
| `target-version` | `py312` |
| `format.quote-style` | `double` — strings *are* normalized to double quotes |
| lint `select` | `F401`, `F541`, `F821`, `F841`, `E741`, and `I` (isort) |
| lint `ignore` | `E501` — the formatter owns line length |
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

`.pre-commit-config.yaml` wires `ruff` (with `--fix`), a second `ruff` pass restricted to `--select I`, and
`ruff-format`, plus `check-yaml`, `check-case-conflict`, `detect-private-key`, `check-added-large-files`
(`--maxkb=1000`), and `requirements-txt-fixer`. The `lint-check` job in `.github/workflows/cicd-main.yml`
runs `pre-commit run --all-files` and blocks the rest of the pipeline on failure.

### License headers

Every `.py` file needs an SPDX / Apache-2.0 header in its first lines. `.github/workflows/copyright-check.yml`
runs on every pull request and hard-fails without it. Copy the header from any existing module. Markdown and
YAML are not checked, but repo convention is to include one anyway.

## Running tests

Suites live in `tests/unit/` and `tests/functional/` — there are no test modules directly under `tests/`.

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
CI's unit lane deselects `pleasefixme`; the functional lane runs on H100 runners. See
[Testing](testing.md) for the full suite layout and how to write new tests.

## Commits

Use conventional-commit subjects, as documented in `CONTRIBUTING.md`:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.

### Sign-off (DCO) — required

Every commit must carry a `Signed-off-by` trailer certifying the Developer Certificate of Origin. Commits
without it are not accepted.

```bash
git commit -s -m "fix(tts): drop trailing silence on interruption"
```

That appends:

```
Signed-off-by: Your Name <your@email.com>
```

To fix an existing branch, amend or rebase with `git commit --amend -s` (single commit) or
`git rebase --signoff main` (a series). The full DCO text is reproduced in `CONTRIBUTING.md`; sign-off is
distinct from GitHub *signed* (GPG-verified) commits, which affect CI triggering only.

## Pull request flow

1. Open an issue first for anything non-trivial, so the approach can be discussed.
2. Fork the repo, branch off `main` with a descriptive name (`feature/...`, `fix/...`).
3. Implement the change together with tests and doc updates.
4. Run the gates locally:

   ```bash
   uv run pre-commit run --all-files
   uv run pytest tests/unit -m "not gpu"
   ```

5. Push to your fork and open a PR against `main` with a description of what changed and why.
6. Address review feedback; squash the history before merge.

### Triggering CI

The pipeline runs on pushes to `main`, to `deploy-release/*`, and to the bot-mirrored
`pull-request/<number>` branches. If your GitHub account uses signed (GPG-verified) commits, CI starts
automatically on each push. Otherwise comment on the PR with the SHA of the commit you want tested:

```
/ok to test a1b2c3d4e5f6
```

Repeat the comment for each new commit. Get the SHA from `git log --oneline -1`.

Docs-only PRs take a different path: `fern-docs-ci.yml` and `fern-docs-preview-build.yml` have a `docs/**`
path filter and gate on MDX safety, `fern check`, and offline link checking. See
[Building the Docs](docs-site.md) before editing anything under `docs/`.

## Where changes usually go

| Change | Start here |
| --- | --- |
| New or swapped model, new pipeline stage | [The Builder API](../extend/builders.md) |
| New config key or default | [Configuration Model](../configure/index.md) |
| New agent-callable tool | [Writing Your Own Tools](../features/custom-tools.md) |
| New evaluation scenario or domain | [Authoring Scenarios](../evaluate/authoring-scenarios.md) |
| Re-importing upstream benchmark fixtures | [Regenerating Benchmark Data](data-import.md) |
| Reporting a vulnerability | [Security Policy](security.md) |

Local artifacts are gitignored and must not be committed: `.venv/`, `nemo_voice_agent.egg-info/`,
`nemo_experiments/`, `eval_results/`, and `*.log` files.

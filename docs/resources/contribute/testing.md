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

# Testing

NeMo Labs Voice Agent uses pytest for tests and the standalone `coverage` CLI for coverage.
`pytest-cov` is deliberately **not** a dependency — coverage is collected by wrapping pytest in
`coverage run`, so that CI can combine data files from several jobs.

## Test layout

There are exactly two suites, and **no test modules live directly under `tests/`**. Every test file
belongs to one of these directories:

| Path | Contents |
| --- | --- |
| `tests/unit/` | In-process tests: config loading, builders, evaluation registries, tool contracts, scenario data, bridge/runner logic. No GPU, no network, no model downloads. |
| `tests/functional/runtime/` | Runtime integrations that do real IO (RTVI action handlers, audio logger) but need no model weights. |
| `tests/functional/models/` | Tests that load real speech/LLM models from a warm cache. GPU required. |
| `tests/functional/vllm/` | The reasoning-budget logits processor running inside vLLM. GPU required. |
| `tests/unit/launch_scripts/`, `tests/functional/launch_scripts/h100/active/` | The exact shell scripts CI executes. Read these to reproduce a CI lane locally. |

`pyproject.toml` sets `testpaths = ["tests"]`, so a bare `pytest` collects both suites. It also sets
`addopts = "--verbose --strict-markers"` — an undeclared marker is a collection **error**, not a warning.

## Running tests

The everyday loop is the unit suite:

```bash
uv run pytest tests/unit -m "not gpu"
```

Narrow it while iterating:

```bash
uv run pytest tests/unit/test_config_manager.py           # one module
uv run pytest tests/unit -k "telecom and sync"            # by name substring
uv run pytest tests/unit -x -q                            # stop at first failure, quiet
```

Functional tests that need no GPU:

```bash
uv run pytest tests/functional/runtime -m "functional and not gpu"
```

`pytest`, `coverage`, and `pytest-timeout` come from the `test` dependency group, which a plain
`uv sync` does not install. If `pytest` is missing, sync the way the CI image does
(see `docker/Dockerfile.ci`):

```bash
uv sync --all-extras --group test
```

## Markers

Six markers are declared under `[tool.pytest.ini_options]` in `pyproject.toml`:

| Marker | Meaning |
| --- | --- |
| `unit` | In-process test that does not require GPU hardware, external services, or live model serving. |
| `functional` | Exercises runtime integrations, model serving, GPU execution, or end-to-end flows. |
| `gpu` | Requires GPU hardware. |
| `slow` | Too slow for the default unit lane. |
| `skipduringci` | Intentionally skipped in CI. |
| `pleasefixme` | Known-broken; excluded from every CI lane until fixed. |

Two things to know before you rely on marker selection:

- **The unit lane is selected by path, not by marker.** Only a few modules under `tests/unit/` carry
  `pytest.mark.unit`, so `-m unit` collects a small subset. Point pytest at `tests/unit` instead.
- **The functional lanes are selected by marker.** Every module under `tests/functional/` sets a
  module-level `pytestmark`, so `functional` (and `gpu` where applicable) is reliable there.

There are no `conftest.py` files and no automatic GPU detection, so a `gpu`-marked test run on a
machine without a GPU fails rather than skips. Always pass `-m "not gpu"` on a CPU box.

## Selecting with `-m`

`-m` takes a boolean expression over marker names:

```bash
uv run pytest tests -m "not gpu"                              # everything runnable on CPU
uv run pytest tests -m "functional and gpu"                   # GPU functional lanes only
uv run pytest tests -m "not slow and not pleasefixme"         # skip slow + known-broken
uv run pytest tests -m "not gpu and not skipduringci"         # what a CI-like CPU run covers
```

Combine `-m` with a path to intersect both filters — that is what the launch scripts do.

## Coverage

Wrap pytest in `coverage run`, exactly as `tests/unit/launch_scripts/Launch_Unit_Tests.sh` does:

```bash
uv run coverage run -a --data-file=.coverage --source=nemo_voice_agent -m pytest \
    -vs tests/unit -m "not pleasefixme"
uv run coverage report -i --precision=2
uv run coverage html          # browsable report in htmlcov/
```

`[tool.coverage.run]` in `pyproject.toml` controls what counts:

- `source = ["nemo_voice_agent"]` — only the library, not `examples/` or `evaluation/`.
- `concurrency = ["thread", "multiprocessing"]` — needed because the bridge and bot servers spawn both.
- `omit` drops `tests/*`, `.venv/*`, and the generated scenario shards
  (`nemo_voice_agent/evaluation/scenarios/data/*/group_*.py`) so machine-written scaffolding does not
  dilute the numbers.

To merge several runs, give each one its own data file and combine:

```bash
COVERAGE_FILE=.coverage.unit uv run coverage run --source=nemo_voice_agent -m pytest tests/unit
COVERAGE_FILE=.coverage.func uv run coverage run --source=nemo_voice_agent -m pytest tests/functional/runtime
uv run coverage combine --keep .coverage.unit .coverage.func
uv run coverage report -i
```

## What CI runs

`.github/workflows/cicd-main.yml` runs the launch scripts inside the CI image built from
`docker/Dockerfile.ci`:

| Job | Script | Path | Marker expression |
| --- | --- | --- | --- |
| `unit-tests` | `Launch_Unit_Tests` | `tests/unit` | `not pleasefixme` |
| `functional-tests-h100` | `L0_Functional_Model_Free_Runtime` | `tests/functional/runtime` | `functional and not gpu and not pleasefixme` |
| `functional-tests-h100` | `L0_Functional_VLLM_Reasoning_Budget` | `tests/functional/vllm` | `functional and gpu and not pleasefixme` |
| `functional-tests-h100` | `L1_Functional_Cached_Model_Runtime` | `tests/functional/models` | `functional and gpu and not pleasefixme` |

A separate `coverage` job combines the uploaded data files and enforces `--fail-under` per flag:
75 for the unit lane, 0 for the end-to-end lane, and 80 for the combined `all` flag. Adding library
code without unit tests is the usual way to trip the 75 gate.

## Adding a test

To add a test that matches the repository layout and CI contracts, complete the following steps:

1. Put the module in `tests/unit/` (default) or under the matching `tests/functional/<area>/`
   directory. Never at the top level of `tests/`.
2. Add the SPDX/Apache header — `copyright-check.yml` hard-fails on any `*.py` without one in its
   first 10 lines.
3. Mark functional modules with a module-level `pytestmark`, adding `pytest.mark.gpu` when the test
   needs hardware. Only use markers from the table above; `--strict-markers` rejects the rest.
4. Format and lint before committing:

   ```bash
   uv run ruff format tests/unit/test_my_thing.py
   uv run ruff check --fix tests/unit/test_my_thing.py
   ```

5. If the test belongs to a new functional lane, add a launch script under
   `tests/functional/launch_scripts/h100/active/` and register it in the `functional-tests-h100`
   matrix — CI does not auto-discover scripts.

Evaluation-harness behavior is covered by unit tests too (scenario registries, gold replay, DB hashing,
scoring aggregation); see [Evaluation](../../evaluate/index.md) for what those tests are asserting, and
[Contributing](index.md) for the overall PR workflow.

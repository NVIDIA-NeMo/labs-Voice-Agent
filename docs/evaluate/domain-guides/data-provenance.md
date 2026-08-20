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

# Fixture Data & Provenance

Every NeMo Labs Voice Agent evaluation domain is backed by fixture data — scenario databases, task
definitions, and agent policy prompts — copied from an upstream open-source benchmark. This page records
where that data came from, what licence it carries, where it lives on disk, and how the loader resolves it
at runtime.

The machine-readable counterpart lives in `nemo_voice_agent/evaluation/data/README.md` (per-file source
mapping) and `THIRD_PARTY_NOTICES.md` (reproduced licence texts). See also
[Third-party notices](../../resources/third-party-notices.md).

## Where fixtures live

Fixtures are packaged **inside the library**, not next to the eval scripts:

```
nemo_voice_agent/evaluation/data/
├── README.md            # per-file upstream source mapping
├── eva_airline/
├── tau2_airline/
├── tau2_retail/
└── tau2_telecom/
```

They ship in the wheel via `[tool.setuptools.package-data]` in `pyproject.toml`, which globs
`evaluation/data/**/*.json`, `**/*.jsonl`, and `**/*.md`. A `pip install` of the package therefore carries
the full eval surface; nothing is downloaded at run time.

## Resolution order and `EVAL_DATA_ROOT`

`get_eval_data_root()` in `nemo_voice_agent/evaluation/__init__.py` returns the fixture root. It is a
function, not a module constant, so an environment change after import still takes effect:

| Order | Source | Notes |
|---|---|---|
| 1 | `$EVAL_DATA_ROOT` | Used verbatim if set and non-empty. No existence check. |
| 2 | `nemo_voice_agent/evaluation/data/` | The packaged directory, used when it exists. This is the normal path. |
| 3 | `<repo>/evaluation/data/` | Legacy layout kept as a last-resort fallback; not present in this repo. |

Set `EVAL_DATA_ROOT` when you want to point the runner and the bots at a scratch copy of the fixtures
(for example, to test a re-import before committing it):

```bash
export EVAL_DATA_ROOT=/path/to/alternate/eval_data
```

Both the bridge process and the bot-server processes call `get_eval_data_root()` independently, so they may
resolve to different absolute roots. That is why fixture paths stored in `shared_state` are always
**relative** to the root — see the next section. Environment variables are catalogued in
[Environment variables](../../reference/runtime/environment.md).

## How fixtures reach the bots

Two seeding styles, both driven by `Scenario.setup_shared_state(state, side)`:

| Style | Used by | Mechanism |
|---|---|---|
| Inline DB | `eva_airline` | The bridge puts the whole scenario DB in `state["db"]`. Eva DBs are ~10-20 KB each, so they fit comfortably in a WebSocket frame. |
| Path reference | all `tau2_*` domains | The bridge puts a relative string in `state["db_path"]` (for example `tau2_airline/db.json`); the bot resolves it against its own `get_eval_data_root()`. |

Path seeding exists because Pipecat's default WebSocket frame cap is 1 MB: inlining a multi-megabyte tau2
DB closes the connection with code `1009` before the payload arrives.

The bot-side resolution happens in the `apply_initialization` RTVI client-message handler
(`create_apply_initialization_action` in `nemo_voice_agent/pipecat/processors/frameworks/rtvi_actions.py`).
It merges the `shared_state_init` JSON payload, then — only when `db_path` is present and `db` is not —
loads the artifact and stores it under `db`. The step is idempotent: an inline `db` short-circuits the load
and the redundant `db_path` key is dropped. A missing file raises `FileNotFoundError` with the currently
resolved root printed in the message.

## Sharded databases and `load_db_artifact`

`load_db_artifact(path)` (same module as `get_eval_data_root()`) accepts a path **without** the `.json`
suffix and probes two on-disk layouts:

1. `<path>.json` — a single file, parsed with `json.loads`.
2. `<path>/` — a directory; every top-level `*.json` file becomes one key in the returned dict, using the
   file stem as the key. Sorted, non-recursive.

Neither form present raises `FileNotFoundError`. The two layouts are equivalent as in-memory dicts, so DB
hashes and gold replays are identical either way — see [Scoring signals](../understand-scoring/scoring.md) for how the hash is
used.

`tau2_airline` is the only domain shipped sharded today: its upstream `db.json` is ~6.8 MB, over the 5 MB
per-file cap on the GitLab mirror, so it lives as `tau2_airline/db/flights.json`, `reservations.json`, and
`users.json` (~4.6 MB combined). Re-shard after a fresh upstream pull:

```bash
python scripts/prepare_tau2_data/shard_db.py nemo_voice_agent/evaluation/data/tau2_airline/db.json
```

The source file is deleted after sharding unless you pass `--keep-source`.

## Upstream sources

| Domain | Upstream project | Pinned version | Licence |
|---|---|---|---|
| `eva_airline` | [ServiceNow/eva](https://github.com/ServiceNow/eva) | `0.1.3` | MIT (Copyright (c) 2026 ServiceNow) |
| `tau2_airline` | [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) | tag `voice-user-sim-v1.0`, commit `17e07b1` | MIT (Copyright (c) 2025 Sierra Research) |
| `tau2_retail` | [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) | tag `voice-user-sim-v1.0`, commit `17e07b1` | MIT (Copyright (c) 2025 Sierra Research) |
| `tau2_telecom` | [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) | tag `voice-user-sim-v1.0`, commit `17e07b1` | MIT (Copyright (c) 2025 Sierra Research) |

`voice-user-sim-v1.0` is an annotated tag, so its object SHA is not a commit SHA. Dereference it with
`git rev-parse voice-user-sim-v1.0^{commit}` to get `17e07b1`.

Adapted Python modules (tools, param models, scenario bases) each carry an inline `# Adapted from <url>`
attribution at the top of the file. Data fixtures are verbatim copies except where noted below.

## Per-domain file inventory

### `eva_airline` — 50 scenarios

| File | Contents |
|---|---|
| `<eva_id>.json` (50 files) | Self-contained scenario world state — `reservations`, `journeys`, `disruptions`, and a `_current_date` key. Upstream `data/airline_scenarios/`. |
| `eva_airline_dataset.jsonl` (50 lines) | Per-scenario metadata: `user_goal`, `decision_tree`, `information_required`, `ground_truth.expected_scenario_db`. Upstream `data/airline_dataset.jsonl`. |
| `airline_agent.yaml` | Complete upstream agent configuration from `configs/agents/airline_agent.yaml`, with only trailing whitespace normalized. `get_agent_prompt()` uses its `role` and `instructions` verbatim before appending NeMo voice/runtime notes. |

Read once per process and indexed by scenario id. See [eva_airline](eva-airline.md).

### `tau2_airline` — 50 tasks

| File | Contents |
|---|---|
| `db/` (3 shards) | The shared airline DB, split per top-level table (`flights`, `reservations`, `users`). |
| `tasks.json` | 50 task definitions: `description`, `user_scenario`, `initial_state`, `evaluation_criteria`, `annotations`. |
| `tasks_voice.json` | Voice-eligible id list plus each task's `persona_name` under the `control` preset. |
| `split_tasks.json` | Split membership — `train` (30), `test` (20), `base` (50). `base` is the eval surface. |
| `policy.md` | Agent system prompt, copied verbatim from upstream. `get_agent_prompt()` returns it followed by an appended `## Additional Notes to Follow` section of voice-realization guidance. |

See [tau2_airline](tau2-airline.md).

### `tau2_retail` — 114 tasks

Same file shape as airline, with `db.json` unsharded (~2.8 MB) and `split_tasks.json` carrying `train` (74),
`test` (40), `base` (114). Of the 114 tasks, 73 are actions-only, 39 carry both `actions` and
`nl_assertions`, 1 is nl-assertion-only, and 1 (task 57) is chitchat with neither.
See [tau2_retail](tau2-retail.md).

### `tau2_telecom` — 114 tasks

The only domain with an import script, because upstream ships its DBs as TOML.

| File | Contents |
|---|---|
| `db.json` | Agent-facing DB (plans, devices, lines, customers, bills). Converted from upstream `db.toml`. |
| `user_db.json` | User-facing DB (mock phone state plus user surroundings). Converted from upstream `user_db.toml`. |
| `tasks.json` | Filtered at import time to the 114 base-split ids. |
| `tasks_voice.json` | Filtered at import time to the same 114 ids. |
| `split_tasks.json` | Verbatim — `base` (114), `small` (20), `train` (74), `test` (40), `full` (2285). |
| `main_policy.md`, `tech_support_manual.md`, `tech_support_workflow.md` | Policy sources; `Tau2TelecomBaseScenario.policy` concatenates `main_policy.md` with one `tech_support_*.md` variant, joined by a `---` rule. |

Re-import from a local tau2-bench checkout:

```bash
python scripts/prepare_tau2_data/prepare_telecom.py --source /path/to/tau2-bench
```

The script verifies the checkout's HEAD against its `PINNED_COMMIT` constant and warns (does not fail) on
mismatch, hard-fails if any base-split id is missing upstream, and round-trips the converted user DB through
the ported Pydantic model so Pydantic-default fields the raw TOML omits are materialized on disk.

Both the `tau2_telecom` and `tau2_telecom_workflow` scenario registrations read this same directory — the
workflow variant inherits `domain = "tau2_telecom"` and differs only in which `tech_support_*.md` file it
renders. See [tau2_telecom](tau2-telecom.md).

## Adding a new source

Import scripts and scaffold generators live under `scripts/`; the workflow is documented in
[Importing benchmark data](../../resources/contribute/data-import.md), and the scenario-class side in
[Authoring domains](../create-evaluations/authoring-domains.md). When you add a source, also:

1. Namespace its files under a new `nemo_voice_agent/evaluation/data/<domain>/` subdirectory so fixtures
   from different upstream libraries cannot collide.
2. Append a section to `nemo_voice_agent/evaluation/data/README.md` recording the upstream URL, pinned
   version or commit, licence, and per-file source mapping.
3. Append the licence text to `THIRD_PARTY_NOTICES.md` if the licence is not already reproduced there.
4. Add an inline `# Adapted from <url>` line to every ported Python module.

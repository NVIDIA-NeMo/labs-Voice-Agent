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

# Regenerating Benchmark Data

Benchmark fixtures and the scenario classes bound to them are **committed artifacts**, not runtime
downloads. Two script directories produce them, and both are developer-only — nothing in CI or in a normal
eval run invokes them.

| Directory | Purpose |
| --- | --- |
| `scripts/prepare_eva_data/` | Scaffold `eva_airline` scenario classes from the packaged eva dataset |
| `scripts/prepare_tau2_data/` | Import tau2-bench upstream data, shard oversized DBs, scaffold the three tau2 domains |

For where the resulting files live on disk and under which licence, see
[Fixture Data & Provenance](../../evaluate/domain-guides/data-provenance.md).

## Script inventory

Use this inventory to identify the script that owns each generated artifact:

| Script | Reads | Writes |
| --- | --- | --- |
| `prepare_tau2_data/prepare_telecom.py` | A local `tau2-bench` checkout | `nemo_voice_agent/evaluation/data/tau2_telecom/` |
| `prepare_tau2_data/shard_db.py` | A single `db.json` | A sibling `db/` directory of per-table shards |
| `prepare_tau2_data/generate_airline_scaffolds.py` | `data/tau2_airline/` | `scenarios/data/tau2_airline/group_Nx.py` |
| `prepare_tau2_data/generate_retail_scaffolds.py` | `data/tau2_retail/` | `scenarios/data/tau2_retail/group_Nx.py` |
| `prepare_tau2_data/generate_telecom_scaffolds.py` | `data/tau2_telecom/` | `scenarios/data/tau2_telecom/group_Nx.py` |
| `prepare_eva_data/generate_airline_scaffolds.py` | `data/eva_airline/eva_airline_dataset.jsonl` | stdout |

Paths in the Reads/Writes columns are relative to `nemo_voice_agent/evaluation/`. Only `prepare_telecom.py`
needs an upstream checkout; every other script reads data that is already committed to this repo.

## Importing tau2 telecom data

`prepare_telecom.py` is the only import script. Airline and retail ship their DBs as `.json` upstream and
were copied in directly when those domains were first ported; telecom ships `db.toml` / `user_db.toml`, so a
conversion step is needed to keep the bot-side loader on a single `json.load` path.

```bash
uv run python scripts/prepare_tau2_data/prepare_telecom.py --source /path/to/tau2-bench
```

`--source` defaults to a developer-local absolute path, so pass it explicitly. `--dest` redirects the output
(useful for diffing a candidate import against the committed tree before overwriting it).

The script performs three transformations and copies the rest verbatim:

| Step | Files | Behavior |
| --- | --- | --- |
| Convert | `db.toml`, `user_db.toml` | Parsed with `tomllib`, validated through the `TelecomDB` / `TelecomUserDB` models in `nemo_voice_agent.evaluation.tools.tau2_telecom_params`, re-serialized as indent-2 JSON. The Pydantic round-trip materializes defaults the raw TOML omits, so predicates and init functions never need defensive lookups. |
| Filter | `tasks.json` | Keeps only entries whose `id` appears in `split_tasks.json["base"]` (114 ids). Raises if a base id has no upstream task definition. |
| Filter | `tasks_voice.json` | Keeps only the base ids in its `configs` map. Raises if a base id has no upstream voice config. |
| Copy | `split_tasks.json`, `audio_difficulty.json`, `main_policy.md`, `tech_support_manual.md`, `tech_support_workflow.md`, `workflows/` | Byte-for-byte. Missing optional inputs produce a warning and are skipped, not a failure. |

The run is idempotent and prints a truncated SHA-256 per output file plus kept/dropped counts for the
filtered ones — diff those hashes against the previous run to see exactly what upstream changed.

Note that the repository does not currently track every file the script can emit: `audio_difficulty.json`
and `workflows/` are traceability-only and are not consumed by the eval pipeline, so they are not committed.
The agent policy is assembled from `main_policy.md` plus `tech_support_manual.md` or
`tech_support_workflow.md` — see [tau2-telecom](../../evaluate/domain-guides/tau2-telecom.md).

## Sharding an oversized database

The GitLab mirror enforces a 5 MB per-file cap, which the airline DB exceeds. `shard_db.py` splits a
top-level dict into one file per key:

```bash
uv run python scripts/prepare_tau2_data/shard_db.py \
    nemo_voice_agent/evaluation/data/tau2_airline/db.json
```

This writes `db/flights.json`, `db/users.json`, `db/reservations.json` and then **deletes the source
`db.json`**; pass `--keep-source` to retain it. `load_db_artifact` in `nemo_voice_agent/evaluation/__init__.py`
probes `<path>.json` first and falls back to `<path>/`, so both layouts reassemble into an identical
in-memory dict and DB hashes are unaffected. Re-shard after any fresh upstream pull that restores a
single-file `db.json`.

## Scaffolding tau2 scenario classes

The three tau2 scaffolders take no arguments. Each reads the already-imported data, chunks the base-split ids
10 per group, **overwrites** every `group_Nx.py` in the target package, and rewrites that package's
`__init__.py` with side-imports so `@register_eval_scenario` fires on import.

```bash
uv run python scripts/prepare_tau2_data/generate_airline_scaffolds.py   # 50 tasks  → 5 groups
uv run python scripts/prepare_tau2_data/generate_retail_scaffolds.py    # 114 tasks → 12 groups
uv run python scripts/prepare_tau2_data/generate_telecom_scaffolds.py   # 114 tasks → 12 groups
```

Emitted classes are deliberately minimal — `name` plus `tau2_id`, with everything else derived by the domain
base class from the upstream task. Airline and retail have integer ids and get decade-based filenames
(`group_0x.py` holds ids 0-9). Telecom ids are descriptive strings, so filenames use a sequential chunk
index and the class/scenario names are parsed out of the id:

```text
tau2_id  [mms_issue]airplane_mode_on|data_mode_off[PERSONA:Hard]
name     tau2_telecom__mms_issue__airplane_mode_on__data_mode_off__hard
class    Tau2TelecomMmsIssueAirplaneModeOnDataModeOffHard
```

A `PERSONA:None` suffix is dropped from both the name and the class. The telecom generator emits **two**
classes per task — one on `Tau2TelecomBaseScenario` (`policy_variant="manual"`) and one on
`Tau2TelecomWorkflowBaseScenario` (`policy_variant="workflow"`) — so 114 tasks produce 228 classes across
the `tau2_telecom` and `tau2_telecom_workflow` registrations. It also checks for class-name and
scenario-name collisions across both variants before writing anything, and aborts rather than emit a
silently shadowed scenario.

## Scaffolding eva_airline scenarios

There is no eva import script; the eva fixtures were copied verbatim from upstream. The generator reads the
packaged `eva_airline_dataset.jsonl` and **streams to stdout** instead of overwriting group files, because
eva scenarios carry curated prose that the dataset alone cannot express.

```bash
# One major group, appended for hand review before committing.
uv run python scripts/prepare_eva_data/generate_airline_scaffolds.py --major 1 \
    >> nemo_voice_agent/evaluation/scenarios/data/eva_airline/group_1x.py

# Every major group in one pass, with splittable section markers on stdout.
uv run python scripts/prepare_eva_data/generate_airline_scaffolds.py > /tmp/all_scaffolds.py
```

`--dataset` points at an alternate JSONL. The per-group manifest goes to stderr so it never pollutes the
generated Python. Five hand-authored seed scenarios (`1.1.2`, `2.1.1`, `3.1.3`, `5.1.1`, `7.2.1`) are skipped
— they remain the reference for prose style. The generator applies the voice-readability rule from
`nemo_voice_agent/utils/voice_prompts.py`, spelling confirmation numbers and airport codes out
character-by-character, and folds each entry's criteria and edge cases into the scenario guidelines.

Treat the output as a starting point: review the prose and prune bullets before committing.

## Bumping the upstream pin

To update the tau2-bench source revision and regenerate affected artifacts, complete the following steps:

1. Check out `sierra-research/tau2-bench` at the new tag or commit.
2. Update `PINNED_COMMIT` in `scripts/prepare_tau2_data/prepare_telecom.py`. The check is intentionally
   soft — a mismatch warns and continues, so you can test against an unpinned checkout first.
3. Re-run `prepare_telecom.py`, then re-shard any DB that grew past the file cap.
4. Re-run every tau2 scaffolder whose `tasks.json` or `split_tasks.json` changed.
5. Update the source and version fields in `nemo_voice_agent/evaluation/data/README.md`.
6. Format and lint the generated Python, then run the guard tests.

```bash
uv run ruff format nemo_voice_agent/evaluation/scenarios/data
uv run ruff check --fix nemo_voice_agent/evaluation/scenarios/data
uv run pytest tests/unit -m "not gpu"
```

`tests/unit/test_tau2_telecom_scenarios.py` asserts 114 manual plus 114 workflow registrations,
`tests/unit/test_tau2_retail_scenarios.py` asserts 114 retail registrations, and the eva suites cover the
airline tools and scenario metadata — a scaffolding regression shows up there first. Generated modules carry
the SPDX header their templates emit, which is what keeps the `copyright-check.yml` gate green.

## Related pages

Use these pages for provenance requirements, authoring guidance, and validation details:

- [Fixture Data & Provenance](../../evaluate/domain-guides/data-provenance.md) — on-disk layout, licences, `EVAL_DATA_ROOT`
- [Authoring Scenarios](../../evaluate/create-evaluations/authoring-scenarios.md) — writing a scenario class by hand
- [Adding a Domain](../../evaluate/create-evaluations/authoring-domains.md) — wiring a new benchmark end to end
- [Testing](testing.md) — markers, suites, and what CI runs

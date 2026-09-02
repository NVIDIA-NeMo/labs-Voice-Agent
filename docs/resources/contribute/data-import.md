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

Benchmark fixtures and their scenario classes are **committed artifacts**, not runtime downloads. Two
developer-only script directories produce them. Continuous integration (CI) and normal evaluation runs do
not invoke these scripts. The directories have the following responsibilities:

| Directory | Purpose |
| --- | --- |
| `scripts/prepare_eva_data/` | Scaffold `eva_airline` scenario classes from the packaged eva dataset |
| `scripts/prepare_tau2_data/` | Import tau2-bench upstream data, shard oversized databases, and scaffold the three tau2 domains |

For the resulting file locations and applicable licenses, refer to
[Fixture Data & Provenance](../../evaluate/domain-guides/data-provenance.md).

## Script Inventory

Use this inventory to identify the script that owns each generated artifact:

| Script | Reads | Writes |
| --- | --- | --- |
| `prepare_tau2_data/prepare_telecom.py` | A local `tau2-bench` checkout | `nemo_voice_agent/evaluation/data/tau2_telecom/` |
| `prepare_tau2_data/shard_db.py` | A single `db.json` | A sibling `db/` directory of per-table shards |
| `prepare_tau2_data/generate_airline_scaffolds.py` | `data/tau2_airline/` | `scenarios/data/tau2_airline/group_Nx.py` |
| `prepare_tau2_data/generate_retail_scaffolds.py` | `data/tau2_retail/` | `scenarios/data/tau2_retail/group_Nx.py` |
| `prepare_tau2_data/generate_telecom_scaffolds.py` | `data/tau2_telecom/` | `scenarios/data/tau2_telecom/group_Nx.py` |
| `prepare_eva_data/generate_airline_scaffolds.py` | `data/eva_airline/eva_airline_dataset.jsonl` | stdout |

Paths in the Reads and Writes columns are relative to `nemo_voice_agent/evaluation/`. Only
`prepare_telecom.py` needs an upstream checkout. Every other script reads data that is already committed to
this repository.

No script needs dependencies beyond the project virtual environment. They use the standard library—`tomllib`
requires Python 3.11 or later, and this project targets Python 3.12 to 3.13—plus an importable
`nemo_voice_agent`, which `prepare_telecom.py` loads lazily for its Pydantic round trip and fails fast
without. Run every script through `uv run` so the packaged library resolves.

## Importing tau2 Telecom Data

`prepare_telecom.py` is the only import script. Airline and retail provide their databases as upstream
`.json` files, which were copied directly when those domains were first ported. Telecom provides `db.toml`
and `user_db.toml`, so a conversion step keeps the bot-side loader on a single `json.load` path. Converting
at copy time also makes the result diff cleanly against future upstream re-syncs, because TOML key reordering
is invisible after the canonicalization pass.

```bash
uv run python scripts/prepare_tau2_data/prepare_telecom.py --source /path/to/tau2-bench
```

`--source` defaults to a developer-local absolute path, so pass it explicitly. Use `--dest` to redirect the
output and compare a candidate import with the committed tree before overwriting it.

The script performs three transformations and copies the rest verbatim:

| Step | Files | Behavior |
| --- | --- | --- |
| Convert | `db.toml`, `user_db.toml` | Parse with `tomllib`, validate with the `TelecomDB` and `TelecomUserDB` models in `nemo_voice_agent.evaluation.tools.tau2_telecom_params`, and serialize as indent-2 JSON. The Pydantic round trip materializes defaults that the raw TOML omits. Predicates and initialization functions do not need defensive lookups. |
| Filter | `tasks.json` | Keep only entries whose `id` appears in `split_tasks.json["base"]` (114 IDs). Raise an error if a base ID has no upstream task definition. This drops the 2,171 non-base definitions, taking roughly 14 MB upstream down to roughly 660 KB on disk—4.7 percent kept. |
| Filter | `tasks_voice.json` | Keep only the base IDs in its `configs` map. Raise an error if a base ID has no upstream voice configuration. The `small`, `train`, `test`, and `full` split entries are stripped, taking roughly 62 MB upstream down to roughly 3 MB. Re-imports preserve the filter. |
| Copy | `split_tasks.json`, `audio_difficulty.json`, `main_policy.md`, `tech_support_manual.md`, `tech_support_workflow.md`, `workflows/` | Copy byte-for-byte. Warn about and skip missing optional inputs instead of failing. |

The run is idempotent. It prints a truncated SHA-256 hash for each output file and kept or dropped counts for
filtered files. Compare these hashes with the previous run to identify upstream changes. Together with the
file sizes above, the hashes and counts are the fastest check that a candidate re-import matches the
committed tree.

The repository does not currently track every file that the script can emit. `audio_difficulty.json`
and `workflows/` are traceability-only and are not consumed by the evaluation pipeline, so they are not committed.
The agent policy is assembled from `main_policy.md` plus `tech_support_manual.md` or
`tech_support_workflow.md`. For details, refer to
[tau2 Telecom](../../evaluate/domain-guides/tau2-telecom.md).

## Sharding an Oversized Database

The GitLab mirror enforces a 5 MB per-file cap, which the airline database exceeds. `shard_db.py` splits a
top-level dictionary into one file per key:

```bash
uv run python scripts/prepare_tau2_data/shard_db.py \
    nemo_voice_agent/evaluation/data/tau2_airline/db.json
```

This writes `db/flights.json`, `db/users.json`, and `db/reservations.json`, and then **deletes the source
`db.json`**. Pass `--keep-source` to retain it. `load_db_artifact` in
`nemo_voice_agent/evaluation/__init__.py` probes `<path>.json` first and falls back to `<path>/`. Both layouts
reassemble into an identical in-memory dictionary, and database hashes are unaffected. Shard the database
again after any upstream pull that restores a single-file `db.json`.

## Scaffolding tau2 Scenario Classes

The three tau2 scaffolders take no arguments. Each reads the imported data and divides the base-split IDs
into groups of 10. It then **overwrites** every `group_Nx.py` in the target package and rewrites that package's
`__init__.py` with side-effect imports so `@register_eval_scenario` fires on import.

```bash
uv run python scripts/prepare_tau2_data/generate_airline_scaffolds.py   # 50 tasks  → 5 groups
uv run python scripts/prepare_tau2_data/generate_retail_scaffolds.py    # 114 tasks → 12 groups
uv run python scripts/prepare_tau2_data/generate_telecom_scaffolds.py   # 114 tasks → 12 groups
```

Generated classes contain only `name` and `tau2_id`. The domain base class derives everything else from the
upstream task. Airline and retail use integer IDs and decade-based filenames (`group_0x.py` holds IDs 0–9).
Telecom IDs are descriptive strings. Its filenames use a sequential chunk index, and the generator parses
the class and scenario names from the ID:

```text
tau2_id  [mms_issue]airplane_mode_on|data_mode_off[PERSONA:Hard]
name     tau2_telecom__mms_issue__airplane_mode_on__data_mode_off__hard
class    Tau2TelecomMmsIssueAirplaneModeOnDataModeOffHard
```

A `PERSONA:None` suffix is dropped from both the name and the class. The telecom generator emits **two**
classes per task—one on `Tau2TelecomBaseScenario` (`policy_variant="manual"`) and one on
`Tau2TelecomWorkflowBaseScenario` (`policy_variant="workflow"`)—so 114 tasks produce 228 classes across
the `tau2_telecom` and `tau2_telecom_workflow` registrations. It also checks for class-name and
scenario-name collisions across both variants before writing anything. It aborts rather than emitting a
silently shadowed scenario.

Re-run a scaffolder whenever the data importer changes its domain's `tasks.json` or `split_tasks.json`, and
also after you edit the scaffold template or the class-naming logic, which changes the generated output with
no data change at all.

## Scaffolding eva_airline Scenarios

There is no eva import script. The eva fixtures were copied verbatim from upstream. The generator reads the
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
generated Python. In a combined dump, each major group is introduced by a
`# === eva_airline major group N.x (K scenarios) ===` line between two comment rules, so an operator can split
the stream into group files mechanically. Names come from the eva ID: `2.1.6` yields
`name = "eva_airline__2_1_6"` and `class EvaAirline216`. Five hand-authored seed scenarios (`1.1.2`, `2.1.1`,
`3.1.3`, `5.1.1`, `7.2.1`) are skipped—they remain the reference for prose style.

The generator applies the voice-readability rule from `nemo_voice_agent/utils/voice_prompts.py`, spelling
confirmation numbers and airport codes out character-by-character, and folds each entry's
`must_have_criteria`, `negotiation_behavior`, `edge_cases`, `failure_condition`, and `escalation_behavior`—all
read from `user_goal.decision_tree`—into the scenario guidelines. Negotiation behavior is truncated to the
first three items and edge cases to the first five.

Re-run the generator after updating `eva_airline_dataset.jsonl`, such as when adding an edge-case entry, and
after tweaking the scaffold template or the prose conventions it emits. Treat the output as a starting point:
review the prose and prune the negotiation and edge-case bullets before committing.

## Bumping the Upstream Pin

To update the tau2-bench source revision and regenerate affected artifacts, complete the following steps:

1. Check out `sierra-research/tau2-bench` at the new tag or commit.
2. Update `PINNED_COMMIT` in `scripts/prepare_tau2_data/prepare_telecom.py`. The check is intentionally
   nonblocking—a mismatch produces a warning and continues, so you can test an unpinned checkout first.
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
airline tools and scenario metadata. A scaffolding regression shows up there first. Generated modules carry
the SPDX header their templates emit, which keeps the `copyright-check.yml` gate green.

## Related Pages

Use these pages for provenance requirements, authoring guidance, and validation details:

- [Fixture Data & Provenance](../../evaluate/domain-guides/data-provenance.md): On-disk layout, licenses, and `EVAL_DATA_ROOT`.
- [Authoring Scenarios](../../evaluate/create-evaluations/authoring-scenarios.md): How to write a scenario class manually.
- [Adding a Domain](../../evaluate/create-evaluations/authoring-domains.md): How to integrate a new benchmark end to end.
- [Testing](testing.md): Markers, suites, and CI behavior.

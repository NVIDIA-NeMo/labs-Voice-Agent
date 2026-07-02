# `prepare_tau2_data/`

One-shot developer-only scripts for two purposes:
1. **Import** tau2-bench upstream data into `nemo_voice_agent/evaluation/data/<tau2_domain>/`.
2. **Scaffold** scenario classes from the imported data into
   `nemo_voice_agent/evaluation/scenarios/data/<tau2_domain>/group_Nx.py`.

Both kinds of scripts run against a local checkout of
[`sierra-research/tau2-bench`](https://github.com/sierra-research/tau2-bench)
pinned at the `voice-user-sim-v1.0` tag (commit `17e07b1`).

## Scripts

### Data import (run against the upstream checkout)

| Script | Domain | What it does |
|---|---|---|
| `prepare_telecom.py` | `tau2_telecom` | Converts `db.toml` + `user_db.toml` → JSON. Copies tasks/policy/workflows verbatim. |

Airline + retail don't have an import script because their DBs ship as `.json` upstream — they were copied in directly when the domains were first ported.

### Scenario scaffolders (run against the imported data)

| Script | Domain | What it does |
|---|---|---|
| `generate_airline_scaffolds.py` | `tau2_airline` | Reads `nemo_voice_agent/evaluation/data/tau2_airline/tasks_voice.json` + `split_tasks.json[base]` and emits one `@register_eval_scenario class Tau2Airline<id>(Tau2AirlineBaseScenario)` per task across `group_{decade}x.py` modules. |
| `generate_retail_scaffolds.py` | `tau2_retail` | Same shape; emits `Tau2Retail<id>` subclasses across 12 group files (114 base-split tasks). |
| `generate_telecom_scaffolds.py` | `tau2_telecom` | Same shape but task ids are descriptive strings (`[mms_issue]airplane_mode_on\|data_mode_off[PERSONA:Hard]`), so class/scenario names are derived from the parts: `Tau2TelecomMmsIssueAirplaneModeOnDataModeOffHard` / `tau2_telecom__mms_issue__airplane_mode_on__data_mode_off__hard`. `PERSONA:None` is dropped from both name and class. 114 base-split tasks → 12 group files. |

All scaffolders **overwrite** existing `group_Nx.py` files and rewrite the package's `__init__.py` to side-import them. The output is committed alongside `base.py` — checked in, not regenerated at runtime.

## Why TOML → JSON conversion (only for telecom)

Upstream tau2-bench ships airline + retail DBs as `db.json` but the telecom domain ships `db.toml` and `user_db.toml`. Rather than add a TOML branch to the bot-side `update_system_prompt` loader (`rtvi_actions.create_update_system_prompt_action`), we normalize at copy time so the bot only ever loads `.json`. Side benefit: the converted JSON diffs cleanly against future upstream re-syncs (TOML reordering is invisible after the canonicalization pass).

## When to re-run

- **Data import** (e.g. `prepare_telecom.py`): after bumping the pinned upstream commit. Update `PINNED_COMMIT` in the script first and verify the new hashes are intentional.
- **Scaffolders** (`generate_*_scaffolds.py`): after re-running the data importer (when the underlying `tasks.json` / `split_tasks.json` changes), or after editing the scaffold template / class-naming logic. Routine eval runs do NOT touch the generators — the committed `group_Nx.py` files are the source of truth at runtime.
- Never in CI — these scripts are developer-only setup. Their output lives in `nemo_voice_agent/evaluation/data/<domain>/` (data) and `nemo_voice_agent/evaluation/scenarios/data/<domain>/group_*x.py` (scaffolds), both committed to the repo.

## Usage

```bash
# Data import
python scripts/prepare_tau2_data/prepare_telecom.py
python scripts/prepare_tau2_data/prepare_telecom.py --source /path/to/tau2-bench
python scripts/prepare_tau2_data/prepare_telecom.py --dest /tmp/test_telecom_import

# Scaffold generation (no flags — read from nemo_voice_agent/evaluation/data/<domain>/,
# write to nemo_voice_agent/evaluation/scenarios/data/<domain>/)
python scripts/prepare_tau2_data/generate_airline_scaffolds.py
python scripts/prepare_tau2_data/generate_retail_scaffolds.py
python scripts/prepare_tau2_data/generate_telecom_scaffolds.py
```

`prepare_telecom.py` verifies the upstream HEAD commit against `PINNED_COMMIT` and warns (does not fail) on mismatch — re-sync work intentionally relaxes this check so the operator can test against an unpinned upstream before committing the new pin. The scaffolders have no upstream-checkout dependency at all — they only read the already-imported data under `nemo_voice_agent/evaluation/data/`.

After running a scaffolder, the existing tests (e.g. `tests/test_tau2_telecom_scenarios.py:test_all_114_base_split_scenarios_registered`) will catch unexpected scenario-count or instantiation regressions.

## Dependencies

Standard library only (`tomllib` requires Python 3.11+; the project pyproject already targets 3.12-3.13, so this is satisfied by `uv sync`).

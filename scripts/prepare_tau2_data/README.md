# `prepare_tau2_data/`

One-shot scripts that import tau2-bench upstream data into
`evaluation/data/<tau2_domain>/`. They run against a local checkout of
[`sierra-research/tau2-bench`](https://github.com/sierra-research/tau2-bench)
pinned at the `voice-user-sim-v1.0` tag (commit `17e07b1`).

## Scripts

| Script | Domain | What it does |
|---|---|---|
| `prepare_telecom.py` | `tau2_telecom` | Converts `db.toml` + `user_db.toml` → JSON. Copies tasks/policy/workflows verbatim. |

Future per-domain scripts (if we ever need to re-sync airline / retail / other tau2 domains) can sit alongside, following the same shape.

## Why TOML → JSON conversion (only for telecom)

Upstream tau2-bench ships airline + retail DBs as `db.json` but the telecom domain ships `db.toml` and `user_db.toml`. Rather than add a TOML branch to the bot-side `update_system_prompt` loader (`rtvi_actions.create_update_system_prompt_action`), we normalize at copy time so the bot only ever loads `.json`. Side benefit: the converted JSON diffs cleanly against future upstream re-syncs (TOML reordering is invisible after the canonicalization pass).

## When to re-run

- After bumping the pinned commit (update `PINNED_COMMIT` in the script first; verify intentional by re-checking each scenario's `expected_scenario_db` hash against the new upstream).
- Never in CI — these scripts are developer-only setup. The output lives in `evaluation/data/tau2_telecom/` and is committed to the repo.

## Usage

```bash
# Default — assumes upstream checkout at /home/heh/github/tau2-bench
python scripts/prepare_tau2_data/prepare_telecom.py

# Override upstream location
python scripts/prepare_tau2_data/prepare_telecom.py --source /path/to/tau2-bench

# Override destination (rare; useful for testing)
python scripts/prepare_tau2_data/prepare_telecom.py --dest /tmp/test_telecom_import
```

The script verifies the upstream HEAD commit against `PINNED_COMMIT` and warns (does not fail) on mismatch — re-sync work intentionally relaxes this check so the operator can test against an unpinned upstream before committing the new pin.

## Dependencies

Standard library only (`tomllib` requires Python 3.11+; the project pyproject already targets 3.12-3.13, so this is satisfied by `uv sync`).

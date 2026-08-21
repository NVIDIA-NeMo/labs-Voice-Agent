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

# Evaluation CLI

The NeMo Labs Voice Agent evaluation harness ships two command-line entry points in `evaluation/`:

| Script | Purpose |
| --- | --- |
| `run_evaluation.py` | Drives the bridge: runs scenarios against a live agent bot and a simulated-user bot, then scores and aggregates the results. |
| `check_resume.py` | Dry-run inspection of an existing result directory: reports which scenarios `--resume` would re-run. Never writes to disk. |

Every default below is read from the `argparse` definitions in those two files.

## Before you run

`run_evaluation.py` only orchestrates — the two bot servers must already be listening. `SERVER_CONFIG_PATH`
is resolved against the current working directory, so `cd evaluation` first (the `run_agent.sh` /
`run_user.sh` helpers do the same and export the ports shown here).

```bash
# Terminal 1 — simulated user bot
cd evaluation && WEBSOCKET_PORT=8766 FASTAPI_PORT=7861 \
  SERVER_CONFIG_PATH=server_configs/user.yaml python bot_server.py

# Terminal 2 — agent under test
cd evaluation && WEBSOCKET_PORT=8765 FASTAPI_PORT=7860 \
  SERVER_CONFIG_PATH=server_configs/agent.yaml python bot_server.py

# Terminal 3 — driver
cd evaluation && python run_evaluation.py --domain eva_airline
```

See [Environment variables](../runtime/environment.md) for the full bot-server variable list and
[Evaluation quickstart](../../evaluate/run-evaluations/quickstart.md) for the end-to-end walkthrough.

## `run_evaluation.py`

Use the following option groups to select scenarios, connect the bots, control runs, and configure scoring.

### Scenario selection and listing

Use these flags to list or select the scenarios included in a run.

| Flag | Default | Description |
| --- | --- | --- |
| `--list` | off | Print every registered scenario, grouped by domain, and exit. |
| `--list-domains` | off | Print every domain with its scenario count, and exit. |
| `--scenarios NAME [NAME ...]` | all registered scenarios | Explicit scenario names. Takes precedence over `--domain`. |
| `--domain DOMAIN` | `None` | Run every scenario whose name starts with the `DOMAIN__` prefix. Exits with status 1 if nothing matches. |

Domain filtering is a literal prefix match, so `--domain tau2_telecom` selects only the
`tau2_telecom__…` scenarios; the parallel workflow-policy registration is a separate domain,
`--domain tau2_telecom_workflow`. The four benchmark domains carry 50 (`eva_airline`), 50
(`tau2_airline`), 114 (`tau2_retail`) and 114 (`tau2_telecom`, mirrored by
`tau2_telecom_workflow`) scenarios — see [Benchmark domains](../../evaluate/understand-scoring/benchmarks.md).

### Connection and audio

Use these flags to configure bot endpoints, result storage, and audio streaming.

| Flag | Default | Description |
| --- | --- | --- |
| `--user-url URL` | `ws://localhost:8766` | WebSocket URL of the simulated-user bot. |
| `--agent-url URL` | `ws://localhost:8765` | WebSocket URL of the agent under test. |
| `--output-dir DIR` | `./eval_results` | Parent directory; each run creates `eval_<TIMESTAMP>/` inside it. |
| `--output-sample-rate HZ` | `16000` | Sample rate of the recorded conversation audio written per scenario. |
| `--audio-chunk-in-seconds SEC` | `0.016` | Size of each audio chunk the bridge streams between the bots. |
| `--pause SEC` | `0.5` | Pause between scenario setup and the scenario run. |

### Run control

Use these flags to set scenario limits, resume runs, and control matching behavior.

| Flag | Default | Description |
| --- | --- | --- |
| `--duration SEC` | `None` | Hard cap per scenario. When unset, each scenario's own `max_duration` applies (900 s for the eva and tau2 bases, shorter for the small demo domains). |
| `--min-agent-turns N` | `3` | Minimum agent LLM responses for a scenario to be scored on its own merits. Pass `0` to disable. |
| `--resume TIMESTAMP` | `None` | Reuse the existing `eval_<TIMESTAMP>/` session directory under `--output-dir`. Exits with status 1 if that directory does not exist. |
| `--strict-match` | off | Force `disallow_extra_items=True` on every scenario, overriding each scenario's own setting, so the action-list comparator requires exact-length matches. |

`--min-agent-turns` is a stall filter for runs where the LLM backend hung. Scenarios below the threshold
are **counted as failures** in the composite success rate and **skipped** in the per-signal rates
(action-match, DB-state, NL-assertion) — they are not dropped from the run. Under `--resume` they are
additionally treated as in-flight and re-run. The turn count comes from the live-recorded
`token_usage.agent.n_calls` in `metrics.json`, falling back to the saved agent LLM context for older runs.

### LLM judge

Use these flags to connect and configure the optional LLM judge.

| Flag | Default | Description |
| --- | --- | --- |
| `--judge-url URL` | `http://localhost:8000/v1/chat/completions` | OpenAI-compatible chat-completions endpoint for the judge. |
| `--judge-model NAME` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | Judge model name. |
| `--judge-api-key KEY` | `None` | Inline API key. Redacted to `<redacted>` in `run_args.json`. |
| `--judge-api-key-name VAR` | `JUDGE_API_KEY` | Environment variable read for the key when `--judge-api-key` is not given. |
| `--judge-threshold F` | `0.9` | Score threshold above which the judge verdict counts as a pass. |
| `--judge-timeout SEC` | `120.0` | Per-request timeout. |
| `--judge-max-tokens N` | `100000` | Generation budget. Reasoning judges spend most of it on thinking, so lowering it can truncate the verdict. |
| `--judge-temperature F` | `1.0` | Sampling temperature. |
| `--judge-top-p F` | `0.95` | Nucleus sampling `top_p`. |
| `--judge-seed N` | `42` | Sampling seed, for run-to-run reproducibility. |
| `--judge-thinking-token-budget N` | `None` | Provider-specific thinking budget; only sent when set. |
| `--judge-include-conversation` | off | Include the bridge transcript turns in the judge input. |
| `--judge-compact-context` | off | Compact the LLM context histories before sending them to the judge. |
| `--judge-context-message-limit N` | `None` | Max context messages, applied when `--judge-compact-context` is on. |
| `--judge-context-system-string-limit N` | `None` | Max system-message length, applied when `--judge-compact-context` is on. |
| `--judge-context-string-limit N` | `None` | Max non-system string length, applied when `--judge-compact-context` is on. |

The judge is constructed only when both `--judge-url` and `--judge-model` are non-empty; passing an empty
string to either disables judging for the run. Judge thinking is always enabled and is deliberately not
exposed as a flag — toggling it would silently change what the scores mean across runs.

Numeric options are validated before the run starts, and a violation exits through `parser.error` (status 2):

| Option | Rule |
| --- | --- |
| `--judge-threshold`, `--judge-top-p` | finite float between 0 and 1 inclusive |
| `--judge-timeout` | finite float greater than 0 |
| `--judge-max-tokens`, `--judge-thinking-token-budget`, `--judge-context-message-limit`, `--judge-context-system-string-limit`, `--judge-context-string-limit` | positive integer |

`--judge-temperature` and `--judge-seed` are passed through unvalidated.

### Exit codes

The evaluation driver exits with one of the following status codes.

| Code | Meaning |
| --- | --- |
| `0` | Run completed, or `--list` / `--list-domains` printed and exited. |
| `1` | Unknown scenario name, empty domain, no registered scenarios, missing `--resume` directory, `KeyboardInterrupt`, or an unhandled exception during the run. |
| `2` | `argparse` usage error, including the judge numeric validation above. |

### Invocation record

Every run writes `run_args.json` into the session directory with the shape
`{"invocations": [...]}` — each entry holds `started_at`, the raw `argv`, the parsed args (judge API key
redacted), and the resolved scenario names and count. A `--resume` invocation appends a new entry and
soft-checks it against the previous one on the scoring-relevant fields `domain`, `scenarios`, `duration`,
`judge_url`, `judge_model`, `judge_threshold`, `judge_max_tokens`, `judge_temperature`, `judge_top_p`,
`judge_seed`, and `strict_match`. Mismatches log a warning and the run proceeds — nothing is blocked.

Result artifacts (`all_metrics.json`, `all_summary.txt`, per-scenario `metrics.json`) are described in
[Reading results](../../evaluate/run-evaluations/results.md) and the [Metrics dictionary](metrics.md); the six scoring
signals are described in [Scoring](../../evaluate/understand-scoring/scoring.md).

## `check_resume.py`

Dry-run classifier for a finished or interrupted session directory. It applies the same
classification the runner uses, so its output predicts what `--resume` would do, without moving or
deleting anything.

```bash
cd evaluation
python check_resume.py ../eval_results/eval_20260618_072325 --min-agent-turns 3
```

| Argument | Default | Description |
| --- | --- | --- |
| `eval_dir` | required | Path to the `eval_<TIMESTAMP>/` session directory. Exits with status 1 if it is not a directory. |
| `--min-agent-turns N` | `0` | Flag scenarios with fewer than N agent LLM responses as stalled. `0` disables the check. |

The `--min-agent-turns` default here is `0`, unlike the runner's `3`. Pass the same value you intend to
use with `--resume` if you want the preview to match.

Each subdirectory is bucketed into one of three states:

| State | Meaning |
| --- | --- |
| `completed` | Has a readable `metrics.json` with at least one turn and enough agent turns. `--resume` loads its metrics from disk and skips the scenario. |
| `rerun` | Missing or unreadable `metrics.json`, zero turns, or fewer agent turns than `--min-agent-turns`. `--resume` moves it to `<scenario>.killed.<resume_timestamp>/` (dropping a `__KILLED__` marker file inside) and runs it again. |
| `fresh` | No subdirectory yet; it runs normally. |

Directories already named `*.killed.*` or containing `__KILLED__`, and top-level files such as
`run_args.json` and `evaluation_log.txt`, are ignored. The script prints per-bucket counts followed by the
re-run and fresh lists, with the classification reason next to each re-run entry.

For the resume workflow itself, see [Resuming a run](../../evaluate/run-evaluations/resume.md).

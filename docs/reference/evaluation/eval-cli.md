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

The `argparse` definitions in those two files specify every default below.

## Before You Run

Start both bot servers before you run `run_evaluation.py`. The code resolves `SERVER_CONFIG_PATH` against the
current working directory, so run `cd evaluation` first. The `run_agent.sh` and `run_user.sh` helpers also
change to that directory and export the following ports.

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

Refer to [Environment Variables](../runtime/environment.md) for the complete bot-server variable list and
[Evaluation Quickstart](../../evaluate/run-evaluations/quickstart.md) for the end-to-end walkthrough.

## `run_evaluation.py`

Use the following option groups to select scenarios, connect the bots, control runs, and configure scoring.

### Scenario Selection and Listing

Use these flags to list or select the scenarios included in a run.

| Flag | Default | Description |
| --- | --- | --- |
| `--list` | off | Print every registered scenario, grouped by domain, and exit. |
| `--list-domains` | off | Print every domain with its scenario count, and exit. |
| `--scenarios NAME [NAME ...]` | all registered scenarios | Explicit scenario names. Takes precedence over `--domain`. |
| `--domain DOMAIN` | `None` | Run every scenario whose name starts with the `DOMAIN__` prefix. Exits with status 1 if nothing matches. |

Domain filtering is a literal prefix match, so `--domain tau2_telecom` selects only the
`tau2_telecom__…` scenarios. The parallel workflow-policy registration is a separate domain,
`--domain tau2_telecom_workflow`. The four benchmark domains carry 50 (`eva_airline`), 50
(`tau2_airline`), 114 (`tau2_retail`), and 114 (`tau2_telecom`, mirrored by
`tau2_telecom_workflow`) scenarios. For domain details, refer to
[Benchmark Domains](../../evaluate/understand-scoring/benchmarks.md).

### Connection and Audio

Use these flags to configure bot endpoints, result storage, and audio streaming.

| Flag | Default | Description |
| --- | --- | --- |
| `--user-url URL` | `ws://localhost:8766` | WebSocket URL of the simulated-user bot. |
| `--agent-url URL` | `ws://localhost:8765` | WebSocket URL of the agent under test. |
| `--output-dir DIR` | `./eval_results` | Parent directory; each run creates `eval_<TIMESTAMP>/` inside it. |
| `--output-sample-rate HZ` | `16000` | Sample rate of the recorded conversation audio written per scenario. |
| `--audio-chunk-in-seconds SEC` | `0.016` | Size of each audio chunk the bridge streams between the bots. |
| `--pause SEC` | `0.5` | Pause between scenario setup and the scenario run. |

### Run Control

Use these flags to set scenario limits, resume runs, and control matching behavior.

| Flag | Default | Description |
| --- | --- | --- |
| `--duration SEC` | `None` | Hard cap per scenario. When unset, each scenario's own `max_duration` applies (900 s for the eva and tau2 bases, shorter for the small demo domains). |
| `--min-agent-turns N` | `2` | Minimum agent large language model (LLM) responses for a scenario to be scored on its own merits. Pass `0` to disable. |
| `--resume TIMESTAMP` | `None` | Reuse the existing `eval_<TIMESTAMP>/` session directory under `--output-dir`. Exits with status 1 if that directory does not exist. |
| `--strict-match` | off | Force `disallow_extra_items=True` on every scenario, overriding each scenario's own setting, so the action-list comparator requires exact-length matches. |

`--min-agent-turns` is a stall filter for runs where the LLM backend hung. Scenarios below the threshold
are **counted as failures** in the composite success rate and **skipped** in the per-signal rates
(action-match, DB-state, NL-assertion) — they are not dropped from the run. Under `--resume` they are
additionally treated as in-flight and re-run. The turn count comes from the live-recorded
`token_usage.agent.n_calls` in `metrics.json`, falling back to the saved agent LLM context for older runs.

The default is `2` because the floor has to sit *below* the shortest legitimate conversation. Across 500
measured scenarios, the agent LLM-call distribution bottoms out at exactly 3 calls, where 21 scenarios sit.
Every one of those is a clean exit (`stop_reason` of `[EXIT]`), and 11 of them succeeded. A floor of `3`
therefore lands inside that legitimate cluster and fails scenarios that did the work; `2` sits below it.

### Automatic Retry

The runner can re-run a scenario immediately, inside the same session, when the finished attempt looks like
an infrastructure failure instead of a task failure. This is separate from `--resume`, which you invoke by
hand after a run ends.

| Flag | Default | Description |
| --- | --- | --- |
| `--no-auto-resume-on-stale` | retry is on | Disable the automatic re-run of a scenario that recorded zero conversation turns. |
| `--auto-resume-on-insufficient-turns` | off | Also re-run scenarios that merely fall below `--min-agent-turns`. |
| `--max-auto-retries N` | `1` | Maximum automatic re-runs per scenario. Bounds the loop when a backend is persistently down. |
| `--auto-retry-backoff-secs SECS` | `5.0` | Seconds to wait before an automatic re-run. An immediate retry against a hung backend is the least likely moment to succeed. |

The two triggers differ in how certain the failure is:

- **Zero conversation turns**, enabled by default. The bots never exchanged audio, so the attempt holds no
  measurement that a retry could bias. Turn off with `--no-auto-resume-on-stale`.
- **Fewer agent turns than `--min-agent-turns`**, disabled by default. That floor is a heuristic, and
  legitimately short scenarios sit on it. Because an insufficient-turns scenario is forced to
  `is_successful: false`, retrying it converts a guaranteed failure into a fresh draw and biases the
  success rate upward. Run-to-run churn within a single arm of these benchmarks measures 12% to 32%, so the
  bias is real rather than theoretical. Enable it only when you accept that trade.

The stale trigger reads the bridge's own conversation record, not `token_usage.agent.n_calls`. Token usage
increments only when the agent bot emits an RTVI token-usage message, so a backend that does not report
usage reads as zero calls on a perfectly healthy run. Gating the default-on trigger on that counter would
silently re-run every scenario.

A retried attempt is preserved on disk as `<scenario>.killed.autoretry<N>.<timestamp>/` with a `__KILLED__`
marker file inside, so its logs stay available for triage. Every `metrics.json` records `auto_retry_count`,
and `auto_retry_reasons` appears when at least one retry fired. Use those fields for downstream analysis
rather than matching directory names.

### LLM Judge

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
| `--judge-thinking-token-budget N` | `None` | Provider-specific thinking budget. Sent only when you set it, because some hosted OpenAI-compatible endpoints reject unknown request fields. |
| `--judge-include-conversation` | off | Include the bridge transcript turns in the judge input. |
| `--judge-compact-context` | off | Compact the LLM context histories before sending them to the judge. |
| `--judge-context-message-limit N` | `None` | Max context messages, applied when `--judge-compact-context` is on. |
| `--judge-context-system-string-limit N` | `None` | Max system-message length, applied when `--judge-compact-context` is on. |
| `--judge-context-string-limit N` | `None` | Max non-system string length, applied when `--judge-compact-context` is on. |

The runner constructs the judge only when both `--judge-url` and `--judge-model` are nonempty. Passing an
empty string to either option disables judging for the run. Judge thinking is always enabled and is not
exposed as a flag. Changing that behavior would silently change score semantics across runs.

The runner validates numeric options before the run starts. A violation exits through `parser.error`
(status 2):

| Option | Rule |
| --- | --- |
| `--judge-threshold`, `--judge-top-p` | finite float between 0 and 1 inclusive |
| `--judge-timeout` | finite float greater than 0 |
| `--judge-max-tokens`, `--judge-thinking-token-budget`, `--judge-context-message-limit`, `--judge-context-system-string-limit`, `--judge-context-string-limit` | positive integer |

The runner passes `--judge-temperature` and `--judge-seed` through without validation.

### Exit Codes

The evaluation driver exits with one of the following status codes.

| Code | Meaning |
| --- | --- |
| `0` | Run completed, or `--list` / `--list-domains` printed and exited. |
| `1` | Unknown scenario name, empty domain, no registered scenarios, missing `--resume` directory, `KeyboardInterrupt`, or an unhandled exception during the run. |
| `2` | `argparse` usage error, including the judge numeric validation above. |

### Invocation Record

Every run writes `run_args.json` into the session directory with the shape
`{"invocations": [...]}`. Each entry records:

- `started_at` and the raw `argv`.
- The parsed arguments, with the judge API key redacted.
- The resolved scenario names and count.

A `--resume` invocation appends a new entry and
soft-checks it against the previous one on the scoring-relevant fields `domain`, `scenarios`, `duration`,
`judge_url`, `judge_model`, `judge_threshold`, `judge_max_tokens`, `judge_temperature`, `judge_top_p`,
`judge_seed`, and `strict_match`. Mismatches log a warning but do not block the run.

For result artifacts (`all_metrics.json`, `all_summary.txt`, and per-scenario `metrics.json`), refer to
[Reading Results](../../evaluate/run-evaluations/results.md) and the [Metrics Dictionary](metrics.md). For the
six scoring signals, refer to [Scoring](../../evaluate/understand-scoring/scoring.md).

## `check_resume.py`

Use `check_resume.py` to classify a finished or interrupted session directory without moving or deleting
anything. The script applies the same classification as the runner, so its output predicts what `--resume`
would do.

```bash
cd evaluation
python check_resume.py ../eval_results/eval_20260618_072325 --min-agent-turns 2
```

| Argument | Default | Description |
| --- | --- | --- |
| `eval_dir` | required | Path to the `eval_<TIMESTAMP>/` session directory. Exits with status 1 if it is not a directory. |
| `--min-agent-turns N` | `2` | Flag scenarios with fewer than N agent LLM responses as stalled. `0` disables the check. |

The `--min-agent-turns` default here is `2`, matching `run_evaluation.py`. Pass the same value you intend
to use with `--resume` whenever you override it, so the preview matches the run. Raising the floor above
`2` is rarely what you want. Legitimately short scenarios bottom out at 3 agent responses, so a floor of
`3` or more re-runs real results instead of stalls.

Each subdirectory is bucketed into one of three states:

| State | Meaning |
| --- | --- |
| `completed` | Has a readable `metrics.json` with at least one turn and enough agent turns. `--resume` loads its metrics from disk and skips the scenario. |
| `rerun` | Missing or unreadable `metrics.json`, zero turns, or fewer agent turns than `--min-agent-turns`. `--resume` moves it to `<scenario>.killed.<resume_timestamp>/` (dropping a `__KILLED__` marker file inside) and runs it again. |
| `fresh` | No subdirectory yet; it runs normally. |

Directories already named `*.killed.*` or containing `__KILLED__`, and top-level files such as
`run_args.json` and `evaluation_log.txt`, are ignored.

The script prints the per-bucket counts first, then up to three lists in this order:

| Section | Contents |
| --- | --- |
| `Would re-run (N):` | One line per `rerun` scenario, with the classification reason in brackets. A scenario the runner already retried in-run gets the suffix `; already auto-retried <n>x in-run` appended to that reason. |
| `Completed after an automatic in-run retry (N):` | One line per `completed` scenario whose `metrics.json` records a nonzero `auto_retry_count`, formatted as `<scenario>  [auto_retry_count=<n>]`. |
| `Fresh / never started (N):` | One line per `fresh` scenario. |

The retry list is informational. It reports which finished results needed an automatic re-run before they
succeeded, and it does not change the `completed`, `rerun`, or `fresh` counts. It covers only scenarios in
the `completed` bucket. A scenario that was retried and still classifies as re-runnable stays under
`Would re-run`, where the inline suffix flags it — that combination is the most diagnostic of the two,
because it means the automatic retry did not fix the problem. The suffix needs a readable `metrics.json`,
so a scenario that has no metrics file at all is listed without it. When nothing is queued for a re-run and
no scenario is fresh, the script prints `All scenarios are complete — nothing to resume.`

Detect retried scenarios through the `auto_retry_count` field in `metrics.json`, which is what this script
reads. Do not glob for `<scenario>.killed.autoretry*/` directories: that layout is an implementation detail
of how the runner preserves failed attempts.

For the resume workflow, refer to [Resuming a Run](../../evaluate/run-evaluations/resume.md).

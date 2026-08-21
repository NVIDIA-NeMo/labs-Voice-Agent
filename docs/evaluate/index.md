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

# About Voice Agent Evaluation

The NeMo Labs Voice Agent evaluation harness scores a voice agent by having a **simulated user** — another
full voice agent — hold a live spoken conversation with it. A **bridge** process sits between the two,
routes audio, measures latency, seeds scenario state, captures what each side actually did, and hands the
result to a runner that scores it.

Nothing is scored from text alone. Every turn goes through TTS on one side and ASR on the other, so ASR
errors, barge-in, and turn-taking failures show up in the score the same way they would with a human caller.

## Workflow Overview

The evaluation workflow connects two independent bot servers through a bridge and then sends the captured
conversation evidence to the runner for scoring.

### Architecture

The following diagram shows the audio, state, and scoring flow between the four runtime components.

```
┌──────────────────────┐      audio + RTVI     ┌──────────────────────┐     audio + RTVI    ┌──────────────────────┐
│   User bot server    │◄─────────────────────►│        Bridge        │◄───────────────────►│  Agent bot server    │
│  (simulated user)    │                       │                      │                     │  (agent under test)  │
│                      │                       │  audio routing       │                     │                      │
│  ASR → LLM → TTS     │                       │  latency metrics     │                     │  ASR → LLM → TTS     │
│  ws://localhost:8766 │                       │  transcript + WAV    │                     │  ws://localhost:8765 │
│                      │                       │  scenario seeding    │                     │                      │
│  shared_state["db"]  │                       │  exit detection      │                     │  shared_state["db"]  │
│  (user-side DB, only │                       │  cross-side sync     │                     │  (agent-side DB:     │
│   for dual-side      │                       │  end-of-run pull     │                     │   reservations,      │
│   domains)           │                       │                      │                     │   bills, lines, ...) │
└──────────────────────┘                       └──────────────────────┘                     └──────────────────────┘
                                                          │
                                                          ▼
                                                 runner → scoring → eval_results/
```

Both bot servers are the same script (`evaluation/bot_server.py`); the `SERVER_CONFIG_PATH` environment
variable picks the role. Each runs its own Pipecat pipeline and holds a per-scenario `shared_state` dict.

| Component | Source | Responsibility |
|---|---|---|
| User bot | `evaluation/bot_server.py` with `server_configs/user.yaml` | Plays the customer. Prompted per scenario with a persona, a task, and the actions to attempt. |
| Agent bot | `evaluation/bot_server.py` with `server_configs/agent.yaml` | The system under test. Receives the domain policy prompt and the domain's tool surface. |
| Bridge | `nemo_voice_agent/evaluation/bridge.py` | One WebSocket client per bot, one thread each, audio shuttled through thread-safe queues. |
| Runner | `nemo_voice_agent/evaluation/runner.py` | Drives the scenario list, scores each result, writes per-scenario and run-level artifacts. |
| CLI | `evaluation/run_evaluation.py` | Argument parsing, scenario selection, resume handling. |

## Key Concepts

The bridge and scoring outputs are the two concepts to understand before you run or extend an evaluation.

### What the bridge does

The bridge is responsible for the following runtime coordination and evidence-capture tasks.

- **Audio routing.** Audio is resampled at the source (matching browser-client behavior) rather than
  leaving small chunks for ASR to resample. Optional additive noise, configured per scenario, is applied
  on the user-to-agent path only — the agent hears a degraded channel, the simulated user does not.
- **Latency measurement.** Every measurement pairs the moment the user stopped speaking with the moment the
  agent started speaking. The bridge reports mean, P50, P95, min, and max per scenario and across the run.
- **Transcript and audio capture.** A timestamped conversation log, a segLST speaker-segment file, and a
  stereo WAV (left channel: user to agent; right channel: agent to user).
- **Scenario setup.** Per scenario the bridge sends `update_system_prompt` (prompt, tool registration,
  shared-state reset) followed by `apply_initialization` (merges the scenario's `shared_state_init` payload,
  resolves `db_path` to a loaded DB, applies init-function mutations). Both bots always receive
  `apply_initialization`, because the DB-load step runs even when a scenario declares no init mutations.
- **Termination detection.** The agent ends a conversation by calling its end-conversation tool, which emits
  an `<exit>` tag; the bridge records stop reason `[EXIT]`. Hitting the time limit records `[TIMEOUT]`.
- **Cross-side state sync.** For dual-side domains, each write tool emits an `action-applied` event; the
  bridge replays it onto shadow DBs, calls the scenario's `sync_state`, and pushes the resulting delta to
  the other bot via `apply_sync_delta`. Single-side domains skip this entirely. See
  [tau2_telecom](domain-guides/tau2-telecom.md).
- **End-of-scenario pull.** The bridge pulls `get_context_history` and `get_scenario_summary` from each bot
  inside that bot's own WebSocket scope. `get_scenario_summary` returns `{actions, db_hash}`; the inline DB
  comes back only when the bridge opts in with `include_db` (needed for DB-state assertions).

The bridge relies on six RTVI actions being registered by both bots — `reset`,
`update_system_prompt`, `get_context_history`, `get_scenario_summary`, `apply_initialization`, and
`apply_sync_delta`, all defined in `nemo_voice_agent/pipecat/processors/frameworks/rtvi_actions.py`. Any
agent that implements them can be evaluated; see [External agents](run-evaluations/external-agents.md) and the
[RTVI message reference](../reference/runtime/rtvi-messages.md).

### What you get out

Each scenario is scored by up to six orthogonal signals. A scenario's domain declares which of them gate
the composite `is_successful` verdict; the rest are still computed and saved as informational.

| Signal | `metrics.json` key | Kind |
|---|---|---|
| `ACTION_MATCH` | `is_action_match` | Deterministic, path-dependent comparison against the gold action list |
| `DB_STATE_MATCH` | `db_state_match` | Deterministic SHA-256 comparison of the post-run DB against the gold DB |
| `DB_STATE_ASSERTION` | `db_state_assertion_pass_rate` | Deterministic per-predicate checks over the pulled DB |
| `NL_ASSERTION` | `nl_assertion_pass_rate` | LLM-judged natural-language claims about the conversation |
| `JUDGE_PASSED` | `judge_passed` | LLM judge score compared against `--judge-threshold` |
| `CLEAN_EXIT` | `clean_exit` | Agent ended the call itself (`[EXIT]`), rather than timing out |

`CLEAN_EXIT` is in every domain's whitelist: an agent that does the right work but never stops talking is
not a successful agent. Full semantics, the per-domain whitelist matrix, and the strict-conjunction rule are
in [Scoring model](understand-scoring/scoring.md).

A run writes `eval_results/eval_<TIMESTAMP>/` containing `all_metrics.json`, `all_summary.txt`,
`all_latencies.csv`, `evaluation_log.txt`, `run_args.json`, and one subdirectory per scenario with its
transcript, stereo WAV, bridge log, pulled actions, DB hash, `metrics.json`, judge output, a snapshot of
the exact prompts and tools used, and both bots' full LLM context histories. See
[Reading results](run-evaluations/results.md) and the [metrics reference](../reference/evaluation/metrics.md).

## Run it

Three terminals. `SERVER_CONFIG_PATH` is resolved against the current working directory, so `cd evaluation`
first — running from the repo root fails with `FileNotFoundError`.

```bash
# Terminal 1 — simulated user
cd evaluation
SERVER_CONFIG_PATH=server_configs/user.yaml WEBSOCKET_PORT=8766 CUDA_VISIBLE_DEVICES=0 python bot_server.py

# Terminal 2 — agent under test
cd evaluation
SERVER_CONFIG_PATH=server_configs/agent.yaml WEBSOCKET_PORT=8765 CUDA_VISIBLE_DEVICES=1 python bot_server.py

# Terminal 3 — bridge + runner
cd evaluation
python run_evaluation.py \
    --user-url ws://localhost:8766 \
    --agent-url ws://localhost:8765 \
    --domain restaurant
```

`run_agent.sh` and `run_user.sh` wrap terminals 1 and 2 with the environment already exported; run them from
`evaluation/` for the same reason. Use `--list-domains` and `--list` to see what is registered.

Two defaults to know before you compare runs:

- `--min-agent-turns` defaults to `3`. Scenarios where the agent completed fewer turns are counted as
  **failures** in the composite success rate and **skipped** in the per-signal rates — this catches a hung
  LLM server, but it also depresses the headline while shrinking each per-signal denominator. Check the
  warning line in `all_summary.txt` before reading the numbers. Pass `0` to disable.
- `--duration` defaults to unset, in which case each scenario's own `max_duration` applies. Passing a value
  overrides every scenario.

The full flag list is in the [eval CLI reference](../reference/evaluation/eval-cli.md); interrupted runs are picked up
with `--resume`, described in [Resuming a run](run-evaluations/resume.md).

## Benchmark domains

The harness includes large benchmark-derived domains and smaller in-repository smoke domains.

| Domain | Scenarios | Notes |
|---|---|---|
| [eva_airline](domain-guides/eva-airline.md) | 50 | Flight changes, irregular operations, refunds, vouchers |
| [tau2_airline](domain-guides/tau2-airline.md) | 50 | Reservation cancel / refund / rebook / upgrade |
| [tau2_retail](domain-guides/tau2-retail.md) | 114 | Order cancel / exchange / return, address changes |
| [tau2_telecom](domain-guides/tau2-telecom.md) | 114 | Dual-side tech support; a parallel `tau2_telecom_workflow` registration runs the same 114 tasks against a procedural policy variant |

Smaller in-repo domains (`restaurant`, `customer_service`, `qa`) exist as smoke sets and as worked examples
of the authoring pattern. Fixtures for all domains are packaged inside the library at
`nemo_voice_agent/evaluation/data/`, resolved by `get_eval_data_root()`, which honors the `EVAL_DATA_ROOT`
environment variable as an override. Upstream sources and licenses are recorded in
[Data provenance](domain-guides/data-provenance.md).

## Next Steps

Continue with the first-run workflow, domain catalogue, scoring details, or extension guides.

- [Quickstart](run-evaluations/quickstart.md) — first run, end to end.
- [Benchmarks and domains](understand-scoring/benchmarks.md) — what each domain measures.
- [Scoring model](understand-scoring/scoring.md) — the six signals and how the composite verdict is formed.
- [Authoring scenarios](create-evaluations/authoring-scenarios.md), [tools](create-evaluations/authoring-tools.md), and
  [domains](create-evaluations/authoring-domains.md) — extend the data layer.
- [External agents](run-evaluations/external-agents.md) — evaluate an agent that is not this pipeline.

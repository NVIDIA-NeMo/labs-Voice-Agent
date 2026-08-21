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

The NeMo Labs Voice Agent evaluation harness measures a voice agent in a live spoken conversation with a
**simulated user**. A **bridge** routes audio, measures latency, seeds scenario state, captures each side's
actions, and sends the evidence to a scoring runner.

Nothing is scored from text alone. Each bot runs an automatic speech recognition (ASR), large language model
(LLM), and text-to-speech (TTS) pipeline. Every turn goes through TTS on one side and ASR on the other. The
score therefore captures ASR errors, barge-in, and turn-taking failures that a human caller would experience.

## Workflow Overview

The evaluation workflow connects two independent bot servers through a bridge and then sends the captured
conversation evidence to the runner for scoring.

### Architecture

The following diagram shows the audio, state, and scoring flow between the four runtime components.
The bots and bridge exchange control messages through the real-time voice interface (RTVI).
For stateful domains, each bot can hold its own database (DB) in shared state.

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
| Command-line interface (CLI) | `evaluation/run_evaluation.py` | Argument parsing, scenario selection, resume handling. |

## Key Concepts

The bridge and scoring outputs are the two concepts to understand before you run or extend an evaluation.

### What the Bridge Does

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
  the other bot through `apply_sync_delta`. Single-side domains skip this step. Refer to
  [tau2_telecom](domain-guides/tau2-telecom.md).
- **End-of-scenario pull.** The bridge pulls `get_context_history` and `get_scenario_summary` from each bot
  inside that bot's own WebSocket scope. `get_scenario_summary` returns `{actions, db_hash}`; the inline DB
  comes back only when the bridge opts in with `include_db` (needed for DB-state assertions).

The bridge relies on both bots registering six RTVI actions — `reset`,
`update_system_prompt`, `get_context_history`, `get_scenario_summary`, `apply_initialization`, and
`apply_sync_delta`, all defined in `nemo_voice_agent/pipecat/processors/frameworks/rtvi_actions.py`. Any
agent that implements them can be evaluated. Refer to [External agents](run-evaluations/external-agents.md) and the
[RTVI message reference](../reference/runtime/rtvi-messages.md).

### What You Get Out

The runner scores each scenario with up to six independent signals. A scenario's domain declares which signals gate
the composite `is_successful` verdict; the rest are still computed and saved as informational.

| Signal | `metrics.json` Key | Kind |
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

A run writes session artifacts to `eval_results/eval_<TIMESTAMP>/`. These include `all_metrics.json`,
`all_summary.txt`, `all_latencies.csv`, `evaluation_log.txt`, and `run_args.json`. Each scenario subdirectory
contains its transcript, stereo WAV, bridge log, pulled actions, DB hash, metrics, judge output, prompt and
tool snapshots, and both bots' full LLM context histories. Refer to
[Reading results](run-evaluations/results.md) and the [metrics reference](../reference/evaluation/metrics.md).

## Run It

Use three terminals. `SERVER_CONFIG_PATH` resolves against the current working directory, so run
`cd evaluation` first. Running from the repository root fails with `FileNotFoundError`.

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
`evaluation/` for the same reason. Use `--list-domains` and `--list` to list registered domains and scenarios.

Two defaults to know before you compare runs:

- `--min-agent-turns` defaults to `3`. Scenarios where the agent completed fewer turns are counted as
  **failures** in the composite success rate and **skipped** in the per-signal rates — this catches a hung
  LLM server, but it also depresses the headline while shrinking each per-signal denominator. Check the
  warning line in `all_summary.txt` before reading the numbers. Pass `0` to disable.
- `--duration` defaults to unset, in which case each scenario's own `max_duration` applies. Passing a value
  overrides every scenario.

The full flag list is in the [eval CLI reference](../reference/evaluation/eval-cli.md); interrupted runs are picked up
with `--resume`, described in [Resuming a run](run-evaluations/resume.md).

## Benchmark Domains

The harness includes large benchmark-derived domains and smaller in-repository verification domains.

| Domain | Scenarios | Notes |
|---|---|---|
| [eva_airline](domain-guides/eva-airline.md) | 50 | Flight changes, irregular operations, refunds, vouchers |
| [tau2_airline](domain-guides/tau2-airline.md) | 50 | Reservation cancel / refund / rebook / upgrade |
| [tau2_retail](domain-guides/tau2-retail.md) | 114 | Order cancel / exchange / return, address changes |
| [tau2_telecom](domain-guides/tau2-telecom.md) | 114 | Dual-side tech support; a parallel `tau2_telecom_workflow` registration runs the same 114 tasks against a procedural policy variant |

Smaller in-repository domains (`restaurant`, `customer_service`, `qa`) provide basic verification sets and
worked examples
of the authoring pattern. Fixtures for all domains are packaged inside the library at
`nemo_voice_agent/evaluation/data/`, resolved by `get_eval_data_root()`, which honors the `EVAL_DATA_ROOT`
environment variable as an override. Upstream sources and licenses are recorded in
[Data provenance](domain-guides/data-provenance.md).

## Next Steps

Continue with the first-run workflow, domain catalog, scoring details, or extension guides.

- [Quickstart](run-evaluations/quickstart.md) — first run, end to end.
- [Benchmarks and domains](understand-scoring/benchmarks.md) — what each domain measures.
- [Scoring model](understand-scoring/scoring.md) — the six signals and how the composite verdict is formed.
- [Authoring scenarios](create-evaluations/authoring-scenarios.md), [tools](create-evaluations/authoring-tools.md), and
  [domains](create-evaluations/authoring-domains.md) — extend the data layer.
- [External agents](run-evaluations/external-agents.md) — evaluate an agent that is not this pipeline.

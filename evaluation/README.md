<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Voice Agent Evaluator

Evaluate a voice agent by having a simulated user — a second voice agent — talk to it over a live
audio connection. A bridge routes audio between the two bots, measures latency, captures the agent's
tool calls, and scores each scenario.

Full documentation:
**https://docs.nvidia.com/nemo/labs-voice-agent/evaluate-voice-agents/overview**

## Run it

Three terminals, all from this directory (`SERVER_CONFIG_PATH` resolves against the current working
directory):

```bash
cd evaluation
WEBSOCKET_PORT=8766 SERVER_CONFIG_PATH=server_configs/user.yaml  python bot_server.py
WEBSOCKET_PORT=8765 SERVER_CONFIG_PATH=server_configs/agent.yaml python bot_server.py
python run_evaluation.py --domain tau2_retail
```

See [Quickstart](https://docs.nvidia.com/nemo/labs-voice-agent/evaluate-voice-agents/run-evaluations/quickstart)
for judge configuration and where results land.

## Where things are documented

| Topic | Page |
| --- | --- |
| Benchmarks and scenario counts | [Benchmarks & Domains](https://docs.nvidia.com/nemo/labs-voice-agent/evaluate-voice-agents/understand-scoring/benchmarks-domains) |
| The six scoring signals and `is_successful` | [Scoring Model](https://docs.nvidia.com/nemo/labs-voice-agent/evaluate-voice-agents/understand-scoring/scoring-model) |
| Output artifacts and how to read them | [Reading Results](https://docs.nvidia.com/nemo/labs-voice-agent/evaluate-voice-agents/run-evaluations/reading-results) |
| `--resume`, `--min-agent-turns`, long runs | [Resuming & Long Runs](https://docs.nvidia.com/nemo/labs-voice-agent/evaluate-voice-agents/run-evaluations/resuming-long-runs) |
| Every CLI flag and its real default | [Evaluation CLI](https://docs.nvidia.com/nemo/labs-voice-agent/reference/evaluation/evaluation-cli) |
| Authoring scenarios, tools, and domains | [Create Evaluations](https://docs.nvidia.com/nemo/labs-voice-agent/evaluate-voice-agents/create-evaluations/authoring-scenarios) |
| Per-domain notes and fixture provenance | [Domain Guides](https://docs.nvidia.com/nemo/labs-voice-agent/evaluate-voice-agents/domain-guides/fixture-data-provenance) |
| Evaluating a non-NeMo agent | [External Agents](https://docs.nvidia.com/nemo/labs-voice-agent/evaluate-voice-agents/run-evaluations/evaluating-an-external-agent) |

## Files here

| Path | Purpose |
| --- | --- |
| `run_evaluation.py` | The bridge and runner |
| `bot_server.py` | Serves either role; `SERVER_CONFIG_PATH` picks which |
| `check_resume.py` | Dry-run preview of what a `--resume` would re-run |
| `run_agent.sh`, `run_user.sh` | Launch wrappers; run them from this directory |
| `server_configs/` | Agent- and user-side bot configs |

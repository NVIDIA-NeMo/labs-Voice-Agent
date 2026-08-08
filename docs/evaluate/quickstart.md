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

# Quickstart

Run your first NeMo Labs Voice Agent evaluation. The harness needs **three processes**: a simulated-user
bot, the agent under test, and the bridge that connects them over audio and scores the result. See
[Evaluation overview](index.md) for how the pieces fit together.

## Before you start

1. Install the package and activate the venv — see [Installation](../get-started/installation.md).
2. Start a vLLM server. Both eval configs (`evaluation/server_configs/agent.yaml` and `user.yaml`) set
   `start_vllm_on_init: false` and point `llm.base_url` at `http://localhost:8000/v1`, so nothing launches
   vLLM for you. Use the flags from the `llm.vllm_server_params` field of those configs — that field is the
   authoritative source if this snippet drifts:

   ```bash
   vllm serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
       --trust-remote-code --enable-prefix-caching --max-num-seqs 1 \
       --gpu-memory-utilization 0.85 --reasoning-parser deepseek_r1
   ```

   Both bots and the default LLM judge share this one endpoint. See [vLLM backend](../models/vllm.md).
3. Only if you switch to the hosted configs (`agent_nvidia.yaml`, `user_nvidia.yaml`): copy
   `evaluation/.env.example` to `evaluation/.env` and fill in `NVIDIA_API_KEY`.

## The one rule that breaks everything

`SERVER_CONFIG_PATH` is resolved **against the current working directory**, not against the script's
directory. `ConfigManager` receives the raw string and calls `os.path.exists` on it. Every command below —
including the helper shell scripts — must therefore be run from `evaluation/`:

```
FileNotFoundError: Server configuration file not found at server_configs/agent.yaml
```

That error means you ran the command from somewhere other than `evaluation/`. `cd evaluation` and retry, or
export an absolute path.

## Terminal 1 — simulated user bot (port 8766)

```bash
cd evaluation
export SERVER_CONFIG_PATH=server_configs/user.yaml
export WEBSOCKET_PORT=8766
export FASTAPI_PORT=7861
export CUDA_VISIBLE_DEVICES=0
python bot_server.py
```

## Terminal 2 — agent under test (port 8765)

```bash
cd evaluation
export SERVER_CONFIG_PATH=server_configs/agent.yaml
export WEBSOCKET_PORT=8765
export FASTAPI_PORT=7860
export CUDA_VISIBLE_DEVICES=1
python bot_server.py
```

`bot_server.py` is the same script for both roles — `SERVER_CONFIG_PATH` is what selects the role. ASR and
TTS run locally on GPU in each process, so give the two bots separate devices when you can. Each process also
starts a small FastAPI app on `FASTAPI_PORT`, so those two values must differ as well.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SERVER_CONFIG_PATH` | `server_configs/agent.yaml` | Selects the role; resolved against the CWD |
| `WEBSOCKET_PORT` | `8765` | Port the bridge connects to |
| `FASTAPI_PORT` | `7860` | HTTP side-car port |
| `SERVER_HOST` | `0.0.0.0` | Bind address |
| `SERVER_PUBLIC_HOST` | `127.0.0.1` | Host advertised in the connect URL |
| `WEBSOCKET_SCHEME` | `ws` | `ws` or `wss` |

Wait until both processes log that they are serving before starting the bridge. Bot-side logs go to
`bot_user_server.log` and `bot_agent_server.log` in `evaluation/`.

### Helper launch scripts

`run_user.sh` and `run_agent.sh` wrap the two invocations above with the environment already exported. They
resolve their own directory only to locate `bot_server.py`; the exported `SERVER_CONFIG_PATH` stays relative,
so **they must also be run from `evaluation/`**:

```bash
cd evaluation
./run_user.sh    # user-sim role,  server_configs/user.yaml,  ws 8766, http 7861
./run_agent.sh   # agent role,     server_configs/agent.yaml, ws 8765, http 7860
```

## Terminal 3 — the bridge

```bash
cd evaluation
python run_evaluation.py \
    --user-url ws://localhost:8766 \
    --agent-url ws://localhost:8765 \
    --domain restaurant
```

`--user-url` and `--agent-url` already default to those values, so the two flags are optional here. Swapping
them silently inverts the roles — the user prompt lands on the agent bot — so keep 8766 on `--user-url`.

Discover what you can run before committing to a long job:

```bash
cd evaluation
python run_evaluation.py --list-domains   # domain names + scenario counts
python run_evaluation.py --list           # every scenario name, grouped by domain
```

`restaurant`, `customer_service`, and `qa` are small, fast smoke domains. `fastbite` and `simple_qa_1`
through `simple_qa_3` carry no `domain__` prefix, so `--list` files them under "Legacy scenarios" and you run
them with `--scenarios fastbite simple_qa_1` rather than `--domain`. The benchmark domains are much larger —
`eva_airline` (50 scenarios), `tau2_airline` (50), `tau2_retail` (114), and `tau2_telecom` (114, plus a
parallel `tau2_telecom_workflow` registration over the same 114 tasks). See
[Benchmark domains](benchmarks.md).

### Flags worth knowing on run one

| Flag | Default | Notes |
| --- | --- | --- |
| `--domain` | none | Runs every scenario whose name starts with `<domain>__` |
| `--scenarios` | none | Explicit scenario names; overrides `--domain` |
| `--output-dir` | `./eval_results` | Relative to the CWD, so it lands under `evaluation/` |
| `--duration` | unset | Per-scenario cap in seconds; when unset, each scenario's own `max_duration` applies |
| `--min-agent-turns` | `3` | Scenarios with fewer completed agent turns are counted as **failures** in the composite rate and skipped in the per-signal rates. Pass `0` to disable |
| `--judge-url` | `http://localhost:8000/v1/chat/completions` | The LLM judge is on by default and reuses your vLLM server |
| `--judge-api-key-name` | `JUDGE_API_KEY` | Environment variable read when `--judge-api-key` is not passed |

Full list: [Evaluation CLI reference](../reference/eval-cli.md).

## Where results land

Each invocation creates a timestamped session directory under `--output-dir`, with one subdirectory per
scenario:

```
evaluation/eval_results/eval_YYYYMMDD_HHMMSS/
├── evaluation_log.txt          # runner log
├── run_args.json               # the invocation(s) that produced this dir (judge key redacted)
├── all_metrics.json            # aggregated metrics across scenarios
├── all_latencies.csv           # every latency measurement
├── all_summary.txt             # human-readable per-scenario + overall summary
└── <scenario_name>/
    ├── conversation_log.txt        # transcript with latency annotations
    ├── conversation_log.seglst.json
    ├── conversation_log.wav        # stereo: L = user to agent, R = agent to user
    ├── bridge_log.txt
    ├── final_agent_response.json   # action records pulled from the bots
    ├── final_scenario_db_hash.txt  # post-run DB hash(es)
    ├── metrics.json                # all signals + the composite is_successful
    ├── judge_result.json           # present when the LLM judge ran
    ├── scenario_config/            # prompts, tools, reference_answer.json, metadata.json
    ├── bot_logs_agent/llm_context.json
    └── bot_logs_user/llm_context.json
```

Read `all_summary.txt` first, then open `metrics.json` and `conversation_log.wav` for any scenario that
failed. `eval_results/` is gitignored.

## Next steps

- [Scoring signals](scoring.md) — the six signals and how the composite verdict is computed.
- [Reading results](results.md) — field-by-field walkthrough of the artifacts above.
- [Resuming a run](resume.md) — pick up a killed run with `--resume <timestamp>`.
- [Evaluating an external agent](external-agents.md) — point the bridge at a non-Pipecat agent.

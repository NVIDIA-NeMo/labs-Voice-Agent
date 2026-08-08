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

# Troubleshooting

Common failures when running NeMo Labs Voice Agent, grouped by where they surface. Each table maps a symptom
to its cause and the fix.

## Read the logs first

The server writes a rotating log file in addition to stderr. The example server hardcodes both the file name
and the level: `examples/generic_voice_agent/server/server.py` calls `setup_logging()` with no arguments, which
defaults to `bot_server.log` at `DEBUG` with daily rotation (`nemo_voice_agent/utils/misc.py`). The path is
relative to the directory you started the process from — so
`examples/generic_voice_agent/server/bot_server.log` for the standard Quickstart invocation. The
`server.log_file` and `server.log_level` keys are read only by the evaluation bots in
`evaluation/bot_server.py`.

Rotated files get a timestamp suffix. When diagnosing a run that just failed, check the
newest `bot_server.<timestamp>.log`, not only `bot_server.log`, which may belong to a still-running process.

## LLM backend

The shipped default (`server_configs/llm_configs/nemotron_nano_v3.yaml`) sets `start_vllm_on_init: false`, so
the voice agent never launches vLLM for you. It also does not probe the endpoint at startup: the server boots
cleanly and the failure only appears on the first LLM turn, right after a browser connects.

| Symptom | Cause | Fix |
|---|---|---|
| Server starts, but the bot never greets you; log shows a connection error against `http://localhost:8000/v1` | vLLM is not running (`start_vllm_on_init: false` in the default model config) | Start vLLM in its own terminal with the flags from that file's `vllm_server_params`, wait for it to report ready, then connect again |
| vLLM is up but the agent still errors | `llm.base_url` points at a different port, or vLLM is serving a different model id than `llm.model` | Confirm with `curl http://localhost:8000/v1/models` and align `llm.model` / `llm.base_url` with what vLLM reports |
| `start_vllm_on_init: true` and the agent quietly starts a *second* vLLM on port 8001 | The already-running server on the requested port serves a different model id, so the launcher scans forward for a free port | Kill the stale server (`lsof -i :8000`) or set `llm.model` to exactly the id the running server reports |
| Log says vLLM "doesn't seem to be supported ... using HuggingFace as the backend" | `llm.type: auto` failed to build a vLLM model config for that repo and fell back to `hf` | Set `type: vllm` explicitly in the model config to force vLLM, or accept the HF backend (which does not support tool calling) |
| Tool calling is configured but the model never calls a tool | Tools require `llm.type: vllm` (with the model's vLLM tool parser flags) or `llm.type: nvidia`; `server.py` gates registration only on `llm.enable_tool_calling` | See [Tool calling](../features/tool-calling.md) and [vLLM backend](../models/vllm.md) |

Check that vLLM is serving before you connect:

```bash
curl -s http://localhost:8000/v1/models
```

## Configuration

`llm.type` accepts `auto`, `hf`, `vllm`, or `nvidia`. See [Server configuration](../configure/server-config.md)
for the full precedence model.

| Symptom | Cause | Fix |
|---|---|---|
| A key you edited in `default.yaml` has no effect | The model sub-YAML referenced by `model_config:` is merged *over* the top-level config, so it wins | Edit the key in the model config (for example `server_configs/llm_configs/nemotron_nano_v3.yaml`) instead |
| `FileNotFoundError: Server configuration file not found at server_configs/agent.yaml` | `SERVER_CONFIG_PATH` is resolved against the current working directory, not the script directory | `cd` into the directory that owns `server_configs/` first, or export an absolute path |
| `llm.enable_reasoning: true` changes nothing | An explicit `llm.model_config:` short-circuits the model-registry lookup that performs the swap to the sibling `*_think.yaml` | Point `model_config:` at the `*_think.yaml` file directly. See [Reasoning mode](../models/reasoning.md) |
| Model loads on the wrong GPU | STT, TTS, and the HuggingFace LLM each have their own device key; diarization does not | Set `stt.device`, `tts.device`, and `llm.device` (HF backend only) to distinct values such as `cuda:0` / `cuda:1`. Diarization has no effective device key — `build_diar` passes `stt.device` to the service, so the `diar.device` entry in the YAML is ignored and diarization always loads onto the STT device |

## Networking and ports

Defaults come from `examples/generic_voice_agent/server/server.py`: `SERVER_HOST=0.0.0.0`,
`WEBSOCKET_PORT=8765`, `FASTAPI_PORT=7860`, `SERVER_PUBLIC_HOST=127.0.0.1`, `WEBSOCKET_SCHEME=ws`. See
[Environment variables](../reference/environment.md).

| Symptom | Cause | Fix |
|---|---|---|
| `OSError: [Errno 98] Address already in use` on startup | A previous server (or another app) still holds 8765 or 7860 | `lsof -i :8765` and `lsof -i :7860`, kill the stale process, or export different `WEBSOCKET_PORT` / `FASTAPI_PORT` values |
| The second browser tab connects and is immediately closed with code `1013` | The server accepts one client at a time; a new connection is rejected and the incumbent session is kept | Close the first tab, then reconnect. The LLM context is preserved across reconnects |
| A remote browser loads the page but cannot open the WebSocket | `/connect` returns a URL built from `SERVER_PUBLIC_HOST`, which defaults to `127.0.0.1` (loopback of the *browser* machine) | Export `SERVER_PUBLIC_HOST` to the server's IP or hostname before starting the server; use `WEBSOCKET_SCHEME=wss` behind TLS termination |
| The client cannot reach `/connect` at all | The browser client dials the FastAPI app on port `7860` on the same host it loaded the page from | Keep `FASTAPI_PORT` at `7860`, or change the port in the client's server configuration in `examples/generic_voice_agent/client/src/app.ts` |

## Browser client

| Symptom | Cause | Fix |
|---|---|---|
| `Error connecting: Cannot read properties of undefined (reading 'enumerateDevices')` | `navigator.mediaDevices` is unavailable because the page is not a secure context (plain HTTP on a non-loopback host) | In Chrome, add `http://<your-machine-ip>:5173/` under `chrome://flags/#unsafely-treat-insecure-origin-as-secure` and restart the browser. Serving the client over HTTPS also works |
| Microphone permission prompt never appears, volume meter stays flat | Same insecure-origin cause, or the site's microphone permission was previously denied | Fix the origin as above, then reset the site's microphone permission in browser settings |
| `SyntaxError: Unexpected reserved word` from `npm run dev` | Node.js is too old for the toolchain (Vite 6 supports Node 18, 20, and 22 or newer) | Upgrade Node.js, then re-run `npm install` |
| `node:internal/errors:496` from `npm run dev` | A partially installed or stale `node_modules` tree | Remove `examples/generic_voice_agent/client/node_modules`, run `npm install`, then `npm run dev` again |
| Port 5173 is taken | Vite's port is pinned in the client's `vite.config.js` | Change the `port` value in that file |

## Speech quality

| Symptom | Cause | Fix |
|---|---|---|
| Transcripts are garbled in a noisy room | The shipped ASR and diarization models are not noise-robust | Use a noise-cancelling microphone or a quieter environment. See [ASR](../models/asr.md) |
| Speaker labels flip between turns, or every turn gets the same speaker | Diarization struggles when voices are similar or accents are under-represented in its training data | Lower `diar.threshold` to raise sensitivity, or disable diarization with `diar.enabled: false`. See [Diarization](../models/diarization.md) |
| The bot interrupts too eagerly, or waits too long before replying | VAD end-of-turn timing | Tune `vad.stop_secs` (silence needed to end a turn), `vad.confidence`, and `vad.min_volume`. See [Turn taking](../models/turn-taking.md) |
| Short acknowledgements like "uh-huh" cut the bot off | Backchannel filtering is disabled | Set `turn_taking.backchannel_phrases_path` to a phrase list instead of `null` |
| You hear the model's reasoning read aloud | The reasoning span is not being stripped before TTS | Set `tts.think_tokens` to the model's think-token pair, or use a vLLM `--reasoning-parser`. See [Reasoning mode](../models/reasoning.md) |

## Model downloads and installation

| Symptom | Cause | Fix |
|---|---|---|
| I/O or network errors while loading a model from HuggingFace | Streaming the repo at startup is failing | Download it first and point the config at the local directory, for example `huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir <local_path>`, then set `llm.model: <local_path>`. The same applies to TTS models |
| A gated repo returns 401/403 | No HuggingFace credentials in the environment | Export `HF_TOKEN` before starting the server; use `HF_HUB_CACHE` to relocate the cache |
| `install.sh` exits with "conda env ... is active" | The installer refuses to run inside a non-`base` conda env, because conda's toolchain breaks source builds such as `cdifflib` | Run `conda deactivate`, then re-run `bash install.sh` |
| A dependency fails to compile during `uv sync` | Missing C toolchain or Python headers | Install `build-essential` and `python3-dev`, or just use `bash install.sh`, which installs them. See [Installation](installation.md) |

## Evaluation harness

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError` for `server_configs/agent.yaml` when starting a bot | `SERVER_CONFIG_PATH` is relative to the current working directory | Start both bots from the `evaluation/` directory |
| Both bots claim the same port | `bot_server.py` is the same script for both roles and defaults to `WEBSOCKET_PORT=8765` | Give each role its own `WEBSOCKET_PORT` (and `FASTAPI_PORT`), as in [Evaluation quickstart](../evaluate/quickstart.md) |
| A tau2 scenario disconnects with WebSocket close code `1009` | A DB payload was inlined into a frame larger than the transport's message cap | Seed the DB by path (`db_path`) rather than inline content; the bot resolves it server-side |

## Still stuck

- Re-read the newest `bot_server.<timestamp>.log` around the first `ERROR` line — most failures above are
  logged with the offending path, port, or model id.
- Turn on audio capture to hear exactly what the pipeline received. See [Audio logging](../configure/audio-logging.md).
- If you are on the hosted backend rather than local models, check the key and endpoint requirements in
  [NVIDIA NIM](../models/nvidia-nim.md).

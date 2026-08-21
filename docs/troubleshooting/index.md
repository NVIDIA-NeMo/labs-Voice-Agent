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

# Troubleshooting NeMo Labs Voice Agent

Use this page to resolve issues that can occur when you run NeMo Labs Voice Agent. Issues are arranged
alphabetically by observable symptom. The final section provides additional diagnostic options.

## A change to default.yaml has no effect

**Cause:** The model sub-YAML referenced by `model_config:` is merged over the top-level configuration, so the
model configuration takes precedence.

**Resolution:** Edit the key in the model configuration instead, such as
`server_configs/llm_configs/nemotron_nano_v3.yaml`.

`llm.type` accepts `auto`, `hf`, `vllm`, or `nvidia`. For the full precedence model, refer to
[Server configuration](../build-voice-agents/configure/server-config.md).

## A dependency fails to compile during uv sync

**Cause:** A C toolchain or Python headers are missing.

**Resolution:** Install `build-essential` and `python3-dev`, or run `bash install.sh`, which installs them. For
more information, refer to [Installation](../get-started/installation.md).

## Agent quietly starts a second vLLM on port 8001

**Cause:** When `start_vllm_on_init: true`, the server on the requested port is already serving a different
model ID. The launcher scans forward for a free port.

**Resolution:** Kill the stale server with `lsof -i :8000`, or set `llm.model` to exactly the ID that the
running server reports.

## Bot interrupts too eagerly or waits too long before replying

**Cause:** Voice activity detection (VAD) end-of-turn timing controls when the bot responds.

**Resolution:** Tune `vad.stop_secs`, which controls the silence required to end a turn, along with
`vad.confidence` and `vad.min_volume`. For more information, refer to
[Turn taking](../about/core-concepts/speech-pipeline/turn-taking.md).

## Bot never greets and the log reports a connection error for http://localhost:8000/v1

**Cause:** The shipped default, `server_configs/llm_configs/nemotron_nano_v3.yaml`, sets
`start_vllm_on_init: false`, so the voice agent does not launch vLLM. It also does not probe the endpoint at
startup. The server starts successfully, and the failure appears only on the first large language model (LLM)
turn after a browser connects.

**Resolution:** Start vLLM in a separate terminal with the flags from that file's `vllm_server_params`. Wait
for it to report ready, and then check that vLLM is serving before you connect again:

```bash
curl -s http://localhost:8000/v1/models
```

## Both evaluation bots claim the same port

**Cause:** `bot_server.py` is the same script for both roles and defaults to `WEBSOCKET_PORT=8765`.

**Resolution:** Give each role its own `WEBSOCKET_PORT` and `FASTAPI_PORT`, as shown in
[Evaluation quickstart](../evaluate/run-evaluations/quickstart.md).

## Client cannot reach /connect

**Cause:** The browser client connects to the FastAPI app on port `7860` on the host from which it loaded the
page.

**Resolution:** Keep `FASTAPI_PORT` at `7860`, or change the port in the client server configuration at
`examples/generic_voice_agent/client/src/app.ts`.

The defaults in `examples/generic_voice_agent/server/server.py` are `SERVER_HOST=0.0.0.0`,
`WEBSOCKET_PORT=8765`, `FASTAPI_PORT=7860`, `SERVER_PUBLIC_HOST=127.0.0.1`, and `WEBSOCKET_SCHEME=ws`. For more
information, refer to [Environment variables](../reference/runtime/environment.md).

## Error connecting: Cannot read properties of undefined (reading 'enumerateDevices') appears

**Cause:** `navigator.mediaDevices` is unavailable because the page is not a secure context, such as plain
HTTP on a non-loopback host.

**Resolution:** In Chrome, add `http://<your-machine-ip>:5173/` under
`chrome://flags/#unsafely-treat-insecure-origin-as-secure` and restart the browser. Alternatively, serve the
client over HTTPS.

## Evaluation bot cannot find server_configs/agent.yaml

**Cause:** When an evaluation bot starts, `SERVER_CONFIG_PATH` is relative to the current working directory.
The bot can report `FileNotFoundError` for `server_configs/agent.yaml`.

**Resolution:** Start both bots from the `evaluation/` directory.

## FileNotFoundError: Server configuration file not found at server_configs/agent.yaml appears

**Cause:** `SERVER_CONFIG_PATH` is resolved against the current working directory, not the script directory.

**Resolution:** Use `cd` to change to the directory that owns `server_configs/` first, or export an absolute
path.

## Gated repository returns 401 or 403

**Cause:** Hugging Face credentials are not available in the environment.

**Resolution:** Export `HF_TOKEN` before starting the server. Use `HF_HUB_CACHE` to relocate the cache.

## Initial failure is unclear from terminal output

**Cause:** The terminal can omit the context needed to identify the initial failure.

**Resolution:** Check the rotating log file that the server writes in addition to stderr. The example server
hardcodes the file name and level: `examples/generic_voice_agent/server/server.py` calls `setup_logging()` with
no arguments. This call defaults to `bot_server.log` at `DEBUG` with daily rotation in
`nemo_voice_agent/utils/misc.py`.

The path is relative to the directory from which the process starts. For the standard Quickstart invocation,
the path is `examples/generic_voice_agent/server/bot_server.log`. The `server.log_file` and `server.log_level`
keys are read only by the evaluation bots in `evaluation/bot_server.py`.

Rotated files have a timestamp suffix. For a run that just failed, check the newest
`bot_server.<timestamp>.log`, not only `bot_server.log`, which can belong to a still-running process. Read the
newest log around the first `ERROR` line. Most failures on this page are logged with the offending path, port,
or model ID.

## install.sh exits with "conda env ... is active"

**Cause:** The installer refuses to run inside a non-`base` conda environment because the conda toolchain
breaks source builds such as `cdifflib`.

**Resolution:** Run `conda deactivate`, and then run `bash install.sh` again.

## Microphone permission prompt does not appear and the volume meter stays flat

**Cause:** The page is not a secure context, or microphone permission was previously denied for the site.

**Resolution:** Serve the client over HTTPS, or add `http://<your-machine-ip>:5173/` under
`chrome://flags/#unsafely-treat-insecure-origin-as-secure` in Chrome and restart the browser. Then reset the
site's microphone permission in the browser settings.

## Model loads on the wrong GPU

**Cause:** Speech-to-text (STT), text-to-speech (TTS), and the Hugging Face LLM each have their own device key.
Diarization does not. `build_diar` passes `stt.device` to the service, so the `diar.device` entry in the YAML is
ignored and diarization always loads onto the STT device.

**Resolution:** Set `stt.device`, `tts.device`, and `llm.device` for the HF backend to distinct values such as
`cuda:0` and `cuda:1`. Diarization has no effective device key and loads onto the STT device.

## Model never calls a configured tool

**Cause:** Tools require `llm.type: vllm` with the model's vLLM tool parser flags, or `llm.type: nvidia`.
`server.py` gates registration only on `llm.enable_tool_calling`.

**Resolution:** Configure a supported backend and its required parser. For more information, refer to
[Tool calling](../build-voice-agents/tools/tool-calling.md) and
[vLLM backend](../build-voice-agents/model-serving/vllm.md).

## Model reasoning is read aloud

**Cause:** The reasoning span is not removed before TTS.

**Resolution:** Set `tts.think_tokens` to the model's think-token pair, or use a vLLM `--reasoning-parser`. For
more information, refer to [Reasoning mode](../about/core-concepts/language-models/reasoning.md).

## Model reports I/O or network errors while loading from Hugging Face

**Cause:** Streaming the repository at startup fails.

**Resolution:** Download the repository first and point the configuration to the local directory. For example,
run `huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir <local_path>`, and then set
`llm.model: <local_path>`. The same resolution applies to TTS models.

## npm run dev reports node:internal/errors:496

**Cause:** `node_modules` is partially installed or stale.

**Resolution:** Remove `examples/generic_voice_agent/client/node_modules`, run `npm install`, and then run
`npm run dev` again.

## npm run dev reports SyntaxError: Unexpected reserved word

**Cause:** Node.js is too old for the toolchain. Vite 6 supports Node 18, 20, and 22 or newer.

**Resolution:** Upgrade Node.js, and then run `npm install` again.

## Port 5173 is unavailable

**Cause:** Vite's port is pinned in the client `vite.config.js`.

**Resolution:** Change the `port` value in that file.

## Reasoning stays disabled when llm.enable_reasoning: true

**Cause:** An explicit `llm.model_config:` short-circuits the model-registry lookup that swaps to the sibling
`*_think.yaml` configuration.

**Resolution:** Point `model_config:` at the `*_think.yaml` file directly. For more information, refer to
[Reasoning mode](../about/core-concepts/language-models/reasoning.md).

## Remote browser loads the page but cannot open the WebSocket

**Cause:** `/connect` returns a URL built from `SERVER_PUBLIC_HOST`, which defaults to `127.0.0.1`, the loopback
address of the browser machine.

**Resolution:** Export `SERVER_PUBLIC_HOST` with the server IP address or hostname before starting the server.
Use `WEBSOCKET_SCHEME=wss` behind Transport Layer Security (TLS) termination.

## Second browser tab closes immediately with code 1013

**Cause:** The server accepts one client at a time. It rejects a new connection while keeping the existing
session.

**Resolution:** Close the first tab, and then reconnect. The LLM context is preserved across reconnects.

## Short acknowledgements such as "uh-huh" interrupt the bot

**Cause:** Backchannel filtering is disabled.

**Resolution:** Set `turn_taking.backchannel_phrases_path` to a phrase list instead of `null`.

## Speaker labels flip between turns or remain the same for every turn

**Cause:** Diarization can struggle when voices are similar or accents are underrepresented in its training
data.

**Resolution:** Lower `diar.threshold` to raise sensitivity, or disable diarization with
`diar.enabled: false`. For more information, refer to
[Diarization](../about/core-concepts/speech-pipeline/diarization.md).

## Startup reports OSError: [Errno 98] Address already in use

**Cause:** A previous server or another application still holds port 8765 or 7860.

**Resolution:** Run `lsof -i :8765` and `lsof -i :7860`, and then kill the stale process or export different
`WEBSOCKET_PORT` and `FASTAPI_PORT` values.

## tau2 scenario disconnects with WebSocket close code 1009

**Cause:** A database (DB) payload is inlined into a frame larger than the transport message cap.

**Resolution:** Seed the DB by path with `db_path` instead of inline content. The bot resolves it server-side.

## Transcripts are garbled in a noisy room

**Cause:** The shipped automatic speech recognition (ASR) and diarization models are not noise-robust.

**Resolution:** Use a noise-canceling microphone or a quieter environment. For more information, refer to
[ASR](../about/core-concepts/speech-pipeline/asr.md).

## vLLM is running but the agent still reports errors

**Cause:** `llm.base_url` points to a different port, or vLLM is serving a model ID that differs from
`llm.model`.

**Resolution:** Run `curl http://localhost:8000/v1/models`, and then align `llm.model` and `llm.base_url` with
the values that vLLM reports.

## vLLM log reports "doesn't seem to be supported ... using HuggingFace as the backend"

**Cause:** `llm.type: auto` cannot build a vLLM model configuration for the repository and falls back to `hf`.

**Resolution:** Set `type: vllm` explicitly in the model configuration to force vLLM, or use the HF backend.
The HF backend does not support tool calling.

## The issue persists after applying a resolution

**Next steps:** If the preceding resolutions do not fix the issue, use these diagnostic options:

- Turn on audio capture to hear exactly what the pipeline received. Refer to
  [Audio logging](../build-voice-agents/configure/audio-logging.md).
- If you are on the hosted backend rather than local models, check the key and endpoint requirements in
  [NVIDIA NIM](../build-voice-agents/model-serving/nvidia-nim.md).

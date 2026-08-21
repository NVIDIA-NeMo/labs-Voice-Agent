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

The following issues might arise when you run NeMo Labs Voice Agent. The issue sections are arranged
alphabetically by the symptom that you observe, followed by additional diagnostic options.

## A change to `default.yaml` has no effect

The model sub-YAML referenced by `model_config:` is merged over the top-level configuration, so the model
configuration wins. Edit the key in the model configuration instead, such as
`server_configs/llm_configs/nemotron_nano_v3.yaml`.

`llm.type` accepts `auto`, `hf`, `vllm`, or `nvidia`. For the full precedence model, refer to
[Server configuration](../build-voice-agents/configure/server-config.md).

## A dependency fails to compile during `uv sync`

This issue occurs when a C toolchain or Python headers are missing. Install `build-essential` and
`python3-dev`, or run `bash install.sh`, which installs them. For more information, refer to
[Installation](../get-started/installation.md).

## Agent quietly starts a second vLLM on port 8001

When `start_vllm_on_init: true`, this symptom occurs if the server on the requested port is already serving a
different model ID. The launcher scans forward for a free port. Kill the stale server with `lsof -i :8000`, or
set `llm.model` to exactly the ID that the running server reports.

## Bot interrupts too eagerly or waits too long before replying

This symptom is caused by VAD end-of-turn timing. Tune `vad.stop_secs`, which controls the silence required to
end a turn, along with `vad.confidence` and `vad.min_volume`. For more information, refer to
[Turn taking](../about/core-concepts/speech-pipeline/turn-taking.md).

## Bot never greets and the log reports a connection error for `http://localhost:8000/v1`

The shipped default, `server_configs/llm_configs/nemotron_nano_v3.yaml`, sets
`start_vllm_on_init: false`, so the voice agent does not launch vLLM. It also does not probe the endpoint at
startup. The server starts successfully, and the failure appears only on the first LLM turn after a browser
connects.

Start vLLM in a separate terminal with the flags from that file's `vllm_server_params`, wait for it to report
ready, then connect again. Check that vLLM is serving before you connect:

```bash
curl -s http://localhost:8000/v1/models
```

## Both evaluation bots claim the same port

`bot_server.py` is the same script for both roles and defaults to `WEBSOCKET_PORT=8765`. Give each role its
own `WEBSOCKET_PORT` and `FASTAPI_PORT`, as shown in
[Evaluation quickstart](../evaluate/run-evaluations/quickstart.md).

## Client cannot reach `/connect`

The browser client connects to the FastAPI app on port `7860` on the same host from which it loaded the page.
Keep `FASTAPI_PORT` at `7860`, or change the port in the client server configuration at
`examples/generic_voice_agent/client/src/app.ts`.

The defaults in `examples/generic_voice_agent/server/server.py` are `SERVER_HOST=0.0.0.0`,
`WEBSOCKET_PORT=8765`, `FASTAPI_PORT=7860`, `SERVER_PUBLIC_HOST=127.0.0.1`, and `WEBSOCKET_SCHEME=ws`. For more
information, refer to [Environment variables](../reference/runtime/environment.md).

## `Error connecting: Cannot read properties of undefined (reading 'enumerateDevices')` appears

This error appears when `navigator.mediaDevices` is unavailable because the page is not a secure context, such
as plain HTTP on a non-loopback host. In Chrome, add `http://<your-machine-ip>:5173/` under
`chrome://flags/#unsafely-treat-insecure-origin-as-secure` and restart the browser. Serving the client over HTTPS
also works.

## Evaluation bot cannot find `server_configs/agent.yaml`

When starting an evaluation bot, you might see `FileNotFoundError` for `server_configs/agent.yaml` because
`SERVER_CONFIG_PATH` is relative to the current working directory. Start both bots from the `evaluation/`
directory.

## `FileNotFoundError: Server configuration file not found at server_configs/agent.yaml` appears

This error occurs because `SERVER_CONFIG_PATH` is resolved against the current working directory, not the
script directory. Use `cd` to change to the directory that owns `server_configs/` first, or export an absolute
path.

## Gated repository returns 401 or 403

This error occurs when HuggingFace credentials are not available in the environment. Export `HF_TOKEN` before
starting the server. Use `HF_HUB_CACHE` to relocate the cache.

## Initial failure is unclear from terminal output

The server writes a rotating log file in addition to stderr. The example server hardcodes the file name and
level: `examples/generic_voice_agent/server/server.py` calls `setup_logging()` with no arguments, which defaults
to `bot_server.log` at `DEBUG` with daily rotation in `nemo_voice_agent/utils/misc.py`. The path is relative to
the directory from which the process starts. For the standard Quickstart invocation, the path is
`examples/generic_voice_agent/server/bot_server.log`. The `server.log_file` and `server.log_level` keys are read
only by the evaluation bots in `evaluation/bot_server.py`.

Rotated files have a timestamp suffix. For a run that just failed, check the newest
`bot_server.<timestamp>.log`, not only `bot_server.log`, which might belong to a still-running process. Re-read
the newest log around the first `ERROR` line; most failures on this page are logged with the offending path,
port, or model ID.

## `install.sh` exits with "conda env ... is active"

The installer refuses to run inside a non-`base` conda environment because the conda toolchain breaks source
builds such as `cdifflib`. Run `conda deactivate`, then run `bash install.sh` again.

## Microphone permission prompt does not appear and the volume meter stays flat

This symptom occurs when the page is not a secure context or microphone permission was previously denied for
the site. Serve the client over HTTPS, or add `http://<your-machine-ip>:5173/` under
`chrome://flags/#unsafely-treat-insecure-origin-as-secure` in Chrome and restart the browser. Then reset the
site's microphone permission in the browser settings.

## Model loads on the wrong GPU

STT, TTS, and the HuggingFace LLM each have their own device key; diarization does not. Set `stt.device`,
`tts.device`, and `llm.device` for the HF backend to distinct values such as `cuda:0` and `cuda:1`.
Diarization has no effective device key: `build_diar` passes `stt.device` to the service, so the `diar.device`
entry in the YAML is ignored and diarization always loads onto the STT device.

## Model never calls a configured tool

Tools require `llm.type: vllm` with the model's vLLM tool parser flags, or `llm.type: nvidia`.
`server.py` gates registration only on `llm.enable_tool_calling`. For more information, refer to
[Tool calling](../build-voice-agents/tools/tool-calling.md) and
[vLLM backend](../build-voice-agents/model-serving/vllm.md).

## Model reasoning is read aloud

The reasoning span is not being removed before TTS. Set `tts.think_tokens` to the model's think-token pair, or
use a vLLM `--reasoning-parser`. For more information, refer to
[Reasoning mode](../about/core-concepts/language-models/reasoning.md).

## Model reports I/O or network errors while loading from HuggingFace

This error occurs when streaming the repository at startup fails. Download it first and point the configuration
to the local directory. For example, run
`huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir <local_path>`, then set
`llm.model: <local_path>`. The same resolution applies to TTS models.

## `npm run dev` reports `node:internal/errors:496`

This error occurs when `node_modules` is partially installed or stale. Remove
`examples/generic_voice_agent/client/node_modules`, run `npm install`, then run `npm run dev` again.

## `npm run dev` reports `SyntaxError: Unexpected reserved word`

This error occurs when Node.js is too old for the toolchain. Vite 6 supports Node 18, 20, and 22 or newer.
Upgrade Node.js, then run `npm install` again.

## Port 5173 is unavailable

This issue occurs because Vite's port is pinned in the client `vite.config.js`. Change the `port` value in that
file.

## Reasoning stays disabled when `llm.enable_reasoning: true`

An explicit `llm.model_config:` short-circuits the model-registry lookup that swaps to the sibling
`*_think.yaml` configuration. Point `model_config:` at the `*_think.yaml` file directly. For more information,
refer to [Reasoning mode](../about/core-concepts/language-models/reasoning.md).

## Remote browser loads the page but cannot open the WebSocket

`/connect` returns a URL built from `SERVER_PUBLIC_HOST`, which defaults to `127.0.0.1`, the loopback address of
the browser machine. Export `SERVER_PUBLIC_HOST` with the server IP address or hostname before starting the
server. Use `WEBSOCKET_SCHEME=wss` behind TLS termination.

## Second browser tab closes immediately with code `1013`

The server accepts one client at a time. A new connection is rejected while the existing session is kept.
Close the first tab, then reconnect. The LLM context is preserved across reconnects.

## Short acknowledgements such as "uh-huh" interrupt the bot

Backchannel filtering is disabled. Set `turn_taking.backchannel_phrases_path` to a phrase list instead of
`null`.

## Speaker labels flip between turns or remain the same for every turn

Diarization can struggle when voices are similar or accents are underrepresented in its training data. Lower
`diar.threshold` to raise sensitivity, or disable diarization with `diar.enabled: false`. For more information,
refer to [Diarization](../about/core-concepts/speech-pipeline/diarization.md).

## Startup reports `OSError: [Errno 98] Address already in use`

A previous server or another application still holds port 8765 or 7860. Run `lsof -i :8765` and
`lsof -i :7860`, then kill the stale process or export different `WEBSOCKET_PORT` and `FASTAPI_PORT` values.

## tau2 scenario disconnects with WebSocket close code `1009`

This error occurs when a DB payload is inlined into a frame larger than the transport message cap. Seed the DB
by path with `db_path` instead of inline content; the bot resolves it server-side.

## Transcripts are garbled in a noisy room

The shipped ASR and diarization models are not noise-robust. Use a noise-canceling microphone or a quieter
environment. For more information, refer to [ASR](../about/core-concepts/speech-pipeline/asr.md).

## vLLM is running but the agent still reports errors

This symptom occurs when `llm.base_url` points to a different port or vLLM is serving a model ID that differs
from `llm.model`. Run `curl http://localhost:8000/v1/models`, then align `llm.model` and `llm.base_url` with the
values that vLLM reports.

## vLLM log reports "doesn't seem to be supported ... using HuggingFace as the backend"

This message appears when `llm.type: auto` cannot build a vLLM model configuration for the repository and
falls back to `hf`. Set `type: vllm` explicitly in the model configuration to force vLLM, or use the HF backend,
which does not support tool calling.

## The issue persists after applying a resolution

If the preceding resolutions do not fix the issue, use the following existing diagnostic options:

- Turn on audio capture to hear exactly what the pipeline received. See [Audio logging](../build-voice-agents/configure/audio-logging.md).
- If you are on the hosted backend rather than local models, check the key and endpoint requirements in
  [NVIDIA NIM](../build-voice-agents/model-serving/nvidia-nim.md).

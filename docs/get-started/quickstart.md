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

Bring up NeMo Labs Voice Agent end to end and talk to it in a browser. This quickstart starts vLLM, the
agent server, and the web client, and then verifies the browser-to-agent connection.

## Prerequisites

Before you begin, prepare the installed project and the terminal sessions used by its three services.

- Complete [Prerequisites](prerequisites.md) and [Installation](installation.md).
- Verify that you can activate the project environment with `source .venv/bin/activate`.
- Open three terminals: one for vLLM, one for the agent server, and one for the web client.

## Quickstart Steps

Complete these four steps in order to start the services and connect from the browser.

### Step 1 — Start vLLM yourself

This is the most common first-run failure. The shipped default LLM config
(`examples/generic_voice_agent/server/server_configs/llm_configs/nemotron_nano_v3.yaml`) sets
`start_vllm_on_init: false`, so `python server.py` alone will **not** bring up a model — the agent server
starts and then fails to reach the OpenAI-compatible endpoint at `http://localhost:8000/v1`.

In the first terminal, serve the default model with the flags from that file's `vllm_server_params`
(the YAML is the authoritative source if this snippet drifts):

```bash
source .venv/bin/activate
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
    --trust-remote-code --tensor-parallel-size 1 --enable-prefix-caching \
    --max-num-seqs 1 --gpu-memory-utilization 0.8 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3
```

Wait until vLLM logs that it is serving before moving on. To have the agent server launch vLLM for you
instead, set `start_vllm_on_init: true` in the model config — see [vLLM backend](../build-voice-agents/model-serving/vllm.md).

Remember that the model sub-YAML **overrides** `default.yaml`, not the other way round. That is why
`default.yaml`'s `llm.type: auto` ends up as `vllm` at runtime. Details in
[Server configuration](../build-voice-agents/configure/server-config.md).

### Step 2 — Start the agent server

In the second terminal:

```bash
source .venv/bin/activate
cd examples/generic_voice_agent/server/
python server.py
```

The server loads models on startup (ASR, diarization, TTS, plus the vLLM client), so the first run takes a
while and downloads weights from HuggingFace. It stays up until you press Ctrl+C.

Environment variables read by `examples/generic_voice_agent/server/server.py` (a `.env` file in the working
directory is loaded and takes precedence over the shell environment):

| Variable | Default | Purpose |
| --- | --- | --- |
| `SERVER_HOST` | `0.0.0.0` | Bind address for both the FastAPI app and the WebSocket server. |
| `FASTAPI_PORT` | `7860` | Port serving the `/connect` handshake endpoint. |
| `WEBSOCKET_PORT` | `8765` | Port carrying the real-time audio WebSocket. |
| `SERVER_PUBLIC_HOST` | `127.0.0.1` | Host name or IP that `/connect` advertises to the browser. |
| `WEBSOCKET_SCHEME` | `ws` | `ws` or `wss`; use `wss` behind TLS termination. |
| `SERVER_CONFIG_PATH` | unset (uses `server_configs/default.yaml`) | Alternate top-level YAML config. Resolved against the current directory. |

`HF_TOKEN` and `HF_HUB_CACHE` are honored by the HuggingFace libraries if you need a gated model or a custom
cache location. The full list lives in [Environment variables](../reference/runtime/environment.md).

### How the two ports fit together

The browser talks to two different ports, and mixing them up is the second-most-common first-run problem.

1. The client POSTs to `http://<server-host>:7860/connect` (FastAPI). The handler in
   `nemo_voice_agent/pipecat/bot_server.py` returns a JSON body with a single `ws_url` key.
2. That URL is assembled by `build_websocket_url` in `nemo_voice_agent/utils/websocket_url.py` from
   `WEBSOCKET_SCHEME`, `SERVER_PUBLIC_HOST`, and `WEBSOCKET_PORT` — for the defaults, `ws://127.0.0.1:8765`.
3. The client then opens that WebSocket and all audio flows over port 8765. Port 7860 is not used again.

The consequence: `SERVER_PUBLIC_HOST` must be reachable **from the browser**, not from the server. Leaving it
at `127.0.0.1` while browsing from another machine produces a successful `/connect` followed by a WebSocket
that never connects. Export the machine's hostname or IP before starting the server:

```bash
export SERVER_PUBLIC_HOST="10.0.0.5"   # hostname or IP the browser will dial
```

The client derives its own base URL from the browser address bar, so
`examples/generic_voice_agent/client/src/app.ts` needs no edit for remote access.

Only one client at a time is supported. While a client is connected, the transport rejects a second
connection with WebSocket close code `1013` and keeps the incumbent; the next client can connect once the
current one disconnects.

### Step 3 — Start the web client

In the third terminal, on the server machine:

```bash
cd examples/generic_voice_agent/client
npm install
npm run dev
```

Vite prints the address it is listening on. It binds `0.0.0.0:5173` by default; change the `port` value in
`examples/generic_voice_agent/client/vite.config.js` if 5173 is taken.

### Step 4 — Connect from the browser

Open `http://<your-machine-ip>:5173/` (or whatever Vite printed).

Microphone capture requires a secure context. In Chrome, add that exact origin to
`chrome://flags/#unsafely-treat-insecure-origin-as-secure` and restart the browser, otherwise the
**Connect** button fails when it asks for microphone access.

Leave the **Server** dropdown on *WebSocket Server* — that is the option pointing at `/connect` on port 7860.
Then press **Connect** and grant microphone permission. The bot speaks first: the server queues an initial
LLM run when the client reports ready, so you should hear the greeting from the system prompt in
`default.yaml` ("Hi, I'm Lisa, your helpful AI assistant...") within a few seconds. Start talking when it
finishes.

## Controls

Use the browser controls to manage the active session and inspect its state.

| Control | Effect |
| --- | --- |
| **Connect** / **Disconnect** | Open or close the WebSocket session. The server keeps running and accepts the next connection. |
| **Mute** | Toggles the local microphone track only (client-side); no audio reaches the server while muted. |
| **Reset** | Sends the `reset` RTVI client request. The server restores the LLM context to the original system prompt and resets the ASR, TTS, diarization, and turn-taking services — so speaker identities are re-learned from scratch. |
| Microphone Volume bar | Local input level, driven by a browser `AnalyserNode`. Useful for confirming the mic is live. |
| Debug Info panel | Timestamped log with user transcripts in blue and bot responses in green. |

## Troubleshooting

Server-side logs are written to `bot_server.log` in the directory you launched `server.py` from (rotated
daily; check the newest `bot_server.<timestamp>.log` for a run that already rotated). The name, level, and
daily rotation are hardcoded: `server.py` calls `setup_logging()` with no arguments, which defaults to
`bot_server.log` at `DEBUG` with `rotation="1 day"` in `nemo_voice_agent/utils/misc.py`. Editing
`server.log_file` / `server.log_level` in `default.yaml` does not change them — only the evaluation bots in
`evaluation/bot_server.py` read those keys.

Quick checks when nothing is heard:

```bash
# Is the LLM endpoint up?
curl -s http://localhost:8000/v1/models

# Does the handshake return a URL the browser can reach?
curl -s -X POST http://localhost:7860/connect
```

If the second command returns `ws://127.0.0.1:8765` but you are browsing from another machine, go back to
`SERVER_PUBLIC_HOST`. More failure modes are collected in [Troubleshooting](../troubleshooting/index.md).

## Next Steps

After the first browser session works, explore the pipeline design, configuration, tools, or evaluation
workflow.

| Continue with | Purpose |
| --- | --- |
| [Architecture](../about/architecture.md) | Learn what the pipeline does with your audio. |
| [Server Configuration](../build-voice-agents/configure/server-config.md) | Swap models, prompts, and VAD settings. |
| [Tool Calling](../build-voice-agents/tools/tool-calling.md) | Let the agent call functions. |
| [Evaluation Quickstart](../evaluate/run-evaluations/quickstart.md) | Score the agent on benchmark scenarios. |

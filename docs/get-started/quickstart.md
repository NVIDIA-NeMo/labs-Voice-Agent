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

Start NeMo Labs Voice Agent and talk to it in a browser. This quickstart launches vLLM, the agent server,
and the web client, then verifies the browser-to-agent connection.

## Prerequisites

Before you begin, prepare the installed project and the terminal sessions used by its three services:

- Complete [Prerequisites](prerequisites.md) and [Installation](installation.md).
- Verify that you can activate the project environment with `source .venv/bin/activate`.
- Open three terminals: one for vLLM, one for the agent server, and one for the web client.

## Quickstart Steps

Complete these four steps in order to start the services and connect from the browser.

### Step 1: Start vLLM Yourself

This is the most common first-run failure. The shipped default large language model (LLM) configuration
(`examples/generic_voice_agent/server/server_configs/llm_configs/nemotron_nano_v3.yaml`) sets
`start_vllm_on_init: false`. As a result, `python server.py` does not start a model. The agent server starts
and then fails to reach the OpenAI-compatible endpoint at `http://localhost:8000/v1`.

In the first terminal, serve the default model with the flags from that file's `vllm_server_params`. The
YAML file is the authoritative source if this example differs from the current configuration:

```bash
source .venv/bin/activate
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
    --trust-remote-code --tensor-parallel-size 1 --enable-prefix-caching \
    --max-num-seqs 1 --gpu-memory-utilization 0.8 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3
```

Wait until the vLLM logs confirm that it is serving before continuing. To let the agent server launch vLLM,
set `start_vllm_on_init: true` in the model configuration. Refer to
[Serving with vLLM](../build-voice-agents/model-serving/vllm.md).

The model sub-YAML **overrides** `default.yaml`, not the other way around. This override changes
`default.yaml`'s `llm.type: auto` to `vllm` at runtime. For details, refer to
[Server configuration](../build-voice-agents/configure/server-config.md).

### Step 2: Start the Agent Server

In the second terminal:

```bash
source .venv/bin/activate
cd examples/generic_voice_agent/server/
python server.py
```

On first startup, the server downloads weights from Hugging Face. It loads automatic speech recognition
(ASR), diarization, text-to-speech (TTS), and the vLLM client. It runs until you press Ctrl+C.

`examples/generic_voice_agent/server/server.py` reads the following environment variables. The server loads
a `.env` file from the working directory, and its values take precedence over the shell environment:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SERVER_HOST` | `0.0.0.0` | Bind address for both the FastAPI app and the WebSocket server. |
| `FASTAPI_PORT` | `7860` | Port serving the `/connect` handshake endpoint. |
| `WEBSOCKET_PORT` | `8765` | Port carrying the real-time audio WebSocket. |
| `SERVER_PUBLIC_HOST` | `127.0.0.1` | Host name or IP that `/connect` advertises to the browser. |
| `WEBSOCKET_SCHEME` | `ws` | `ws` or `wss`. Use `wss` behind TLS termination. |
| `SERVER_CONFIG_PATH` | unset (uses `server_configs/default.yaml`) | Alternate top-level YAML config. Resolved against the current directory. |

The Hugging Face libraries honor `HF_TOKEN` and `HF_HUB_CACHE` if you need a gated model or a custom cache
location. For the complete list, refer to [Environment Variables](../reference/runtime/environment.md).

### How the Two Ports Fit Together

The browser uses two ports. Confusing their roles is the second-most-common first-run problem.

1. The client sends a POST request to `http://<server-host>:7860/connect` (FastAPI). The handler in
   `nemo_voice_agent/pipecat/bot_server.py` returns a JSON body with a single `ws_url` key.
2. That URL is assembled by `build_websocket_url` in `nemo_voice_agent/utils/websocket_url.py` from
   `WEBSOCKET_SCHEME`, `SERVER_PUBLIC_HOST`, and `WEBSOCKET_PORT`. With the defaults, the URL is
   `ws://127.0.0.1:8765`.
3. The client then opens that WebSocket and all audio flows over port 8765. Port 7860 is not used again.

`SERVER_PUBLIC_HOST` must be reachable **from the browser**, not only from the server. If you browse from
another machine, `127.0.0.1` produces a successful `/connect` request followed by a WebSocket connection
failure. Export the server machine's hostname or IP before starting the server:

```bash
export SERVER_PUBLIC_HOST="10.0.0.5"   # hostname or IP the browser will dial
```

The client derives its own base URL from the browser address bar, so
`examples/generic_voice_agent/client/src/app.ts` needs no edit for remote access.

The server supports one client at a time. While a client is connected, the transport rejects another
connection with WebSocket close code `1013` and keeps the active client. A different client can connect
after the active client disconnects.

### Step 3: Start the Web Client

In the third terminal, on the server machine:

```bash
cd examples/generic_voice_agent/client
npm install
npm run dev
```

Vite prints its listening address. It binds `0.0.0.0:5173` by default. If port 5173 is unavailable, change
the `port` value in `examples/generic_voice_agent/client/vite.config.js`.

### Step 4: Connect From the Browser

Open `http://<your-machine-ip>:5173/` (or whatever Vite printed).

Microphone capture requires a secure context. In Chrome, add that exact origin to
`chrome://flags/#unsafely-treat-insecure-origin-as-secure` and restart the browser. Otherwise, the
**Connect** button fails when it requests microphone access.

Leave the **Server** dropdown on **WebSocket Server**. This option points to `/connect` on port 7860.
Then press **Connect** and grant microphone permission. The bot speaks first: the server queues an initial
LLM run when the client reports ready. You hear the greeting from the system prompt in `default.yaml`:
"Hi, I'm Lisa, your helpful AI assistant..." Start talking after the greeting finishes.

## Controls

Use the browser controls to manage the active session and inspect its state:

| Control | Effect |
| --- | --- |
| **Connect** and **Disconnect** | Open or close the WebSocket session. The server keeps running and accepts the next connection. |
| **Mute** | Toggle the local microphone track. No audio reaches the server while muted. |
| **Reset** | Send the `reset` real-time voice inference (RTVI) client request. The server restores the LLM context to the original system prompt. It also resets the ASR, TTS, diarization, and turn-taking services, so the system learns speaker identities again. |
| **Microphone Volume** bar | View the local input level from a browser `AnalyserNode`. Use it to confirm that the microphone is active. |
| **Debug Info** panel | View a timestamped log with user transcripts in blue and bot responses in green. |

## Troubleshooting

The server writes logs to `bot_server.log` in the directory where you launched `server.py`. The log rotates
daily. For an earlier run, check the newest `bot_server.<timestamp>.log` file. The filename, level, and daily
rotation are hardcoded. `server.py` calls `setup_logging()` with no arguments, which defaults to
`bot_server.log` at `DEBUG` with `rotation="1 day"` in `nemo_voice_agent/utils/misc.py`. Editing
`server.log_file` and `server.log_level` in `default.yaml` does not change these values. Only the evaluation
bots in `evaluation/bot_server.py` read those keys.

Run these checks when you hear no audio:

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
workflow:

| Continue with | Purpose |
| --- | --- |
| [Architecture](../about/architecture.md) | Learn what the pipeline does with your audio. |
| [Server Configuration](../build-voice-agents/configure/server-config.md) | Configure models, prompts, and voice activity detection (VAD) settings. |
| [Tool Calling](../build-voice-agents/tools/tool-calling.md) | Let the agent call functions. |
| [Evaluation Quickstart](../evaluate/run-evaluations/quickstart.md) | Score the agent on benchmark scenarios. |

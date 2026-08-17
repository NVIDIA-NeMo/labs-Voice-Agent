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

# Client Protocol

Talking to NeMo Labs Voice Agent is a two-step handshake. A client first calls `POST /connect` on the
FastAPI port to learn where the media socket lives, then opens that WebSocket and exchanges
Protobuf-encoded frames — raw PCM audio in both directions plus JSON control messages in the RTVI
envelope.

The two servers are started together by `run_bot_with_fastapi` in `nemo_voice_agent/pipecat/bot_server.py`.

| Server | Default port | Env var | Carries |
| --- | --- | --- | --- |
| FastAPI | 7860 | `FASTAPI_PORT` | `POST /connect` discovery only |
| WebSocket | 8765 | `WEBSOCKET_PORT` | Audio frames and RTVI messages |

## Step 1: POST /connect

The endpoint takes no meaningful request body and returns one key:

```bash
curl -s -X POST http://127.0.0.1:7860/connect
# {"ws_url":"ws://127.0.0.1:8765"}
```

The URL is assembled by `build_websocket_url` in `nemo_voice_agent/utils/websocket_url.py` from three
server-side values — it never echoes anything the client sent:

| Env var | Default | Notes |
| --- | --- | --- |
| `WEBSOCKET_SCHEME` | `ws` | Must be `ws` or `wss`; anything else raises `ValueError` at request time. |
| `SERVER_PUBLIC_HOST` | `127.0.0.1` | Must be reachable **from the client machine**. If you pass a full URL, only the host part is kept; IPv6 literals are bracketed. |
| `WEBSOCKET_PORT` | `8765` | Port the transport listens on. |

The app is created with permissive CORS (`allow_origins=["*"]`), so a browser on another origin can call
it directly. The `/ws` route on the FastAPI app is an unimplemented stub — ignore it. Full variable list
is in [Environment variables](../reference/environment.md).

## Step 2: the WebSocket

The transport is pipecat's `SingleClientWebsocketServerTransport`, built by `build_ws_transport` in
`nemo_voice_agent/pipecat/services/nemo/builders.py` with a `ProtobufFrameSerializer`. Every WebSocket
message — in both directions — is a binary Protobuf `Frame` with a oneof:

| Oneof field | Direction | Meaning |
| --- | --- | --- |
| `audio` | both | `AudioRawFrame`: `audio` bytes, `sample_rate`, `num_channels`, `pts`. |
| `message` | both | `MessageFrame`: `data` is a JSON **string** holding an RTVI message. |
| `text` | both | Plain `TextFrame`. |
| `transcription` | both | `TranscriptionFrame` with `text`, `user_id`, `timestamp`. |
| `interruption` | both | Interruption signal. |

Audio is 16-bit signed little-endian PCM, mono. Inbound audio is expected at
`transport.audio_in_sample_rate` (default 16000). Outbound audio follows the TTS service's rate unless
you set `transport.audio_out_sample_rate`; the JavaScript transport plays back at 24 kHz by default.

### RTVI messages

Control traffic rides inside `MessageFrame.data` as a JSON object with `label`, `type`, `id`, and `data`
keys. Two exchanges matter to every client:

1. **`client-ready`** — the client must send it after connecting. The server replies with `bot-ready`,
   and, because the example server passes `talk_first=True`, queues an `LLMRunFrame` so the bot speaks
   first. A client that never sends `client-ready` will sit in silence.
2. **`client-message`** — a request/response call whose `data` is `{"t": "<type>", "d": {...}}`. The
   server answers with `server-response` (payload under `d`) or `error-response` on an unknown type or
   a raised exception. Dispatch is installed by `register_client_message_handlers` in
   `nemo_voice_agent/pipecat/processors/frameworks/rtvi_actions.py`.

The example server registers exactly one custom type, `reset`, which clears the conversation context
back to the original system prompt. The evaluation bots register five more; see
[RTVI control plane](rtvi-actions.md) and [RTVI message reference](../reference/rtvi-messages.md).

Server-to-client event messages (`user-transcription`, `bot-transcription`, `bot-llm-text`,
`bot-tts-text`, `bot-started-speaking`, `bot-stopped-speaking`, `metrics`, `server-message`, …) are
emitted by the `RTVIObserver` attached to the pipeline worker.

## Browser client

`examples/generic_voice_agent/client/` is a vanilla-TypeScript Vite app built on
`@pipecat-ai/client-js` and `@pipecat-ai/websocket-transport`. The whole connection is three calls in
`src/app.ts`:

```ts
const transport = new WebSocketTransport();
const client = new PipecatClient({ transport, enableMic: true, enableCam: false, callbacks });
await client.initDevices();
await client.connect({ endpoint: "http://<host>:7860/connect" });
```

`connect()` issues the `POST`, reads `ws_url` out of the JSON response, and hands it to the transport —
you never dial the socket yourself. The `reset` message goes through
`client.sendClientRequest("reset", {})`, which resolves with the handler's return value.

Run it with `npm install && npm run dev`. The dev server listens on port 5173 on all interfaces and
proxies `/connect` to `http://0.0.0.0:7860`. The demo page has a **Server** dropdown; leave it on
"WebSocket Server", which points at port 7860. Startup details are in the
[Quickstart](../get-started/quickstart.md).

## Non-browser client

Anything that speaks WebSocket works. `nemo_voice_agent/evaluation/bridge.py` is a complete
Python implementation — it drives both bots in the eval harness — and is the best reference. A minimal
client must:

1. `POST /connect` and read `ws_url` (or skip discovery and dial `ws://host:8765` directly).
2. Send a `client-ready` RTVI message and wait for `bot-ready`.
3. Stream 16 kHz mono PCM as serialized `OutputAudioRawFrame`s, paced in real time.
4. Deserialize incoming frames: audio arrives as `InputAudioRawFrame`, RTVI JSON as
   `InputTransportMessageFrame` with the parsed dict on `.message`.

```python
import asyncio
import json

import requests
import websockets
from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.serializers.protobuf import MessageFrame, ProtobufFrameSerializer


async def main() -> None:
    serializer = ProtobufFrameSerializer()
    ws_url = requests.post("http://127.0.0.1:7860/connect", timeout=10).json()["ws_url"]

    async with websockets.connect(ws_url, ping_timeout=None) as ws:
        ready = {
            "label": "rtvi-ai",
            "type": "client-ready",
            "id": "client-ready-1",
            "data": {"version": "1.1.0", "about": {"library": "my-client", "library_version": "0.1.0"}},
        }
        await ws.send(await serializer.serialize(MessageFrame(data=json.dumps(ready))))

        # Then, per 20 ms tick: send microphone PCM.
        chunk = b"\x00" * 640  # 320 samples of int16 at 16 kHz
        await ws.send(await serializer.serialize(
            OutputAudioRawFrame(audio=chunk, sample_rate=16000, num_channels=1)
        ))

        async for message in ws:
            frame = await serializer.deserialize(message)
            print(type(frame).__name__)


asyncio.run(main())
```

Send `OutputAudioRawFrame` even though the server sees it as input — that is the direction the
serializer's tables expect.

## Connection rules

- **One client at a time.** While a client is connected, a second connection is closed immediately with
  WebSocket code `1013` and reason `Server already has a connected client`. The incumbent keeps talking.
- **The pipeline outlives the connection.** On disconnect the server deliberately does not end the
  pipeline task — the WebSocket listener lives inside the input transport, so ending the task would
  leave `/connect` advertising a dead port. The next client is accepted into the same running pipeline.
- **Context survives reconnects.** Because the pipeline persists, so does the LLM conversation history.
  Send the `reset` message to clear it explicitly.

## Next

- [RTVI control plane](rtvi-actions.md) — add your own client message types.
- [RTVI message reference](../reference/rtvi-messages.md) — every message the server understands.
- [How it works](../get-started/architecture.md) — where the transport sits in the pipeline.
- [Troubleshooting](../get-started/troubleshooting.md) — connection failures and what causes them.

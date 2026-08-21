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

# Environment Variables

NeMo Labs Voice Agent keeps almost all behavior in YAML (see [Server Configuration](../../build-voice-agents/configure/server-config.md)).
Environment variables cover only the things YAML cannot: process wiring (host, ports, which config to load),
credentials, and fixture locations. This page lists every variable the repository's own code reads, with its
default and the file that reads it.

## How variables are loaded

Both server entrypoints call `load_dotenv(override=True)` at import time, as does the evaluation LLM judge in
`nemo_voice_agent/evaluation/utils.py`. Two consequences:

- A `.env` file is discovered by python-dotenv's `find_dotenv()`, which walks up from the directory of the
  calling module to the filesystem root. `evaluation/bot_server.py` therefore picks up `evaluation/.env`.
- Because `override=True` is passed, values in `.env` **win over** variables you already exported in the shell.
  If an `export` appears to have no effect, check for a stale `.env` above it.

`evaluation/.env.example` is a starting template — copy it to `evaluation/.env` and fill it in. It also carries
placeholder keys (`OPENAI_API_KEY`, `ELEVENLABS_API_KEY`, `HUGGINGFACE_TOKEN`) that no shipped code path in this
repository reads; only the variables tabulated below are consumed.

## General Environment Variables

The following groups cover the environment variables that the runtime and evaluation harness read directly.

### Server and transport

Read at module import in `examples/generic_voice_agent/server/server.py` and, for the evaluation harness, in
`evaluation/bot_server.py`. The two files agree on every default except `SERVER_CONFIG_PATH`.

| Variable | Default (example server) | Default (`evaluation/bot_server.py`) | Purpose |
| --- | --- | --- | --- |
| `SERVER_HOST` | `0.0.0.0` | `0.0.0.0` | Bind address for both the WebSocket server and the FastAPI app. |
| `WEBSOCKET_PORT` | `8765` | `8765` | Port the Pipecat WebSocket transport listens on. Parsed with `int()`. |
| `FASTAPI_PORT` | `7860` | `7860` | Port for the FastAPI app that serves `POST /connect`. Parsed with `int()`. |
| `SERVER_CONFIG_PATH` | unset (`None`) | `server_configs/agent.yaml` | Path to the top-level YAML config. |
| `SERVER_PUBLIC_HOST` | `127.0.0.1` | `127.0.0.1` | Hostname or IP that `/connect` hands back to the browser client. |
| `WEBSOCKET_SCHEME` | `ws` | `ws` | Scheme in the URL returned by `/connect`. Only `ws` and `wss` are accepted. |

### `SERVER_CONFIG_PATH`

The value is passed straight to `ConfigManager` and checked with `os.path.exists`, so a relative value resolves
against the **current working directory**, not the script directory. Always `cd` to the directory the path is
written relative to:

```bash
export SERVER_CONFIG_PATH="server_configs/default_nvidia.yaml"
cd examples/generic_voice_agent/server/
python server.py
```

When it is unset, the example server falls back to `server_configs/default.yaml` resolved against the server
directory, so a bare `python server.py` works from anywhere. `evaluation/bot_server.py` has no such fallback —
its default is the relative string `server_configs/agent.yaml`, which is why every evaluation command starts
with `cd evaluation`. `SERVER_CONFIG_PATH` is also how one script serves both evaluation roles:
`evaluation/run_agent.sh` exports `server_configs/agent.yaml` on port 8765 and `evaluation/run_user.sh` exports
`server_configs/user.yaml` on port 8766.

### `SERVER_PUBLIC_HOST` and `WEBSOCKET_SCHEME`

These two never affect what the server binds — only the URL string `POST /connect` returns. Set
`SERVER_PUBLIC_HOST` to your machine's routable hostname or IP when the browser runs on a different machine,
and set `WEBSOCKET_SCHEME=wss` when a reverse proxy terminates TLS in front of the WebSocket port.

Both are normalized by `build_websocket_url` in `nemo_voice_agent/utils/websocket_url.py` each time `/connect`
is served. It strips a scheme prefix from the host if you include one, brackets bare IPv6 literals, and raises
`ValueError` on an empty host or on a scheme other than `ws`/`wss` — surfacing as a failed `/connect` request
rather than a startup crash.

## Credentials

The following variables provide credentials or configure model-library authentication and caches.

| Variable | Read by | Default | Notes |
| --- | --- | --- | --- |
| `NVIDIA_API_KEY` | STT, TTS, and LLM builders in `nemo_voice_agent/pipecat/services/nemo/` | falls back to the `api_key` key in YAML, else the literal string `"None"` | Required by the hosted NVIDIA backends. The LLM builder raises `ValueError` at startup when `llm.base_url` is `https://integrate.api.nvidia.com/v1` and no key resolves. |
| `NVIDIA_INFERENCE_API_KEY` | LLM builder only | same fallback chain | Replaces `NVIDIA_API_KEY` when `llm.base_url` is `https://inference-api.nvidia.com/v1`; missing value raises `ValueError`. |
| `JUDGE_API_KEY` | `LLMJudge` in `nemo_voice_agent/evaluation/utils.py` | none | Only read when `--judge-api-key` is not passed. The variable *name* is itself configurable with `--judge-api-key-name` (default `JUDGE_API_KEY`); `LLMJudge`'s own constructor default for `api_key_name` is `API_KEY`, which the runner overrides. |
| `HF_TOKEN` | HuggingFace libraries (not this repo's code) | none | Needed for gated repos such as `meta-llama/Llama-3.1-8B-Instruct`. Request access on the model page first. |
| `HF_HUB_CACHE` | HuggingFace libraries (not this repo's code) | HuggingFace default | Relocates the downloaded-model cache off the home directory. |

`--judge-api-key` is redacted before the runner writes its argument snapshot to disk, so prefer passing the key
through the environment or `.env` either way. See [Evaluation CLI](../evaluation/eval-cli.md) for the full judge flag list and
[NVIDIA NIM and Riva](../../build-voice-agents/model-serving/nvidia-nim.md) for the hosted-backend setup.

## Evaluation data

The following variable overrides where the evaluation harness resolves its packaged fixture data.

| Variable | Read by | Default | Purpose |
| --- | --- | --- | --- |
| `EVAL_DATA_ROOT` | `get_eval_data_root()` in `nemo_voice_agent/evaluation/__init__.py` | the packaged `nemo_voice_agent/evaluation/data/` directory | Overrides the fixture root for scenario DBs, policies, and task indexes. |

`get_eval_data_root()` is a function rather than a module constant, so changing the variable after import takes
effect — which is what lets the bridge process and each bot process resolve the same relative `db_path` to
different absolute roots. Fixture paths stored in `shared_state_init` are always relative to this root. If a
scenario fails to seed its DB, the `apply_initialization` handler logs the resolved root in the error message.
See [Evaluation Overview](../../evaluate/index.md).

## Packaging

The following variable controls whether the package version includes source-control metadata.

| Variable | Read by | Default | Purpose |
| --- | --- | --- | --- |
| `NO_VCS_VERSION` | `nemo_voice_agent/package_info.py` | `1` | Set to `0` to append the short git SHA to `__version__` (for example `0.1.0+a47d81d`). |

The sense is inverted: the default `1` means "no VCS version", and `0` enables the git lookup. The value is
parsed with `int()`, so a non-numeric string raises at import.

## Set by the code, not read from your shell

`CUDA_VISIBLE_DEVICES` is written into the child environment when the LLM service spawns vLLM itself
(`start_vllm_on_init: true`). `nemo_voice_agent/pipecat/services/nemo/llm.py` copies the current environment,
then derives the value from the `llm.device` config key: `cuda:1` becomes `CUDA_VISIBLE_DEVICES=1`, and `cpu`
becomes the empty string. A plain `cuda` leaves the variable untouched, so whatever you exported is inherited
by the vLLM process.

## Quick reference

Use the following examples as a quick lookup for common local-development and evaluation environments.

Local development against the default HuggingFace or vLLM backend:

```bash
export HF_TOKEN="hf_..."                       # only for gated model repos
export HF_HUB_CACHE="/path/to/large/disk/hf"   # only to relocate the cache
cd examples/generic_voice_agent/server/
python server.py
```

Serving a browser on another machine:

```bash
export SERVER_PUBLIC_HOST="10.0.0.42"
export WEBSOCKET_SCHEME="ws"
cd examples/generic_voice_agent/server/
python server.py
```

Running the evaluation harness with a hosted judge:

```bash
cd evaluation
export NVIDIA_API_KEY="nvapi-..."
export JUDGE_API_KEY="nvapi-..."
# Each of the next three commands blocks — run them in three terminals.
WEBSOCKET_PORT=8766 FASTAPI_PORT=7861 SERVER_CONFIG_PATH=server_configs/user.yaml  python bot_server.py
WEBSOCKET_PORT=8765 FASTAPI_PORT=7860 SERVER_CONFIG_PATH=server_configs/agent.yaml python bot_server.py
python run_evaluation.py --domain restaurant
```

Both bot servers also bind a FastAPI app, so `FASTAPI_PORT` must differ between them as well — leaving both at
the `7860` default makes the second process fail with `[Errno 98] Address already in use`. The shipped
launchers pair the ports the same way: `evaluation/run_agent.sh` uses `8765`/`7860` and
`evaluation/run_user.sh` uses `8766`/`7861`.

## Related Topics

Use these pages for the procedures and troubleshooting context associated with the variables in this reference:

- [Installation](../../get-started/installation.md)
- [Quickstart](../../get-started/quickstart.md)
- [Evaluation Quickstart](../../evaluate/run-evaluations/quickstart.md)
- [Troubleshooting](../../troubleshooting/index.md)

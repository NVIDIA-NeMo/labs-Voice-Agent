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

# Security Policy

NVIDIA is dedicated to the security and trust of our software products and services, including all
source code repositories managed through our organization. The `SECURITY.md` file at the root of the
NeMo Labs Voice Agent repository is the authoritative policy; this page summarizes it and adds
deployment notes specific to this codebase.

## Report a vulnerability

**Do not report security vulnerabilities through GitHub.** Do not open a public issue, discussion, or
pull request describing a potential vulnerability. If a potential security issue is inadvertently
reported through a public channel, NVIDIA maintainers may limit public discussion and redirect the
reporter to a private disclosure channel.

Use one of these contact points instead:

| Channel | Where |
| --- | --- |
| Web form | [Security Vulnerability Submission Form](https://www.nvidia.com/object/submit-security-vulnerability.html) |
| E-mail | psirt@nvidia.com |
| PGP key for encrypted e-mail | [NVIDIA public PGP key](https://www.nvidia.com/en-us/security/pgp-key) |
| Policy details | [NVIDIA PSIRT policies](https://www.nvidia.com/en-us/security/psirt-policies/) |
| Product security portal | [nvidia.com/en-us/security](https://www.nvidia.com/en-us/security) |

### What to include in a report

Include enough detail for NVIDIA PSIRT to reproduce and assess the issue:

- Product name and the version, branch, or commit that contains the vulnerability.
- Type of vulnerability (code execution, denial of service, buffer overflow, and so on).
- Step-by-step instructions to reproduce it.
- Proof-of-concept or exploit code, if you have it.
- Potential impact, including how an attacker could exploit the issue.

NVIDIA does not run a bug bounty program. NVIDIA does offer acknowledgement when an externally
reported security issue is addressed under its coordinated vulnerability disclosure policy.

## Deployment notes for this repository

The example server in `examples/generic_voice_agent/server/server.py` and the evaluation bot server in
`evaluation/bot_server.py` are reference implementations for local development and benchmarking. They
are not hardened for untrusted networks. Review the following before exposing either one beyond
`localhost`.

| Behavior | Where it comes from | What to do |
| --- | --- | --- |
| The WebSocket server binds to `0.0.0.0` by default | `SERVER_HOST` environment variable, read in `server.py` | Set `SERVER_HOST=127.0.0.1` for local-only use, or put the server behind a reverse proxy that terminates TLS and authenticates clients |
| The FastAPI helper app enables permissive CORS (`allow_origins=["*"]`, `allow_credentials=True`, all methods and headers) | `create_fastapi_app` in `nemo_voice_agent/pipecat/bot_server.py` | Restrict origins in your own deployment wrapper rather than shipping this app as-is |
| No authentication or authorization on the WebSocket or on the `/connect` endpoint | Same module | Add an authenticating proxy in front of both ports |
| One client at a time; additional connections are closed with WebSocket code `1013` | Pipecat's WebSocket server transport | Do not treat this as an access control; it is a capacity guard, and LLM context is preserved across reconnects |

Because context survives a reconnect on the same process, anyone who can reach the port can resume the
previous caller's conversation. That is the main reason to keep the port private.

## Handling credentials

Hosted-endpoint backends read API keys from the environment. `load_dotenv(override=True)` runs at
import time in `server.py`, `evaluation/bot_server.py`, and the evaluation judge helper in
`nemo_voice_agent/evaluation/utils.py`, so a `.env` file is picked up automatically and **overrides**
already-exported variables.

| Variable | Used by |
| --- | --- |
| `NVIDIA_API_KEY` | NIM/Riva-backed ASR, TTS, and LLM services |
| `NVIDIA_INFERENCE_API_KEY` | LLM service, when a separate inference-endpoint key is required |

Practices to follow:

- Keep keys in `.env` or your secret manager, never in a YAML config that is committed. `.env` is
  covered by `.gitignore`, but a key pasted into a `server_configs/*.yaml` file is not.
- Do not paste keys into issues, pull requests, or logs. Server logs (`bot_server.log` and its rotated
  `bot_server.*.log` siblings) are gitignored, but they can still capture request metadata.
- Rotate any key that has been committed or shared, then report the exposure through the channels
  above if it belonged to someone else.

See [Environment variables](../../reference/runtime/environment.md) for the full list of variables the servers
read, and [NVIDIA NIM endpoints](../../build-voice-agents/model-serving/nvidia-nim.md) for how the hosted backends are configured.

## Recorded audio and transcripts

Audio logging is off by default: `transport.record_audio_data` is `false` in the shipped
`default.yaml`. When you turn it on, the logger writes per-turn WAV files plus JSON metadata that
includes transcriptions, under `transport.audio_log_dir` (`./audio_logs` by default). These
recordings are personal data in most jurisdictions. Treat the output directory accordingly, obtain
consent before recording, and delete sessions you no longer need. See
[Audio logging](../../build-voice-agents/configure/audio-logging.md) for the configuration keys and output layout.

Evaluation runs also persist full conversation transcripts and LLM contexts under the run's result
directory. Those artifacts can contain scenario fixture data as well as anything a live speaker said
during a run.

## Dependencies and licenses

Third-party components ship with their own security advisories and license terms. Vulnerabilities in
an upstream dependency should be reported to that project first; report them to NVIDIA PSIRT as well
if this repository's use of the dependency is what makes the issue exploitable. See
[Third-party notices](../third-party-notices.md) for the dependency inventory.

## Related pages

Use these pages for the broader contribution and installation requirements:

- [Contributing](index.md) — development workflow, linting, and pre-commit setup.
- [Installation](../../get-started/installation.md) — supported Python versions and install procedure.

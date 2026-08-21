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

# Prerequisites

Verify the requirements for the workflow you plan to run before installing NeMo Voice Agent.

## Hardware Requirements

Verify that your machine and interactive audio setup meet these hardware requirements:

| Requirement | Detail |
| --- | --- |
| GPU | An NVIDIA GPU and driver compatible with the CUDA wheel selected in `pyproject.toml`. The default is CUDA 13.0. |
| Model support | The default NVFP4 language model requires hardware with FP4 support. Speech and text-to-speech models also use CUDA by default. |
| Audio devices | A microphone and speaker available to the browser for an interactive voice session. |

## Software Requirements

Install or make available the software required by the workflow you plan to run:

| Requirement | Detail |
| --- | --- |
| Operating system | Linux. `install.sh` installs system packages with `apt-get`. With another package manager, install the equivalent packages yourself. |
| Python | Python 3.12 or 3.13. The project configures `uv` to download and manage a compatible interpreter. |
| Node.js and npm | Required for the browser client. `install.sh` installs both on systems that provide `apt-get`. |

## Pre-Installation Checklist

Run these commands to confirm that the GPU and optional browser tools are available:

```bash
nvidia-smi
node --version
npm --version
```

## Troubleshoot Prerequisites

Do not run the installer inside a non-`base` conda environment. Run `conda deactivate` first. This prevents
the conda compiler and Python headers from interfering with packages that build native extensions.

## Next Steps

After the prerequisite checks pass, install the project or review the complete first-run workflow:

- [Installation](installation.md) creates the virtual environment and installs dependencies.
- [Quickstart](quickstart.md) starts the model server, voice-agent server, and browser client.

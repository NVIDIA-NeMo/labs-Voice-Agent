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

# Installation

NeMo Labs Voice Agent installs from source into a `uv`-managed virtual environment. For the shortest setup
path, run `bash install.sh` at the repository root. Use the manual steps to inspect or adapt the installation.

## Prerequisites

Before installing, verify your operating system, Python, GPU, model, browser, and audio requirements on the
[Prerequisites](prerequisites.md) page.

## Installation Methods

Choose the installation script for the shortest setup path, or run the equivalent commands manually when
you need to adapt the environment.

### Installation Script: Recommended

Run the repository installation script to create the environment and prefetch the required runtime resources.

```bash
git clone https://github.com/NVIDIA-NeMo/labs-Voice-Agent.git
cd labs-Voice-Agent
bash install.sh
```

You can rerun `install.sh` safely. The script resolves the environment again in place.

When it finishes, activate the environment or prefix commands with `uv run`:

```bash
source .venv/bin/activate
# or
uv run python -c "import nemo_voice_agent; print(nemo_voice_agent.__version__)"
```

### Manual Installation

Run the equivalent commands directly when you need to inspect or adapt each installation step.

```bash
sudo apt-get update
sudo apt-get install -y npm nodejs build-essential python3-dev

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync
uv run python -c "import nltk; nltk.download('cmudict'); nltk.download('averaged_perceptron_tagger_eng')"
```

To install the test tooling used in continuous integration (CI), run:

```bash
uv sync --all-extras --group test
```

### What install.sh Does

The installation script prepares the operating system, Python environment, and runtime text resources.

1. **Refuses to run inside a non-`base` conda environment.** If `CONDA_DEFAULT_ENV` is set to anything other than
   `base`, the script prints an error and exits 1. Conda's GCC combined with system Python headers breaks
   C extensions that compile from source, so run `conda deactivate` first.
2. **Installs operating system packages** with `sudo apt-get install -y npm nodejs build-essential python3-dev`.
3. **Installs `uv`** from `https://astral.sh/uv/install.sh` if it is not already on `PATH`. The installer
   places it in `~/.local/bin`.
4. **Runs `uv sync`**, which reads `pyproject.toml` plus `uv.lock` and creates `.venv/` in the current
   directory.
5. **Prefetches two Natural Language Toolkit (NLTK) corpora** to avoid a network download during a live
   text-to-speech (TTS) session.

### Why Those apt Packages

The operating system packages support dependency compilation and the browser client:

| Package | Reason |
| --- | --- |
| `build-essential`, `python3-dev` | Some dependencies ship only source distributions and compile during installation. `cdifflib`, a transitive dependency of `nemo-toolkit[tts]`, requires a C/C++ toolchain and `Python.h`. |
| `npm`, `nodejs` | Build and serve the Vite browser client in `examples/generic_voice_agent/client/`. Not needed for a headless or evaluation-only install. |

### The NLTK Prefetch

Kokoro TTS phonemizes out-of-vocabulary words through an Apache-2.0 `g2p_en` fallback
(`nemo_voice_agent/pipecat/services/nemo/_g2p_fallback.py`). This fallback replaces misaki's GPL-3.0
espeak-ng path and requires two NLTK corpora:

```bash
uv run python -c "import nltk; nltk.download('cmudict'); nltk.download('averaged_perceptron_tagger_eng')"
```

If you skip this step, a manual installation downloads the corpora during the first TTS use.

## Additional Setup

After the Python environment is installed, configure the CUDA build, model access, cache location, and
browser dependencies required by your workflow.

### Choose CUDA Wheels

`uv` selects PyTorch and vLLM wheels using `torch-backend` under `[tool.uv]` in `pyproject.toml`. The shipped
value is `cu130`. To target a different build, edit that key and re-run `uv sync`:

| Value | Wheel Index |
| --- | --- |
| `cu130` | CUDA 13.0 (default) |
| `cu128` | CUDA 12.8 |
| `cu124` | CUDA 12.4 |
| `cpu` | CPU-only (no GPU inference) |

Verify the selected build:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Configure Hugging Face Credentials and Cache

The repository code does not read either variable directly. The `huggingface_hub` library consumes both,
and every model download uses that library. Export the variables in the shell that starts the server:

| Variable | Purpose |
| --- | --- |
| `HF_TOKEN` | Required for gated repositories such as `meta-llama/Llama-3.1-8B-Instruct`. Request access on the model page first. |
| `HF_HUB_CACHE` | Moves the model cache from the default location. This is useful when your home directory is small. |

```bash
export HF_TOKEN="hf_..."
export HF_HUB_CACHE="/path/to/large/disk/huggingface"
```

`examples/generic_voice_agent/server/server.py` calls `load_dotenv(override=True)`. Instead of exporting the
variables, you can add them to a `.env` file next to `server.py` or in any parent directory. Refer to
[Environment variables](../reference/runtime/environment.md) for the variables the server itself reads.

If Hugging Face downloads fail with I/O errors, download the repository in advance and point the configuration
to the local path:

```bash
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 --local-dir /path/to/model
```

Then set `llm.model` to `/path/to/model`. You can use the same approach for TTS models.

### Install Browser Client Dependencies

`install.sh` installs Node.js but not the client's JavaScript packages. Install them in a separate terminal:

```bash
cd examples/generic_voice_agent/client
npm install
```

## Installation Verification

Verify that the package imports successfully, then run the CPU-compatible unit tests.

```bash
uv run python -c "import nemo_voice_agent; print(nemo_voice_agent.__version__)"
uv run pytest tests/unit -m "not gpu"
```

The unit suite runs in-process and needs no GPU or model serving.

## Troubleshoot the Installation

If installation or runtime verification fails, review [Troubleshooting](../troubleshooting/index.md) for
known failures and recovery steps.

## Next Steps

After verification succeeds, launch the default agent or learn how to use a different model-serving path.

The default configuration points `llm.model_config` at
`server_configs/llm_configs/nemotron_nano_v3.yaml`, which sets `start_vllm_on_init: false`. You must start
vLLM yourself before launching the server. Continue with:

- [Quickstart](./quickstart.md): Start vLLM, the server, and the browser client.
- [Serving with vLLM](../build-voice-agents/model-serving/vllm.md): Learn the serving flags and how the server communicates with vLLM.
- [Hosted NVIDIA NIM endpoints](../build-voice-agents/model-serving/nvidia-nim.md): Skip local GPU serving.

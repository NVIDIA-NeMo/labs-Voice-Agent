# NeMo Voice Agent

A fully open-source framework to build, deploy and evaluate voice agents with NVIDIA Nemotron and other
open-source models.

No API keys required to get started. Happy hacking~!

**📖 Documentation: https://docs.nvidia.com/nemo/labs-voice-agent**

## ✨ Key Features

- Open-source, local deployment, and flexible customization.
- Talk to most LLMs from vLLM/HuggingFace with configurable prompts.
- Streaming speech recognition with low latency and end-of-utterance detection.
- Low latency TTS for fast audio response generation.
- Speaker diarization up to 4 speakers in different user turns.
- WebSocket server for easy deployment.
- Tool calling for LLMs to use external tools and adjust its own behavior.
- Voice-agent evaluation harness with deterministic + LLM-judged scoring, and 328 scenarios across 4
  primary benchmark domains (`eva_airline`, `tau2_airline`, `tau2_retail`, `tau2_telecom`).

## 🚀 Quick Start

You need a Linux machine with an NVIDIA GPU, a microphone, and a speaker. The shipped default LLM runs
on a single GPU with FP4 support.

```bash
bash install.sh                        # deps + venv; see the docs for the manual path
source .venv/bin/activate

# The default config expects vLLM to already be running — start it in its own terminal:
# Needs ~24GB+ GPU memory. NVFP4 runs natively on Blackwell (GB200, DGX Spark/GB10, RTX 5090);
# on Hopper (H100/H200) or Ampere it falls back to W4A16 kernels.
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
    --trust-remote-code --tensor-parallel-size 1 --enable-prefix-caching \
    --max-num-seqs 1 --gpu-memory-utilization 0.8 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3

cd examples/generic_voice_agent/server && python server.py     # terminal 1
cd examples/generic_voice_agent/client && npm install && npm run dev   # terminal 2
```

If your GPU has less memory or doesn't support NVFP4, swap in a smaller, unquantized model instead by
pointing `llm.model` / `llm.model_config` in `default.yaml` at one of the other bundled `llm_configs/`
(e.g. `nemotron_nano_v2.yaml` at 9B params, or `qwen3-8B.yaml` / `llama3.1-8B-instruct.yaml` /
`qwen2.5-7B.yaml`, all of which fit comfortably on a single consumer GPU).

Then open the address printed by the client. Full walkthrough:
[Installation](https://docs.nvidia.com/nemo/labs-voice-agent/get-started/installation) ·
[Quickstart](https://docs.nvidia.com/nemo/labs-voice-agent/get-started/quickstart)

## 📚 Documentation

| | |
| --- | --- |
| [Get Started](https://docs.nvidia.com/nemo/labs-voice-agent/get-started/quickstart) | Install, run your first agent, and understand the pipeline |
| [Core Concepts](https://docs.nvidia.com/nemo/labs-voice-agent/about/architecture) | Architecture, ASR, diarization, TTS, turn-taking, and LLM backends |
| [Build Voice Agents](https://docs.nvidia.com/nemo/labs-voice-agent/build-voice-agents/overview) | Configuration, model serving, tool calling, and extending the pipeline |
| [Evaluate](https://docs.nvidia.com/nemo/labs-voice-agent/evaluate-voice-agents/overview) | The two-bot harness, scoring model, and authoring your own scenarios |
| [Reference](https://docs.nvidia.com/nemo/labs-voice-agent/reference/runtime/server-config-schema) | Config schema, environment variables, CLI flags, and metrics |
| [Troubleshooting](https://docs.nvidia.com/nemo/labs-voice-agent/troubleshooting/troubleshooting) | Common failures and how to diagnose them |

## 📊 Evaluation

The repo ships a full evaluation harness under [`evaluation/`](evaluation/). It runs your agent against a
**simulated user** — a second voice agent — routes audio between the two over WebSocket, and scores each
scenario on up to six orthogonal signals.

| Domain | Scenarios | Source |
| --- | --- | --- |
| `eva_airline` | 50 | Airline customer service, from [ServiceNow/eva](https://github.com/ServiceNow/eva) (MIT) |
| `tau2_airline` | 50 | Airline reservations, from [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) (MIT) |
| `tau2_retail` | 114 | Online retail customer service, same source |
| `tau2_telecom` | 114 | Telecom tech support — dual-side, with cross-side state sync |

See [Evaluate](https://docs.nvidia.com/nemo/labs-voice-agent/evaluate-voice-agents/overview).

## 📅 Latest Updates

- **2026-09-09** — `tau2_airline` evaluation now scores natural-language assertions on the 24 scenarios
  whose expected database is unchanged, so an agent that does nothing no longer passes them. One upstream
  reference action that the domain's own policy forbids is dropped from gold replay.
- **2026-09-02** — Shipped default LLM switched to
  [Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4).
  Added LLM configs for
  [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) and
  [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
- **2026-08-06** — NeMo Voice Agent graduates from
  [NVIDIA-NeMo/Speech](https://github.com/NVIDIA-NeMo/Speech/tree/main/examples/voice_agent) into its own repo.
- **2026-06-13** — Evaluation harness shipped: four benchmark domains and per-scenario `success_signals` scoring.
- **2026-05-15** — Support for
  [Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4).

Full history: [Release Notes](https://docs.nvidia.com/nemo/labs-voice-agent/about/release-notes)

## 💡 Upcoming Next

- Accuracy and robustness ASR model improvements.
- Combine ASR and speaker diarization model to handle overlapping speech.
- More evaluation domains and scenarios.

## Acknowledgments

- This project uses the [Pipecat](https://github.com/pipecat-ai/pipecat) orchestrator framework.
- The `eva_airline` evaluation domain (50 airline customer-service scenarios) is adapted from
  [ServiceNow/eva](https://github.com/ServiceNow/eva) (MIT-licensed, version `0.1.3`). Per-scenario fixtures
  and tool function bodies carry inline `# Adapted from ...` attribution; see
  [`nemo_voice_agent/evaluation/data/README.md`](nemo_voice_agent/evaluation/data/README.md) for the full
  source/license inventory.
- The `tau2_airline`, `tau2_retail`, and `tau2_telecom` evaluation domains (278 scenarios total) are ported
  from [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) (MIT-licensed) at the
  `voice-user-sim-v1.0` tag (commit `17e07b1`). Upstream tasks, DBs, and policies are imported via the
  scripts under [`scripts/prepare_tau2_data/`](scripts/prepare_tau2_data/); generated scenario classes carry
  inline attribution headers. The companion `tau2_telecom_workflow` registration pairs each telecom task with
  an alternate policy variant for A/B comparison; it shares the underlying 114 tasks with `tau2_telecom` and
  is not counted separately.

## Contributing

We welcome contributions. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) — it covers the dev environment,
the `ruff` formatting and lint gates, the test layout, conventional commits, and the DCO `Signed-off-by`
requirement that CI enforces.

See also [`SECURITY.md`](SECURITY.md) for reporting security issues and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for third-party licenses.

# Extending the Bot Pipeline

This guide is for users who want to evaluate something different than the stock NeMo-Voice-Agent pipeline — a different LLM backend, a different STT/TTS stack, a custom processor in the chain, or a wholly different pipecat pipeline assembly. For adding **scenarios, tools, or domains** (the data layer), see [`EXTENDING_DATA.md`](EXTENDING_DATA.md). This doc covers the **bot compute layer** (the pipeline that runs inside the agent / user-sim bot processes).

## Contents

- [How customization composes with the eval harness](#how-customization-composes-with-the-eval-harness)
- [Three customization tiers](#three-customization-tiers)
- [Tier 1 — Swap models or configs via YAML](#tier-1--swap-models-or-configs-via-yaml)
- [Tier 2 — Insert a custom pipecat processor](#tier-2--insert-a-custom-pipecat-processor)
- [Tier 3 — Build a whole new pipecat pipeline](#tier-3--build-a-whole-new-pipecat-pipeline)
- [Verifying your customization](#verifying-your-customization)

## How customization composes with the eval harness

The eval harness is a **separate process** (`evaluation/bridge.py`) from the bot. It opens a WebSocket client to your bot, shuttles audio between two bot connections (agent + user-sim), and orchestrates scenario lifecycle via RTVI control messages. As long as your bot satisfies two things, the rest of pipecat (and the model/processor choices inside) is yours:

1. **Speaks pipecat's WebSocket transport protocol** on the configured port (`WEBSOCKET_PORT`).
2. **Has an `RTVIProcessor` in its pipeline** with the five RTVI actions registered (see [Tier 3](#tier-3--build-a-whole-new-pipecat-pipeline) below).

The bridge doesn't care how the audio is generated, what model produced the response, or what processors sit between STT and LLM. It cares about the wire protocol and the control plane.

## Three customization tiers

Pick the tier that matches how much you want to change. Higher tiers are strictly more flexible but require more code.

| Tier | What changes | Effort | When to use |
|---|---|---|---|
| **1. YAML swap** | `server_configs/*.yaml` only | No Python | Swap LLM / TTS / STT to a different model or endpoint that the existing `build_*` functions already support. |
| **2. Custom processor** | Insert a `FrameProcessor` into the existing pipeline | ~20 LOC Python | Add a transform between stages (markdown sanitizer between LLM and TTS, transcript redactor between STT and LLM, etc.). |
| **3. Whole new pipecat pipeline** | Replace `run_bot_websocket()` entirely | ~150 LOC Python | Use entirely different services or a custom pipeline shape. Free choice on everything except the WS-transport + RTVI contract. |

## Tier 1 — Swap models or configs via YAML

The cheapest customization: change which model the existing builders use. No Python edits.

### Two parallel config directories

There are two top-level `server_configs/` directories in the repo, one for each entrypoint:

| Path | Used by | Purpose |
|---|---|---|
| `examples/generic_voice_agent/server/server_configs/` | The standalone demo server (`examples/generic_voice_agent/server/server.py`) | Browser-client demos. |
| **`evaluation/server_configs/`** | The eval bots (`evaluation/bot_server.py`) | The bots that the bridge connects to. **This is the one you want for eval runs.** |

The eval directory ships ready-to-use configs you can use as starting points:

- `agent.yaml`, `agent_nvidia.yaml`, `agent_nvidia_omni.yaml`, `agent_think.yaml` — agent-side variants (different LLMs, with or without reasoning mode).
- `user.yaml`, `user_nvidia.yaml`, `user_think.yaml` — user-sim variants matching the same model families.

The `*_think.yaml` variants enable reasoning mode and automatically pull in the matching `*_think` sub-config. `_nvidia*` variants target NVIDIA-hosted endpoints.

### Where things are wired

`nemo_voice_agent/pipecat/services/nemo/builders.py` exposes one `build_*` function per pipeline stage: `build_stt`, `build_llm`, `build_tts`, `build_vad_analyzer`, `build_turn_taking`, `build_ws_transport`, `build_audio_logger`, `build_context_and_aggregators`. Each reads its config from the top-level YAML at the corresponding key:

```
evaluation/server_configs/<your_config>.yaml
├── stt:      → build_stt
├── diar:     → build_diar (optional)
├── llm:      → build_llm
├── tts:      → build_tts
├── vad:      → build_vad_analyzer
└── turn_taking: → build_turn_taking
```

Each component's section typically has a `model_config:` field pointing at a sub-YAML in `examples/generic_voice_agent/server/server_configs/<component>_configs/<model>.yaml` that carries the per-model parameters. The eval top-level configs reference these sub-YAMLs by relative path — they're shared across both entrypoints, so changing a sub-YAML affects both demo and eval runs.

### Example: swap the LLM model

To run the eval agent against a different vLLM-served model:

1. **Pick a starting point.** Copy one of the existing eval configs that's closest to what you want (e.g. `evaluation/server_configs/agent_nvidia.yaml` if you're targeting a different NVIDIA-hosted endpoint, or `evaluation/server_configs/agent_think.yaml` if you want reasoning mode on).
2. **Drop a new model sub-YAML** at `examples/generic_voice_agent/server/server_configs/llm_configs/my_custom_model.yaml` (copy a sibling — e.g. `llama3_70b.yaml` — as a starting point). Edit the `model_id`, `vllm_server_params`, sampling params to match your target.
3. **Edit your eval config** (`evaluation/server_configs/agent_my_custom.yaml` or whichever you copied) so the `llm.model_config:` field points at `llm_configs/my_custom_model.yaml`.
4. **Run the bot:**
   ```bash
   WEBSOCKET_PORT=8765 \
   SERVER_CONFIG_PATH=server_configs/agent_my_custom.yaml \
   python evaluation/bot_server.py
   ```
   `SERVER_CONFIG_PATH` is resolved relative to the current working directory; the convention is to `cd evaluation` first.

Same pattern for `tts:` / `stt:` — sibling `tts_configs/` and `stt_configs/` directories under `examples/generic_voice_agent/server/server_configs/`.

### Examples of what Tier 1 can do without writing Python

- Swap LLM backend between `hf`, `vllm`, or `auto` modes via the `llm.type` field.
- Switch TTS between Kokoro, FastPitch-HiFiGAN, or Magpie.
- Switch STT between different NeMo Parakeet variants.
- Toggle reasoning mode (`llm.enable_reasoning: true`) — pulls in the `*_think.yaml` model variant automatically.
- Configure backchannels (`turn_taking.backchannel_phrases_path`).
- Adjust VAD stop-seconds, sample rates, audio chunk size.

If your goal is "use the same pipeline shape but with a different model," Tier 1 is almost always enough. Don't move up tiers until you actually need to modify the processor chain.

## Tier 2 — Insert a custom pipecat processor

When you need something the existing builders don't provide — typically a transform between stages — write a `FrameProcessor` subclass and slot it into the pipeline list.

### Where the pipeline is assembled

`evaluation/bot_server.py:run_bot_websocket()` (or `examples/generic_voice_agent/server/server.py:run_bot_websocket()` — same shape) builds the pipeline as a flat list:

```python
pipeline_list = [ws_transport.input(), rtvi, stt]
# ... optional diarization
pipeline_list.extend([user_agg, llm, tts, ws_transport.output(), assistant_agg])
pipeline = Pipeline(pipeline_list)
```

The standard positions for a custom processor:

| Position | Use case |
|---|---|
| Before `stt` (after `ws_transport.input()`) | Audio-level processing (e.g. noise suppression, custom VAD). |
| Between `stt` and `user_agg` | Transcript transforms (e.g. ASR error correction, sensitive-content redaction, language detection). |
| Between `user_agg` and `llm` | Context shaping (e.g. memory injection, prompt augmentation, intent classification). |
| Between `llm` and `tts` | LLM output transforms (e.g. **Markdown stripper**, profanity filter, per-domain text post-processing). |
| Between `tts` and `ws_transport.output()` | Audio-level output processing (e.g. compression, voice morph, beep insertion). |

### Example — Markdown sanitizer between LLM and TTS

A real Tier 2 case: agent LLMs sometimes emit `**bold**` / `# heading` / `- bullet` Markdown despite explicit prompt-level prohibition. The TTS would otherwise pronounce these as "asterisk", "pound", etc. A pre-TTS sanitizer fixes this deterministically without depending on model compliance:

```python
import re
from pipecat.frames.frames import TextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class MarkdownStripper(FrameProcessor):
    """Strip Markdown formatting characters from TextFrames before TTS.

    Preserves identifier characters (e.g. '#W2378156' order IDs, '*123'
    extensions) by only matching formatting patterns, not bare characters.
    """

    # **bold** or *emphasis* — require non-word chars at boundaries so '#W123'
    # and '*123' (identifier prefixes) are not matched.
    _BOLD = re.compile(r"\*\*([^*]+?)\*\*")
    _EMPH = re.compile(r"(?<!\w)\*([^*\s][^*]*?)\*(?!\w)")
    _UNDERLINE = re.compile(r"(?<!\w)_([^_\s][^_]*?)_(?!\w)")
    # Line-start markers.
    _HEADING = re.compile(r"^#+\s+", flags=re.MULTILINE)
    _LIST = re.compile(r"^[-*]\s+", flags=re.MULTILINE)
    _NUMBERED = re.compile(r"^\d+\.\s+", flags=re.MULTILINE)

    @classmethod
    def sanitize(cls, text: str) -> str:
        text = cls._BOLD.sub(r"\1", text)
        text = cls._EMPH.sub(r"\1", text)
        text = cls._UNDERLINE.sub(r"\1", text)
        text = cls._HEADING.sub("", text)
        text = cls._LIST.sub("", text)
        text = cls._NUMBERED.sub("", text)
        return text

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
            frame.text = self.sanitize(frame.text)
        await self.push_frame(frame, direction)
```

Wire it into the pipeline in your `bot_server.py`:

```python
from your_module import MarkdownStripper

# After build_llm + build_tts, before the Pipeline call:
markdown_stripper = MarkdownStripper()

pipeline_list = [ws_transport.input(), rtvi, stt]
pipeline_list.extend([
    user_agg,
    llm,
    markdown_stripper,       # ← inserted between LLM and TTS
    tts,
    ws_transport.output(),
    assistant_agg,
])
```

### Notes

- Custom processors should usually subclass `FrameProcessor`, override `process_frame`, and always call `super().process_frame(frame, direction)` first then `self.push_frame(frame, direction)` to forward.
- Mind the **direction**. Downstream frames flow from input toward output; upstream frames (typically control / metrics) flow back. Most transforms act on `DOWNSTREAM` only.
- Mind the **frame type**. Filter on `isinstance(frame, TextFrame)` (or whichever type you target) so other frame types pass through unchanged.
- If your processor needs configuration, accept it in `__init__` and read from a sub-section of the top-level YAML (mirrors how the existing builders work).

## Tier 3 — Build a whole new pipecat pipeline

When Tier 2 isn't enough — you want different services, a different pipeline shape, or to integrate something the existing builders don't cover — replace `run_bot_websocket()` entirely. The eval harness contract is narrow:

### The contract

Your custom pipeline MUST have:

1. **A pipecat WebSocket transport** — `WebSocketServerTransport` (from `pipecat.transports.network.websocket_server`) bound to `WEBSOCKET_PORT` (env var, default 8765 agent / 8766 user-sim). This handles the wire protocol (protobuf-framed audio + control messages) so you don't have to.

2. **An `RTVIProcessor`** somewhere in the pipeline with the **five required actions** registered:

| Action factory | Direction | Purpose |
|---|---|---|
| `create_update_system_prompt_action` | bridge → bot | Bridge sends per scenario: sets the agent's system prompt, registers tools from the `tool_domain` registry, clears prior `shared_state` (preserving dict identity). |
| `create_apply_initialization_action` | bridge → bot | Bridge sends per scenario: merges `shared_state_init` JSON, resolves `db_path → db` from disk, dispatches per-side init function mutations. |
| `create_apply_sync_delta_action` | bridge → bot | Bridge sends for telecom-only cross-side sync. Single-side domains never receive it; registering it as a no-op is harmless. |
| `create_get_scenario_summary_action` | bot → bridge | Bridge pulls at end of scenario: returns `{actions, db_hash, db?}` from `shared_state`. |
| `create_get_context_history_action` | bot → bridge | Bridge pulls at end of scenario for `bot_logs_*/llm_context.json`: returns the LLM conversation history. |

All five factories live in `nemo_voice_agent/pipecat/processors/frameworks/rtvi_actions.py`. Each takes a small set of pipeline references (a `TaskRef`, a `SharedStateRef`, etc.) and returns an `RTVIAction` that the processor calls when the matching wire message arrives.

### Minimal Tier 3 skeleton

```python
import os
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIProcessor
from pipecat.transports.network.websocket_server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)

# RTVI action factories — the only mandatory imports for eval compatibility.
from nemo_voice_agent.pipecat.processors.frameworks.rtvi_actions import (
    TaskRef,
    SharedStateRef,
    create_update_system_prompt_action,
    create_apply_initialization_action,
    create_apply_sync_delta_action,
    create_get_scenario_summary_action,
    create_get_context_history_action,
)

# Tool registry — used by update_system_prompt to look up per-scenario tools.
from nemo_voice_agent.evaluation.tools import get_schema_tool_for_eval


async def run_custom_bot():
    port = int(os.environ.get("WEBSOCKET_PORT", "8765"))
    host = os.environ.get("SERVER_HOST", "0.0.0.0")

    # ---- 1. Build your services ---------------------------------------
    # Use ANY pipecat-compatible services here. Stock NeMo builders, your
    # own custom services, third-party plugins — whatever your pipeline needs.
    stt = ...    # your STT
    llm = ...    # your LLM (must produce TextFrames + ToolCallFrames)
    tts = ...    # your TTS
    vad = ...    # your VAD analyzer

    # ---- 2. WebSocket transport (REQUIRED for eval compatibility) -----
    ws_transport = WebsocketServerTransport(
        host=host,
        port=port,
        params=WebsocketServerParams(vad_analyzer=vad, audio_in_enabled=True, audio_out_enabled=True),
    )

    # ---- 3. RTVI processor + the 5 required actions (REQUIRED) --------
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))
    task_ref = TaskRef()
    shared_state_ref = SharedStateRef()

    # Build your context + aggregators so they're available to the actions.
    # (See nemo_voice_agent/pipecat/services/nemo/builders.py:build_context_and_aggregators
    # for the reference assembly — typically a few lines.)
    user_agg, assistant_agg, context, original_messages, resettable = build_my_context(...)

    rtvi.register_action(
        create_update_system_prompt_action(
            task_ref, user_agg, assistant_agg, original_messages, resettable,
            system_role="system",
            system_prompt_suffix="",
            enable_tool_calling=True,
            llm=llm,
            context=context,
            rtvi=rtvi,
            tool_factory=get_schema_tool_for_eval,
            register_schema_tools=register_schema_tools_to_llm,  # your tool-registration adapter
            shared_state_ref=shared_state_ref,
        )
    )
    rtvi.register_action(create_apply_initialization_action(shared_state_ref))
    rtvi.register_action(create_apply_sync_delta_action(shared_state_ref))
    rtvi.register_action(create_get_scenario_summary_action(task_ref, shared_state_ref))
    rtvi.register_action(create_get_context_history_action(task_ref, assistant_agg))

    # ---- 4. Assemble the pipeline -------------------------------------
    # The shape is up to you. ws_transport.input() / output() and `rtvi` are
    # the only required-position elements; the rest is your design.
    pipeline = Pipeline([
        ws_transport.input(),
        rtvi,
        stt,
        user_agg,
        llm,
        tts,
        ws_transport.output(),
        assistant_agg,
    ])

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))
    task_ref.task = task

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_custom_bot())
```

### Free-choice points

Inside the contract, everything is yours:

- **Services.** Use any STT/LLM/TTS that emits/consumes pipecat frames. Roll your own `FrameProcessor` subclass with `process_frame` if you need to wrap a service that isn't pipecat-native.
- **Pipeline shape.** Add intermediate processors, parallel branches, anything pipecat supports.
- **Tool calling.** If your LLM service supports OpenAI-compatible function calling, you can reuse `register_schema_tools_to_llm` (see `nemo_voice_agent/utils/tool_calling/__init__.py`). Otherwise plug in your own tool-registration callback compatible with the `register_schema_tools` arg.
- **Context management.** The `user_agg` / `assistant_agg` pair just needs to be pipecat `LLMUserContextAggregator` / `LLMAssistantContextAggregator` instances (or duck-typed equivalents). You can use a custom `OpenAILLMContext` subclass to alter context shape, but the aggregators need to honor the standard frame protocol.
- **Reasoning, thinking-token budgets, parser plugins.** Out of scope of the contract — entirely your business.

### What you're NOT free to change

- The WebSocket wire protocol (use a pipecat WS transport).
- The five RTVI action signatures (your bot reads from / writes to the same wire messages the bridge sends/expects).
- The tool registry namespace key (`scenario.domain`) — your `register_schema_tools_to_llm` adapter must accept the same `tool_factory(name, domain=...)` interface so the bridge-sent `tool_domain` resolves to the right tool set.

### Non-pipecat bots

Out of scope for this guide. If you genuinely need to evaluate a non-pipecat agent (LiveKit, Rasa, hand-rolled), you'd have to re-implement pipecat's WS transport protocol + RTVI message handling. Possible but a large undertaking — start by reading `pipecat.transports.network.websocket_server` and `pipecat.serializers.protobuf` and treat them as the wire spec.

## Verifying your customization

Same flow regardless of tier. Use a small, fast scenario for the first iteration.

```bash
# Terminal 1 — your custom agent bot
WEBSOCKET_PORT=8765 SERVER_CONFIG_PATH=server_configs/agent.yaml python your/bot_server.py

# Terminal 2 — stock user-sim bot (or your custom user-sim, same contract)
WEBSOCKET_PORT=8766 SERVER_CONFIG_PATH=server_configs/user.yaml python evaluation/bot_server.py

# Terminal 3 — single scenario, judge optional
python evaluation/run_evaluation.py --scenarios restaurant__pizza_pepperoni
```

Inspect the resulting `eval_results/eval_<ts>/restaurant__pizza_pepperoni/`:

| File | Looking for |
|---|---|
| `metrics.json` | `is_successful` should be `True`, `False`, or `"N/A"` — never missing. If missing, your bot didn't return a usable `get_scenario_summary` payload. |
| `bridge_log.txt` | Search for `unknown action` errors → an RTVI action you forgot to register. Search for `update_system_prompt` events → your bot is receiving prompts. Search for `[AGENT METRICS] ttfb` → your LLM service is emitting TTFB events (informational only). |
| `bot_logs_agent/llm_context.json` | Should contain the scenario's system prompt as the first message and the agent's tool calls as `assistant.tool_calls` entries. If the system prompt is wrong, `update_system_prompt` didn't propagate. If tool calls are missing, your tool-registration adapter didn't wire the registry into the LLM service. |
| `final_scenario_db_hash.txt` | Should contain a `db_hash:` line. If missing, `get_scenario_summary` isn't returning the expected `{actions, db_hash}` shape. |

The [eval-result-analyzer skill](#) works on custom-pipeline runs too — it reads the per-scenario artifacts, not pipecat internals. Run it against the result dir and check the report for "framework"-class root causes (which is how contract-compliance issues surface).

---

For scenario / tool / domain extensions, see [`EXTENDING_DATA.md`](EXTENDING_DATA.md).
For the operator-facing view (how to run, read results), see [`README.md`](README.md).

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

# Custom Frame Processors

A Pipecat `FrameProcessor` is a single node in the bot pipeline: it receives frames, optionally
transforms or absorbs them, and forwards the rest. Writing one is the lowest-scope way to add
behavior that NeMo Labs Voice Agent's builders do not provide. Examples include an automatic speech
recognition (ASR) post-corrector, a Markdown sanitizer before text-to-speech (TTS), and transcript redaction.

## Prerequisites

Before you add a processor, complete the following preparation:

1. Run the [Quickstart](../../../get-started/quickstart.md) with the shipped pipeline.
2. Identify the frame type you need to transform and the pipeline stage that emits it.
3. Choose the demo server or evaluation bot entrypoint where you will insert the processor.

## Processor or Builder Swap?

Choose a processor for frame transformations and a builder change for service construction.

| You Want to | Do This |
| --- | --- |
| Use a different STT, LLM, or TTS model or endpoint | Edit YAML. Refer to [Server Config](../../configure/server-config.md). |
| Change how an existing stage is constructed | Swap the builder. Refer to [The Builder API](builders.md). |
| Add a transform *between* two existing stages | Write a `FrameProcessor` (this page). |
| Change the pipeline shape, transport, or control plane | Refer to [Building Your Own Pipeline](custom-pipeline.md). |

A processor is the right tool when the stages themselves are fine and you only need to touch the
frames flowing between them. It requires no changes to `nemo_voice_agent/`, only a few lines in your
copy of the server entrypoint.

## Where the Pipeline Is Assembled

Both entrypoints — `examples/generic_voice_agent/server/server.py` and `evaluation/bot_server.py` —
build the pipeline in `run_bot_websocket()` as a flat list, appending optional stages only when their
builder returned a value:

```python
pipeline_list = [ws_transport.input()]
if vad_processor is not None:
    pipeline_list.append(vad_processor)
pipeline_list.extend([rtvi, stt])
if diar is not None:
    pipeline_list.append(diar)
if turn_taking is not None:
    pipeline_list.append(turn_taking)
if user_audio_buffer is not None:
    pipeline_list.append(user_audio_buffer)
pipeline_list.extend([user_agg, llm])
if llm_text_processor is not None:
    pipeline_list.append(llm_text_processor)
pipeline_list.extend([tts, ws_transport.output(), assistant_agg])
pipeline = Pipeline(pipeline_list)
```

Insert your processor by adding one more element to that list. Refer to
[How It Works](../../../about/architecture.md) for what each stage does.

## Insertion Points

Pick the position by the frame type you need to see. Frame classes come from
`pipecat.frames.frames` unless noted.

| Position | Frames Arriving There | Typical Use |
| --- | --- | --- |
| After `ws_transport.input()` | `InputAudioRawFrame` | Resampling, gain control, noise suppression |
| Between `stt` and `user_agg` | `TranscriptionFrame`, `InterimTranscriptionFrame` | ASR error correction, PII redaction, language routing |
| After `diar` | `DiarResultFrame` (from `nemo_voice_agent.pipecat.frames.frames`) | Rewriting or filtering speaker labels |
| Between `user_agg` and `llm` | `LLMRunFrame` and everything from upstream | Context shaping, retrieval, or memory injection |
| Between `llm` and `llm_text_processor` | `LLMTextFrame` (streaming token chunks) | Token-level filtering; text may be split mid-word |
| Between `llm_text_processor` and `tts` | `AggregatedTextFrame` (whole sentences) | Markdown stripping, profanity filter, pronunciation rewrites |
| Between `tts` and `ws_transport.output()` | `TTSAudioRawFrame` | Output audio effects, loudness metering |

Two things decide between the last two text positions. `LLMTextProcessor` converts
`LLMTextFrame` into sentence-sized `AggregatedTextFrame`, as implemented by `build_llm_text_processor` in
`nemo_voice_agent/pipecat/services/nemo/builders.py`. Any regex that must match across token
boundaries belongs *after* it. Note that it is only present when `tts.use_text_aggregator` is true —
the default; when it is false, the TTS service does its own aggregation internally and only
`LLMTextFrame` reaches that point.

## Rules

Follow these rules so unrelated frames and pipeline direction continue to work as designed:

- **Subclass `FrameProcessor`** and override `async def process_frame(self, frame, direction)`.
- **Call `await super().process_frame(frame, direction)` first.** The base implementation handles
  `StartFrame`, `CancelFrame`, `InterruptionFrame`, and pause/resume bookkeeping. It does *not*
  forward anything.
- **Always forward with `await self.push_frame(frame, direction)`** unless you intentionally consume
  the frame. Dropping a control frame stalls everything downstream.
- **Filter on frame type.** `TranscriptionFrame`, `InterimTranscriptionFrame`, `LLMTextFrame`, and
  `AggregatedTextFrame` are all subclasses of `TextFrame`, so an `isinstance(frame, TextFrame)` test
  matches user speech as well as bot speech. Match the narrowest class you mean.
- **Check `direction`.** `FrameDirection.DOWNSTREAM` runs input toward output;
  `FrameDirection.UPSTREAM` carries errors and control signals back. Transforms should almost always
  guard on `DOWNSTREAM` and pass upstream frames through untouched.
- **Frames are mutable dataclasses**, so in-place edits such as `frame.text = ...` work and preserve
  the subclass and its metadata fields.
- **System frames jump the queue.** `InputAudioRawFrame` and other `SystemFrame` subclasses are
  dispatched ahead of queued data frames, so do not assume ordering between audio and transcripts.

## Example: Strip Markdown Before TTS

LLMs emit `**bold**`, `# heading`, and `- bullet` markup even when the system prompt forbids it, and
the TTS voice reads the punctuation aloud. A sanitizer placed immediately before `tts` fixes this
deterministically instead of relying on model compliance.

```python
import re

from pipecat.frames.frames import AggregatedTextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class MarkdownStripper(FrameProcessor):
    """Remove Markdown formatting from sentences on their way to TTS.

    Identifier-like tokens ('#W2378156' order IDs, '*123' extensions) survive
    because only formatting patterns are matched, never bare characters.
    """

    _BOLD = re.compile(r"\*\*([^*]+?)\*\*")
    _EMPH = re.compile(r"(?<!\w)\*([^*\s][^*]*?)\*(?!\w)")
    _UNDERLINE = re.compile(r"(?<!\w)_([^_\s][^_]*?)_(?!\w)")
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
        if isinstance(frame, AggregatedTextFrame) and direction == FrameDirection.DOWNSTREAM:
            frame.text = self.sanitize(frame.text)
        await self.push_frame(frame, direction)
```

Wire it into your copy of the entrypoint:

```python
from your_package.processors import MarkdownStripper

markdown_stripper = MarkdownStripper()

pipeline_list.extend([user_agg, llm])
if llm_text_processor is not None:
    pipeline_list.append(llm_text_processor)
pipeline_list.append(markdown_stripper)  # after aggregation, before TTS
pipeline_list.extend([tts, ws_transport.output(), assistant_agg])
```

## Side Effects to Watch

A processor can affect downstream context, timing, and logging even when it changes only one frame type:

- **The assistant context sees your edits.** `assistant_agg` sits at the end of the pipeline and
  builds the assistant turn from text frames whose `append_to_context` is true — the same objects
  your processor already mutated. For Markdown stripping that is desirable; for a change you want
  spoken but not remembered, emit a modified copy instead of editing in place.
- **Stateful processors need a reset hook.** The `resettable` list passed to the RTVI actions is
  iterated by `_reset_services` in
  `nemo_voice_agent/pipecat/processors/frameworks/rtvi_actions.py`, which calls a plain synchronous
  `reset()` on any entry that has one and skips `None`. Add your processor to that list and give it
  a `reset()` method if it carries per-conversation state. Refer to
  [RTVI Control Plane](../protocols/rtvi-actions.md).
- **Configuration belongs in YAML.** Accept options in `__init__` and read them from a section of
  the server config, the way the existing builders do, rather than hardcoding them.
- **Reasoning spans are handled elsewhere.** TTS already skips text between `tts.think_tokens`; do
  not reimplement that in a processor. Refer to
  [Reasoning Mode](../../../about/core-concepts/language-models/reasoning.md).

## Test It

Processors are unit-testable without a pipeline: keep the transform in a pure classmethod, then
drive `process_frame` directly with a stubbed `push_frame` to check forwarding and frame typing.
`tests/unit/test_runtime_state_machines.py` uses exactly this pattern for `UserAudioBuffer`.

```bash
uv run pytest tests/unit -m "not gpu"
```

Then run the bot end to end and listen. Refer to [Quickstart](../../../get-started/quickstart.md). If your
processor does not process frames, confirm it is in `pipeline_list` because optional surrounding stages are
appended conditionally. Also confirm that the selected frame class reaches that position.

## Next Steps

Continue with the guide for the extension boundary you need:

- [The Builder API](builders.md) — swap how a stage is constructed instead of what flows between stages.
- [Building Your Own Pipeline](custom-pipeline.md) — replace `run_bot_websocket()` entirely.
- [Extension Overview](../index.md) — how the extension points fit together.

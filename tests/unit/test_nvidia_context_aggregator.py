# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for NVIDIA context aggregation state machines."""

import asyncio
from copy import deepcopy

from pipecat.frames.frames import (
    AggregationType,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.processors.frame_processor import FrameDirection

from nemo_voice_agent.pipecat.frames.riva import RivaInterimTranscriptionFrame
from nemo_voice_agent.pipecat.processors.nvidia_context_aggregator import (
    NvidiaAssistantContextAggregator,
    NvidiaTTSResponseCacher,
    NvidiaUserContextAggregator,
)


def _drive(awaitable):
    """Run an async context-aggregator helper from a synchronous pytest test."""
    return asyncio.run(awaitable)


class _FakeContext:
    """Small OpenAI context stand-in for aggregation tests."""

    def __init__(self, messages=None):
        """Initialize with a copy of the supplied messages."""
        self._messages = list(messages or [])

    def get_messages(self):
        """Return the current context messages."""
        return self._messages

    def set_messages(self, messages):
        """Replace the current context messages."""
        self._messages = list(messages)

    def add_message(self, message):
        """Append a message to the context."""
        self._messages.append(message)

    def __deepcopy__(self, memo):
        """Deep-copy the fake context the same way the real context is copied."""
        copied = type(self)(deepcopy(self._messages, memo))
        memo[id(self)] = copied
        return copied


def _bare_user_aggregator(messages=None, *, chat_history_limit=-1, preserve_prompt_messages=1):
    """Construct a NvidiaUserContextAggregator without initializing the Pipecat base class."""
    aggregator = NvidiaUserContextAggregator.__new__(NvidiaUserContextAggregator)
    aggregator._context = _FakeContext(messages)
    aggregator._role = "user"
    aggregator._aggregation = ""
    aggregator.send_interims = True
    aggregator.chat_history_limit = chat_history_limit
    aggregator.preserve_prompt_messages = preserve_prompt_messages
    aggregator.last_transcript = None
    aggregator._user_speaking = False
    aggregator.seen_final = True
    aggregator._last_final_transcript = ""
    aggregator.pushed = []
    aggregator.reset_count = 0

    async def _push_frame(frame, direction=None):
        """Capture frames pushed by the aggregator."""
        aggregator.pushed.append((frame, direction))

    async def _reset():
        """Capture reset calls."""
        aggregator.reset_count += 1

    aggregator.push_frame = _push_frame
    aggregator.reset = _reset
    return aggregator


def _bare_assistant_aggregator(messages=None):
    """Construct a NvidiaAssistantContextAggregator without initializing the Pipecat base class."""
    aggregator = NvidiaAssistantContextAggregator.__new__(NvidiaAssistantContextAggregator)
    aggregator._context = _FakeContext(messages)
    aggregator._role = "assistant"
    aggregator._aggregation = ""
    aggregator.pushed = []
    aggregator.reset_count = 0

    async def _push_frame(frame, direction=None):
        """Capture frames pushed by the aggregator."""
        aggregator.pushed.append((frame, direction))

    async def _reset():
        """Capture reset calls."""
        aggregator.reset_count += 1

    aggregator.push_frame = _push_frame
    aggregator.reset = _reset
    return aggregator


def test_user_context_truncation_preserves_prompts_and_recent_user_turns():
    """Context truncation keeps configured prompt messages and the most recent user turns."""
    aggregator = _bare_user_aggregator(
        [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "seed"},
            {"role": "assistant", "content": "seed response"},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "developer", "content": "developer note"},
            {"role": "user", "content": "new question"},
            {"role": "assistant", "content": "new answer"},
        ],
        chat_history_limit=1,
        preserve_prompt_messages=2,
    )

    truncated = _drive(aggregator.get_truncated_context())

    assert truncated is not aggregator.context
    assert truncated.get_messages() == [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "seed"},
        {"role": "developer", "content": "developer note"},
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "new answer"},
    ]
    assert aggregator.context.get_messages()[2]["content"] == "seed response"


def test_user_push_aggregation_updates_existing_user_message_and_pushes_interruption_first():
    """User aggregation updates the current user turn and emits interruption before context."""
    aggregator = _bare_user_aggregator([{"role": "user", "content": "hello"}])
    aggregator._last_final_transcript = "hello"
    aggregator._aggregation = "there"

    _drive(aggregator.push_aggregation())

    assert aggregator.context.get_messages() == [{"role": "user", "content": "hello there"}]
    assert isinstance(aggregator.pushed[0][0], InterruptionFrame)
    assert isinstance(aggregator.pushed[1][0], OpenAILLMContextFrame)
    assert aggregator.reset_count == 1
    assert aggregator._aggregation == ""


def test_user_process_frame_sends_stable_interim_once_and_deduplicates_final():
    """Stable interim transcripts are pushed while duplicate finals are ignored."""
    aggregator = _bare_user_aggregator()
    aggregator._user_speaking = True
    interim = RivaInterimTranscriptionFrame(
        text="book a table",
        user_id="user",
        timestamp="now",
        stability=1.0,
    )

    _drive(aggregator.process_frame(interim, FrameDirection.DOWNSTREAM))
    _drive(aggregator.process_frame(TranscriptionFrame("book a table", "user", "now"), FrameDirection.DOWNSTREAM))

    assert aggregator.context.get_messages() == [{"role": "user", "content": "book a table"}]
    assert aggregator.last_transcript is None
    assert aggregator.seen_final is True
    assert aggregator._last_final_transcript == ""
    assert sum(isinstance(frame, OpenAILLMContextFrame) for frame, _ in aggregator.pushed) == 1


def test_assistant_push_aggregation_updates_existing_assistant_message():
    """Assistant aggregation replaces an in-progress assistant response in context."""
    aggregator = _bare_assistant_aggregator([{"role": "assistant", "content": "partial"}])
    aggregator._aggregation = "final answer"

    _drive(aggregator.push_aggregation())

    assert aggregator.context.get_messages() == [{"role": "assistant", "content": "final answer"}]
    assert isinstance(aggregator.pushed[0][0], OpenAILLMContextFrame)
    assert aggregator.reset_count == 1


def test_tts_response_cacher_releases_cached_response_after_user_stops_speaking():
    """TTS response frames are cached during user speech and released after speech stops."""
    cacher = NvidiaTTSResponseCacher()
    pushed = []

    async def _push_frame(frame, direction=None):
        """Capture pushed TTS frames."""
        pushed.append((frame, direction))

    cacher.push_frame = _push_frame

    _drive(cacher.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM))
    _drive(cacher.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM))
    _drive(cacher.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM))
    _drive(
        cacher.process_frame(
            TTSAudioRawFrame(audio=b"\x00\x01", sample_rate=16000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
    )
    _drive(
        cacher.process_frame(TTSTextFrame("hello", aggregated_by=AggregationType.SENTENCE), FrameDirection.DOWNSTREAM)
    )
    _drive(cacher.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM))
    _drive(cacher.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM))

    released_types = [type(frame) for frame, _ in pushed]
    assert released_types == [
        UserStartedSpeakingFrame,
        LLMFullResponseStartFrame,
        TTSStartedFrame,
        TTSAudioRawFrame,
        TTSTextFrame,
        LLMFullResponseEndFrame,
        UserStoppedSpeakingFrame,
    ]
    assert cacher._cache == []
    assert cacher.user_stopped_speaking is True


def test_tts_response_cacher_clears_cached_frames_on_interruption():
    """Interruptions clear speculative TTS cache and forward the interruption."""
    cacher = NvidiaTTSResponseCacher()
    pushed = []

    async def _push_frame(frame, direction=None):
        """Capture pushed frames."""
        pushed.append((frame, direction))

    cacher.push_frame = _push_frame

    _drive(cacher.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM))
    _drive(cacher.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM))
    _drive(cacher.process_frame(InterruptionFrame(), FrameDirection.UPSTREAM))

    assert cacher._cache == []
    assert isinstance(pushed[-1][0], InterruptionFrame)
    assert pushed[-1][1] is FrameDirection.UPSTREAM

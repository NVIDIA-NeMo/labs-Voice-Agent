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

"""Unit tests for runtime state machines that do not require live model services."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    StartInterruptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transcriptions.language import Language

from nemo_voice_agent.pipecat.services.nemo.turn_taking import NeMoTurnTakingService
from nemo_voice_agent.pipecat.services.nvidia_llm import NvidiaLLMService


def _drive(awaitable):
    """Run an async helper from a synchronous pytest test."""
    return asyncio.run(awaitable)


def _capture_pushes(service):
    """Patch a FrameProcessor's push_frame to capture emitted frames."""
    pushed = []

    async def _push_frame(frame, direction=None):
        """Capture pushed frames and directions."""
        pushed.append((frame, direction))

    service.push_frame = _push_frame
    return pushed


def _transcription(text: str) -> TranscriptionFrame:
    """Build a minimal transcription frame for turn-taking tests."""
    return TranscriptionFrame(text, "", "now", Language.EN_US, result={"text": text})


def _bare_nvidia_llm(**attrs):
    """Construct NvidiaLLMService without creating an OpenAI client."""
    service = NvidiaLLMService.__new__(NvidiaLLMService)
    service._mistral_model_support = attrs.pop("_mistral_model_support", False)
    service._is_processing = attrs.pop("_is_processing", False)
    service._prompt_tokens = attrs.pop("_prompt_tokens", 0)
    service._completion_tokens = attrs.pop("_completion_tokens", 0)
    service._total_tokens = attrs.pop("_total_tokens", 0)
    service._has_reported_prompt_tokens = attrs.pop("_has_reported_prompt_tokens", False)
    for name, value in attrs.items():
        setattr(service, name, value)
    service._reset_think_filter_state()
    return service


def test_turn_taking_loads_backchannels_from_list_and_cleans_text():
    """Backchannel phrases are normalized and EOU/EOB markers are stripped before matching."""
    service = NeMoTurnTakingService(backchannel_phrases=["Yeah!", "Uh huh"], use_vad=True)

    assert service.clean_text("  Yeah!!!<EOU>") == "yeah"
    assert service.clean_text("It's, okay.<EOB>") == "it's okay"
    assert service.is_backchannel("yeah") is True
    assert service.is_backchannel("<speaker_0> Uh huh") is True
    assert service.is_backchannel("not a backchannel") is False


def test_turn_taking_loads_backchannels_from_yaml_file(tmp_path):
    """Backchannel phrases can be loaded from a YAML file path."""
    path = tmp_path / "backchannels.yaml"
    path.write_text("- okay\n- yep\n")

    service = NeMoTurnTakingService(backchannel_phrases=str(path))

    assert service.is_backchannel("okay") is True
    assert service.is_backchannel("yep!") is True


def test_turn_taking_rejects_invalid_backchannel_source():
    """Invalid backchannel phrase sources raise a clear ValueError."""
    with pytest.raises(ValueError, match="Invalid backchannel phrases"):
        NeMoTurnTakingService(backchannel_phrases=object())


def test_turn_taking_vad_start_pushes_interruption_when_no_backchannels():
    """A VAD start event emits user-started and interruption frames when backchannels are disabled."""
    service = NeMoTurnTakingService(backchannel_phrases=None)
    pushed = _capture_pushes(service)

    _drive(service._handle_vad_user_started_speaking(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM))

    assert isinstance(pushed[0][0], VADUserStartedSpeakingFrame)
    assert any(isinstance(frame, UserStartedSpeakingFrame) for frame, _ in pushed)
    assert any(isinstance(frame, StartInterruptionFrame) for frame, _ in pushed)
    assert service._have_sent_user_started_speaking is True


def test_turn_taking_eou_transcription_emits_completed_turn_and_stop():
    """An EOU transcript while VAD is active emits start, final transcript, and stop frames."""
    service = NeMoTurnTakingService(backchannel_phrases=["yeah"], use_vad=True)
    pushed = _capture_pushes(service)
    service._vad_user_speaking = True

    _drive(service._handle_transcription(_transcription("book a table<EOU>"), FrameDirection.DOWNSTREAM))

    frames = [frame for frame, _ in pushed]
    assert any(isinstance(frame, UserStartedSpeakingFrame) for frame in frames)
    assert any(isinstance(frame, StartInterruptionFrame) for frame in frames)
    assert any(isinstance(frame, TranscriptionFrame) and frame.text == "book a table" for frame in frames)
    assert any(isinstance(frame, UserStoppedSpeakingFrame) for frame in frames)
    assert service._user_speaking_buffer == ""


def test_turn_taking_partial_buffer_emits_interim_completed_words():
    """Long partial transcripts emit completed words while keeping the trailing unfinished word buffered."""
    service = NeMoTurnTakingService(max_buffer_size=2, use_vad=True)
    pushed = _capture_pushes(service)
    service._vad_user_speaking = True

    _drive(service._handle_transcription(_transcription("hello there friend"), FrameDirection.DOWNSTREAM))

    assert any(isinstance(frame, InterimTranscriptionFrame) and frame.text == "hello there" for frame, _ in pushed)
    assert service._user_speaking_buffer == " friend"


def test_turn_taking_backchannel_while_bot_speaking_goes_upstream():
    """A backchannel EOU while the bot is speaking is wrapped and sent upstream."""
    service = NeMoTurnTakingService(backchannel_phrases=["yeah"], use_vad=True)
    pushed = _capture_pushes(service)
    service._vad_user_speaking = True
    service._bot_speaking = True

    _drive(service._handle_transcription(_transcription("yeah<EOU>"), FrameDirection.DOWNSTREAM))

    assert len(pushed) == 1
    assert isinstance(pushed[0][0], TranscriptionFrame)
    assert pushed[0][0].text == "(yeah)"
    assert pushed[0][1] is FrameDirection.UPSTREAM
    assert service._user_speaking_buffer == ""


def test_turn_taking_vad_stop_ignores_speaker_tag_only_buffer():
    """VAD stop does not emit a transcript when the buffer only contains a speaker tag."""
    service = NeMoTurnTakingService(use_diar=True)
    pushed = _capture_pushes(service)
    service._user_speaking_buffer = "<speaker_1>"

    _drive(service._handle_vad_user_stopped_speaking(SimpleNamespace(), FrameDirection.DOWNSTREAM))

    assert len(pushed) == 1
    assert pushed[0][0].__class__ is SimpleNamespace


def test_nvidia_llm_mistral_preprocessing_merges_consecutive_roles():
    """Mistral preprocessing combines adjacent messages with the same role."""
    service = _bare_nvidia_llm(_mistral_model_support=True)
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "hello"},
        {"role": "user", "content": "again"},
        {"role": "assistant", "content": "hi"},
        {"role": "assistant", "content": "there"},
    ]

    assert service._preprocess_messages_for_mistral(messages) == [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "hello again"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_nvidia_llm_mistral_preprocessing_noops_when_disabled():
    """Mistral preprocessing returns the original list when support is disabled."""
    service = _bare_nvidia_llm(_mistral_model_support=False)
    messages = [{"role": "user", "content": "hello"}]

    assert service._preprocess_messages_for_mistral(messages) is messages


def test_nvidia_llm_think_filter_handles_split_end_tag_and_then_streams_content():
    """Think-token filtering suppresses thought text until a split closing tag completes."""
    service = _bare_nvidia_llm()

    assert service._filter_think_token("<think>hidden") == ""
    assert service._filter_think_token("</thi") == ""
    assert service._filter_think_token("nk>visible") == "visible"
    assert service._filter_think_token(" next") == " next"


def test_nvidia_llm_think_filter_resets_state():
    """Resetting think-filter state clears buffers and end-tag tracking."""
    service = _bare_nvidia_llm()
    service._filter_think_token("<think>hidden")

    service._reset_think_filter_state()

    assert service._seen_end_tag is False
    assert service._buffer == ""
    assert service._thinking_aggregation == ""
    assert service._partial_tag_buffer == ""


@pytest.mark.parametrize(
    "status, expected",
    [
        (401, "bad API key"),
        (404, "wrong URL or model"),
        (500, "HTTP 500"),
    ],
)
def test_nvidia_llm_error_mapping_for_http_statuses(status, expected):
    """HTTP status errors are re-raised with endpoint/model configuration hints."""
    service = _bare_nvidia_llm()
    request = httpx.Request("GET", "https://example.test/v1")
    response = httpx.Response(status, request=request)
    err = httpx.HTTPStatusError("failed", request=request, response=response)

    with pytest.raises(ValueError, match=expected):
        service._raise_llm_error(err)


def test_nvidia_llm_error_mapping_for_connectivity_and_auth_strings():
    """Connectivity and auth-looking plain errors are mapped to actionable ValueErrors."""
    service = _bare_nvidia_llm()

    with pytest.raises(ValueError, match="Cannot connect"):
        service._raise_llm_error(httpx.ConnectError("connection refused"))
    with pytest.raises(ValueError, match="bad API key"):
        service._raise_llm_error(RuntimeError("401 unauthorized"))
    with pytest.raises(ValueError, match="model not found"):
        service._raise_llm_error(RuntimeError("model not found"))


def test_nvidia_llm_usage_metrics_accumulate_only_while_processing():
    """Incremental NVIDIA token usage is accumulated only during active processing."""
    service = _bare_nvidia_llm(_is_processing=True)

    _drive(service.start_llm_usage_metrics(LLMTokenUsage(prompt_tokens=5, completion_tokens=1, total_tokens=6)))
    _drive(service.start_llm_usage_metrics(LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10)))
    service._is_processing = False
    _drive(service.start_llm_usage_metrics(LLMTokenUsage(prompt_tokens=9, completion_tokens=8, total_tokens=17)))

    assert service._prompt_tokens == 5
    assert service._completion_tokens == 3
    assert service._has_reported_prompt_tokens is True

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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
import numpy as np
import pytest
from pipecat.audio.vad.vad_analyzer import VADState
from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMRunFrame,
    StartInterruptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transcriptions.language import Language

from nemo_voice_agent.pipecat.services.common import UserAudioBuffer
from nemo_voice_agent.pipecat.services.nemo.stt import NeMoSTTInputParams, NemoSTTService
from nemo_voice_agent.pipecat.services.nemo.tts import BaseNemoTTSService
from nemo_voice_agent.pipecat.services.nemo.turn_taking import NeMoTurnTakingService
from nemo_voice_agent.pipecat.services.nvidia_llm import NvidiaLLMService
from nemo_voice_agent.pipecat.transports.base_input import BaseInputTransport
from nemo_voice_agent.pipecat.transports.network.websocket_server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)


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


def _bare_tts(**attrs):
    """Construct BaseNemoTTSService without loading a real TTS model."""
    service = BaseNemoTTSService.__new__(BaseNemoTTSService)
    service._think_tokens = attrs.pop("_think_tokens", None)
    service._have_seen_think_tokens = attrs.pop("_have_seen_think_tokens", False)
    service._ignore_strings = attrs.pop("_ignore_strings", None)
    for name, value in attrs.items():
        setattr(service, name, value)
    return service


def _bare_stt(fake_model, *, has_turn_taking=False):
    """Construct NemoSTTService with a fake streaming ASR model."""
    service = NemoSTTService.__new__(NemoSTTService)
    service._audio_logger = None
    service._audio_buffer = bytearray()
    service._audio_timestamps = []
    service._has_logged_audio_chunk = False
    service._bytes_per_buffer = 4
    service._params = NeMoSTTInputParams(buffer_size=1)
    service._model = fake_model
    service._sample_rate = 16000
    service._has_generated_metrics = False
    service._has_turn_taking = has_turn_taking
    service._user_id = ""
    service._model_name = "fake-asr"
    service._backend = "legacy"
    service.user_is_speaking = False
    service._is_vad_active = False

    async def _noop(*_args, **_kwargs):
        """Metrics no-op for bare service tests."""
        return None

    service.start_ttfb_metrics = _noop
    service.stop_ttfb_metrics = _noop
    service.start_processing_metrics = _noop
    service.stop_processing_metrics = _noop
    return service


class _FakeContext:
    """Fake LLM context that captures audio-frame messages."""

    def __init__(self, messages=None):
        """Initialize context messages and add-audio call capture."""
        self.messages = list(messages or [])
        self.audio_messages = []

    async def add_audio_frames_message(self, audio_frames, text):
        """Capture the added audio turn and mirror it into context messages."""
        self.audio_messages.append((list(audio_frames), text))
        self.messages.append({"role": "user", "content": list(audio_frames)})


class _FakeAggregator:
    """Fake user context aggregator that captures pushed frames."""

    def __init__(self):
        """Initialize pushed frame capture."""
        self.frames = []

    async def push_frame(self, frame):
        """Capture one frame."""
        self.frames.append(frame)


def _audio_frame(value: int = 1) -> InputAudioRawFrame:
    """Build a small input audio frame."""
    return InputAudioRawFrame(audio=np.array([value], dtype=np.int16).tobytes(), sample_rate=16000, num_channels=1)


def _bare_input_transport(*, new_vad_state, can_create_user_frames=True, turn_analyzer=None):
    """Construct BaseInputTransport with fake VAD analysis and frame capture."""
    transport = BaseInputTransport.__new__(BaseInputTransport)
    transport._params = SimpleNamespace(
        turn_analyzer=turn_analyzer,
        can_create_user_frames=can_create_user_frames,
        audio_in_enabled=True,
    )
    transport._paused = False
    transport._audio_in_queue = asyncio.Queue()
    transport._vad_analyze = lambda _frame: asyncio.sleep(0, result=new_vad_state)
    transport.pushed = []
    transport.interruptions = []

    async def _push_frame(frame, direction=None):
        """Capture pushed VAD/user frames."""
        transport.pushed.append((frame, direction))

    async def _handle_user_interruption(vad_state):
        """Capture interruption handling calls."""
        transport.interruptions.append(vad_state)

    transport.push_frame = _push_frame
    transport._handle_user_interruption = _handle_user_interruption
    return transport


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


def test_base_nemo_tts_think_token_filtering_tracks_multi_chunk_thoughts():
    """TTS think-token filtering suppresses thought chunks and resumes after the end marker."""
    service = _bare_tts(_think_tokens=["<think>", "</think>"])

    assert service._handle_think_tokens("before <think>hidden") == "before "
    assert service._have_seen_think_tokens is True
    assert service._handle_think_tokens("still hidden") is None
    assert service._handle_think_tokens("done</think> visible") == " visible"
    assert service._have_seen_think_tokens is False
    assert service._handle_think_tokens("<think>hidden</think> answer") == " answer"


def test_base_nemo_tts_drops_special_tokens_and_converts_audio_formats():
    """TTS helpers remove ignored strings and convert common audio containers to bytes."""
    service = _bare_tts(_ignore_strings={"<noise>", "[silence]"})

    assert service._drop_special_tokens("hello <noise> world [silence]") == "hello  world "
    assert service._convert_to_bytes(b"abc") == b"abc"
    assert service._convert_to_bytes(bytearray(b"def")) == b"def"
    assert (
        service._convert_to_bytes(np.array([0.0, 1.0, -1.0], dtype=np.float32))
        == np.array([0, 32767, -32767], dtype=np.int16).tobytes()
    )
    assert (
        service._convert_to_bytes(np.array([1, 2, 3], dtype=np.int32)) == np.array([1, 2, 3], dtype=np.int16).tobytes()
    )


def test_nemo_stt_run_stt_emits_final_transcription_from_fake_model():
    """NemoSTTService buffering emits a final transcription frame from a fake ASR result."""
    fake_model = SimpleNamespace(
        transcribe=lambda _audio: SimpleNamespace(
            text="hello",
            is_final=True,
            eou_latency=0.1,
            eob_latency=None,
            eou_prob=0.9,
            eob_prob=None,
            processing_time=0.01,
        )
    )
    service = _bare_stt(fake_model, has_turn_taking=False)

    frames = _drive(_collect_async(service.run_stt(np.array([1, 2], dtype=np.int16).tobytes())))

    assert len(frames) == 1
    assert isinstance(frames[0], TranscriptionFrame)
    assert frames[0].text == "hello"
    assert service.user_is_speaking is False
    assert service._has_generated_metrics is True


def test_nemo_stt_run_stt_uses_interim_frames_when_turn_taking_enabled():
    """Turn-taking mode emits interim frames even when the ASR result is final."""
    fake_model = SimpleNamespace(
        transcribe=lambda _audio: SimpleNamespace(
            text="partial",
            is_final=True,
            eou_latency=None,
            eob_latency=None,
            eou_prob=None,
            eob_prob=None,
            processing_time=0.01,
        )
    )
    service = _bare_stt(fake_model, has_turn_taking=True)

    frames = _drive(_collect_async(service.run_stt(np.array([1, 2], dtype=np.int16).tobytes())))

    assert len(frames) == 1
    assert isinstance(frames[0], InterimTranscriptionFrame)
    assert frames[0].text == "partial"
    assert service.user_is_speaking is True


def test_nemo_stt_run_stt_wraps_model_errors_in_error_frame():
    """ASR exceptions are converted to ErrorFrame instances instead of escaping."""

    def _raise(_audio):
        """Raise a fake ASR failure."""
        raise RuntimeError("asr boom")

    service = _bare_stt(SimpleNamespace(transcribe=_raise))

    frames = _drive(_collect_async(service.run_stt(np.array([1, 2], dtype=np.int16).tobytes())))

    assert len(frames) == 1
    assert isinstance(frames[0], ErrorFrame)
    assert "asr boom" in frames[0].error


async def _collect_async(async_iterable):
    """Collect an async iterable into a list."""
    return [item async for item in async_iterable]


class _FakeWebsocketOutput:
    """Fake websocket output transport for callback routing tests."""

    def __init__(self):
        """Initialize captured connections."""
        self.connections = []

    async def set_client_connection(self, websocket):
        """Capture the websocket assigned by the transport callback."""
        self.connections.append(websocket)


def test_websocket_transport_reuses_input_and_output_instances():
    """WebsocketServerTransport lazily creates and caches input/output transports."""
    transport = WebsocketServerTransport(WebsocketServerParams(), host="127.0.0.1", port=9999)

    assert transport.input() is transport.input()
    assert transport.output() is transport.output()


def test_websocket_transport_connection_callbacks_route_to_output_and_handlers():
    """Client connect/disconnect callbacks update output transport and call registered handlers."""
    transport = WebsocketServerTransport(WebsocketServerParams())
    output = _FakeWebsocketOutput()
    events = []
    websocket = object()
    transport._output = output

    async def _call_event_handler(name, *args):
        """Capture event handler calls."""
        events.append((name, args))

    transport._call_event_handler = _call_event_handler

    _drive(transport._on_client_connected(websocket))
    _drive(transport._on_client_disconnected(websocket))
    _drive(transport._on_session_timeout(websocket))
    _drive(transport._on_websocket_ready())

    assert output.connections == [websocket, None]
    assert events == [
        ("on_client_connected", (websocket,)),
        ("on_client_disconnected", (websocket,)),
        ("on_session_timeout", (websocket,)),
        ("on_websocket_ready", ()),
    ]


def test_user_audio_buffer_keeps_preroll_then_adds_audio_turn_with_transcript():
    """UserAudioBuffer keeps pre-roll audio, appends speech frames, and emits an LLM run frame."""
    context = _FakeContext()
    aggregator = _FakeAggregator()
    service = UserAudioBuffer(
        context=context,
        user_context_aggregator=aggregator,
        pre_cache_duration_secs=0.032,
        raw_audio_frame_len_in_secs=0.016,
        use_transcript=True,
    )
    pushed = _capture_pushes(service)

    _drive(service.process_frame(_audio_frame(1), FrameDirection.DOWNSTREAM))
    _drive(service.process_frame(_audio_frame(2), FrameDirection.DOWNSTREAM))
    _drive(service.process_frame(_audio_frame(3), FrameDirection.DOWNSTREAM))
    _drive(service.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM))
    _drive(service.process_frame(_audio_frame(4), FrameDirection.DOWNSTREAM))
    _drive(service.process_frame(TranscriptionFrame("hello there", "", "now"), FrameDirection.DOWNSTREAM))
    _drive(service.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM))

    audio_frames, text = context.audio_messages[0]
    assert len(audio_frames) == 3
    assert "Follow instructions" in text
    assert "hello there" in text
    assert isinstance(aggregator.frames[0], LLMRunFrame)
    assert service._audio_frames == []
    assert service._transcript_buffer == []
    assert any(isinstance(frame, UserStoppedSpeakingFrame) for frame, _ in pushed)


def test_user_audio_buffer_replaces_previous_audio_turn_when_configured():
    """When configured for only the latest audio turn, older audio content is replaced by transcript text."""
    context = _FakeContext(messages=[{"role": "user", "content": [_audio_frame(1)]}])
    aggregator = _FakeAggregator()
    service = UserAudioBuffer(
        context=context,
        user_context_aggregator=aggregator,
        use_transcript=True,
        keep_only_last_audio_turn=True,
    )
    service._previsous_user_text = "previous transcript"
    service._audio_frames = [_audio_frame(2)]
    service._transcript_buffer = ["new transcript"]

    _drive(service.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM))

    assert context.messages[0]["content"] == "previous transcript"
    assert context.audio_messages[0][1].endswith("new transcript")


def test_user_audio_buffer_reset_clears_audio_and_transcript_state():
    """UserAudioBuffer.reset clears buffered audio, transcript, and speaking state."""
    service = UserAudioBuffer(context=_FakeContext(), user_context_aggregator=_FakeAggregator())
    service._audio_frames = [_audio_frame()]
    service._transcript_buffer = ["hello"]
    service._user_speaking = True

    service.reset()

    assert service._audio_frames == []
    assert service._transcript_buffer == []
    assert service._user_speaking is False


def test_base_input_transport_vad_speaking_emits_vad_and_user_frames():
    """BaseInputTransport emits VAD start and handles user interruption on speaking transitions."""
    transport = _bare_input_transport(new_vad_state=VADState.SPEAKING)

    state = _drive(transport._handle_vad(_audio_frame(), VADState.QUIET))

    assert state is VADState.SPEAKING
    assert any(isinstance(frame, VADUserStartedSpeakingFrame) for frame, _ in transport.pushed)
    assert transport.interruptions == [VADState.SPEAKING]


def test_base_input_transport_vad_quiet_respects_turn_analyzer_gate():
    """Turn-analyzer speech-triggered state suppresses user-frame interruption handling."""
    turn_analyzer = SimpleNamespace(speech_triggered=True)
    transport = _bare_input_transport(new_vad_state=VADState.QUIET, turn_analyzer=turn_analyzer)

    state = _drive(transport._handle_vad(_audio_frame(), VADState.SPEAKING))

    assert state is VADState.QUIET
    assert any(isinstance(frame, VADUserStoppedSpeakingFrame) for frame, _ in transport.pushed)
    assert transport.interruptions == []


def test_base_input_transport_push_audio_frame_sets_timestamp_and_queues_when_enabled():
    """Audio frames are timestamped and queued only when audio input is enabled and not paused."""
    transport = _bare_input_transport(new_vad_state=VADState.SPEAKING)
    frame = _audio_frame()

    _drive(transport.push_audio_frame(frame))

    assert frame.timestamp is not None
    assert transport._audio_in_queue.get_nowait() is frame


def test_base_input_transport_push_audio_frame_does_not_queue_when_paused():
    """Paused audio input still timestamps frames but does not enqueue them."""
    transport = _bare_input_transport(new_vad_state=VADState.SPEAKING)
    transport._paused = True
    frame = _audio_frame()

    _drive(transport.push_audio_frame(frame))

    assert frame.timestamp is not None
    assert transport._audio_in_queue.empty()


def test_turn_taking_missing_backchannel_file_blames_the_file_not_the_type(tmp_path):
    """A bad path must raise FileNotFoundError naming the path.

    This used to fall through the string branch (which also required is_file())
    into the type-error branch, reporting "Invalid backchannel phrases of type
    <class 'str'>" — which pointed at the wrong thing entirely.
    """
    missing = str(tmp_path / "no_such_backchannels.yaml")
    with pytest.raises(FileNotFoundError) as excinfo:
        NeMoTurnTakingService(backchannel_phrases=missing, use_vad=True)
    assert missing in str(excinfo.value)


def test_turn_taking_rejects_genuinely_wrong_types():
    """Values that are neither a path nor a list still raise ValueError."""
    with pytest.raises(ValueError, match="Invalid backchannel phrases of type"):
        NeMoTurnTakingService(backchannel_phrases={"not": "supported"}, use_vad=True)


def test_turn_taking_loads_backchannels_from_yaml_file(tmp_path):
    """A real file is read and its phrases normalized for matching."""
    path = tmp_path / "backchannels.yaml"
    path.write_text("- Uh huh\n- Yeah!\n")

    service = NeMoTurnTakingService(backchannel_phrases=str(path), use_vad=True)

    assert service.is_backchannel("uh huh") is True
    assert service.is_backchannel("what is the weather") is False

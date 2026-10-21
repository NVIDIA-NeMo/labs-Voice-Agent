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

"""Unit coverage for lightweight service builders and server orchestration."""

import asyncio
import base64
import json
import wave
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    EndWorkerFrame,
    InputAudioRawFrame,
    LLMRunFrame,
    StopFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
)
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies, UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager

from nemo_voice_agent.evaluation.tools.waitlist_tools import DropWaitListTool, GetWaitlistTool, JoinWaitListTool
from nemo_voice_agent.pipecat import bot_server
from nemo_voice_agent.pipecat.services.common import UserAudioBuffer
from nemo_voice_agent.pipecat.services.nemo import builders
from nemo_voice_agent.pipecat.utils.text.simple_text_aggregator import SimpleSegmentedTextAggregator


def _config(**overrides):
    """Build the small ConfigManager surface consumed by builders."""
    server_config = SimpleNamespace(
        transport={
            "audio_in_sample_rate": 8000,
            "audio_out_sample_rate": 24000,
        },
        diar={"enabled": True},
        turn_taking={"type": "nemo"},
        stt=SimpleNamespace(kind="stt"),
        tts=SimpleNamespace(kind="tts"),
        llm={"kind": "llm", "inject_dummy_user_message": True, "dummy_user_message": "Hi"},
        server={"log_file": "voice.log", "log_level": "INFO", "create_new_log": True},
    )
    config = SimpleNamespace(
        server_config=server_config,
        SAMPLE_RATE=16000,
        TRANSPORT_AUDIO_OUT_10MS_CHUNKS=7,
        DIAR_MODEL="diar-model",
        STT_DEVICE="cpu",
        USE_DIAR=True,
        TURN_TAKING_TYPE="nemo",
        TURN_TAKING_MAX_BUFFER_SIZE=3,
        TURN_TAKING_BOT_STOP_DELAY=0.25,
        TURN_TAKING_BACKCHANNEL_PHRASES_PATH=["yeah"],
        TURN_TAKING_USER_SPEECH_TIMEOUT=0.6,
        TURN_TAKING_USER_TURN_STOP_TIMEOUT=None,
        SYSTEM_ROLE="system",
        SYSTEM_PROMPT="policy",
        get_vad_params=lambda: "vad-params",
        get_diar_params=lambda: "diar-params",
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def _set_turn_taking(config, **turn_taking):
    """Rewrite the ``turn_taking`` block the way ConfigManager would.

    ConfigManager keeps the raw block and the derived ``TURN_TAKING_*``
    attributes in agreement, so tests must too — reading one but not the other
    would let a builder keep the stale type.
    """
    config.server_config.turn_taking = dict(turn_taking)
    config.TURN_TAKING_TYPE = turn_taking.get("type", "nemo")
    config.TURN_TAKING_USER_SPEECH_TIMEOUT = turn_taking.get("user_speech_timeout", 0.6)
    config.TURN_TAKING_USER_TURN_STOP_TIMEOUT = turn_taking.get("user_turn_stop_timeout") or None
    return config


def test_builder_factories_delegate_and_preserve_configuration(monkeypatch):
    """Thin builders pass resolved configuration to their dependencies."""
    config = _config()
    monkeypatch.setattr(builders, "SileroVADAnalyzer", lambda **kwargs: ("vad", kwargs))
    monkeypatch.setattr(builders, "ProtobufFrameSerializer", lambda: "serializer")
    monkeypatch.setattr(builders, "SingleClientWebsocketServerParams", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(builders, "SingleClientWebsocketServerTransport", lambda **kwargs: ("transport", kwargs))
    monkeypatch.setattr(builders, "get_stt_service_from_config", lambda cfg, logger: ("stt", cfg, logger))
    monkeypatch.setattr(builders, "get_tts_service_from_config", lambda cfg, logger: ("tts", cfg, logger))
    monkeypatch.setattr(builders, "get_llm_service_from_config", lambda cfg: ("llm", cfg))

    vad = builders.build_vad_analyzer(config)
    assert vad == ("vad", {"sample_rate": 8000, "params": "vad-params"})
    transport = builders.build_ws_transport(config, vad, "0.0.0.0", 8765)
    assert transport[1]["host"] == "0.0.0.0"
    assert transport[1]["port"] == 8765
    params = transport[1]["params"]
    assert params.serializer == "serializer"
    assert params.audio_in_sample_rate == 8000
    assert params.audio_out_sample_rate == 24000
    assert params.audio_out_10ms_chunks == 7
    assert builders.build_stt(config, "logger")[0] == "stt"
    assert builders.build_tts(config, "logger")[0] == "tts"
    assert builders.build_llm(config)[0] == "llm"


def test_optional_audio_diar_and_turn_taking_builders(monkeypatch):
    """Optional builders cover disabled and configured service paths."""
    config = _config()
    created = []

    class FakeAudioLogger:
        def __init__(self, **kwargs):
            created.append(("audio", kwargs))

    monkeypatch.setattr(builders, "AudioLogger", FakeAudioLogger)
    assert builders.build_audio_logger(config) is None
    config.server_config.transport["record_audio_data"] = True
    config.server_config.transport["audio_log_dir"] = "/tmp/audio"
    assert isinstance(builders.build_audio_logger(config), FakeAudioLogger)
    assert created[-1][1]["log_dir"] == "/tmp/audio"
    assert created[-1][1]["session_id"].startswith("session_")

    monkeypatch.setattr(builders, "NemoDiarService", lambda **kwargs: ("diar", kwargs))
    diar = builders.build_diar(config, "logger")
    assert diar[1] == {
        "model": "diar-model",
        "device": "cpu",
        "params": "diar-params",
        "sample_rate": 16000,
        "backend": "legacy",
        "enabled": True,
    }
    config.server_config.diar["enabled"] = False
    assert builders.build_diar(config) is None

    monkeypatch.setattr(builders, "NeMoTurnTakingService", lambda **kwargs: ("turn", kwargs))
    turn = builders.build_turn_taking(config, "logger", use_diar=False, use_vad=False)
    assert turn[1]["use_diar"] is False
    assert turn[1]["use_vad"] is False
    assert turn[1]["audio_logger"] == "logger"
    _set_turn_taking(config, type="speech_timeout")
    assert builders.build_turn_taking(config) is None


def test_build_turn_taking_defaults_to_nemo_when_type_is_omitted(monkeypatch):
    """An omitted ``turn_taking.type`` still builds the NeMo service."""
    monkeypatch.setattr(builders, "NeMoTurnTakingService", lambda **kwargs: ("turn", kwargs))
    config = _set_turn_taking(_config(), max_buffer_size=2)

    assert builders.build_turn_taking(config)[0] == "turn"


def test_context_and_logging_builders(monkeypatch):
    """Context construction preserves prompts and returns both aggregators."""
    config = _config()

    context, user, assistant, original = builders.build_context_and_aggregators(object(), config)
    assert context.get_messages() == [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "Hi"},
    ]
    assert isinstance(user, LLMUserAggregator)
    assert isinstance(assistant, LLMAssistantAggregator)
    # turn_taking.type is nemo in _config(), so NeMoTurnTakingService owns the
    # turn boundaries and the aggregator must defer to it rather than deriving
    # turns from VAD itself.
    assert isinstance(user._params.user_turn_strategies, ExternalUserTurnStrategies)
    assert original == context.get_messages()
    assert original is not context.get_messages()
    assert builders.resolve_log_file_path(config) == ("voice.log", "INFO", True)
    assert builders.overwrite_existing_log(config) is False
    config.server_config.server["overwrite_existing_log"] = 1
    assert builders.overwrite_existing_log(config) is True


class _Registrar:
    def __init__(self):
        self.handlers = {}

    def event_handler(self, name):
        def decorator(handler):
            self.handlers[name] = handler
            return handler

        return decorator


class _RTVI(_Registrar):
    def __init__(self):
        super().__init__()
        self._client_ready = True
        self._bot_ready = True
        self.ready_calls = 0

    async def set_bot_ready(self):
        self.ready_calls += 1


class _Task:
    def __init__(self):
        self.queued = []

    async def queue_frames(self, frames):
        self.queued.extend(frames)


def test_run_bot_websocket_server_exercises_registered_lifecycle(monkeypatch):
    """Registered handlers initialize, reset, and finalize a session.

    Notably they do *not* terminate the pipeline: see the assertion below.
    """
    ws = _Registrar()
    rtvi = _RTVI()
    task = _Task()
    task_ref = SimpleNamespace(task=None, running=False)
    audio_logger = SimpleNamespace(finalize_calls=0)
    audio_logger.finalize_session = lambda: setattr(audio_logger, "finalize_calls", audio_logger.finalize_calls + 1)
    reset_service = SimpleNamespace(reset_calls=0)
    reset_service.reset = lambda: setattr(reset_service, "reset_calls", reset_service.reset_calls + 1)
    disconnected = []

    async def hook():
        disconnected.append(True)

    class Runner:
        async def run(self, passed_task):
            assert passed_task is task
            client = SimpleNamespace(remote_address=("127.0.0.1", 1234))
            await rtvi.handlers["on_client_ready"](rtvi)
            await ws.handlers["on_client_connected"](ws, client)
            await ws.handlers["on_client_disconnected"](ws, client)
            await ws.handlers["on_session_timeout"](ws, client)

    monkeypatch.setattr(bot_server, "PipelineRunner", Runner)
    asyncio.run(
        bot_server.run_bot_websocket_server(
            task,
            ws,
            rtvi,
            task_ref=task_ref,
            audio_logger=audio_logger,
            initial_frame_factory=lambda: "initial",
            on_disconnect_reset_services=[None, reset_service],
            on_disconnect_hook=hook,
        )
    )
    assert task_ref.task is task
    assert task_ref.running is False
    assert rtvi.ready_calls == 1
    assert rtvi._client_ready is False
    assert rtvi._bot_ready is False
    assert task.queued[0] == "initial"
    # Disconnect and session-timeout must NOT end the pipeline. The WebSocket
    # server lives inside the input transport, so ending it would stop the
    # server and make the bot unreachable for the rest of the process. Both
    # handlers reset services instead.
    assert not any(isinstance(frame, (EndWorkerFrame, EndFrame, StopFrame)) for frame in task.queued)
    assert reset_service.reset_calls == 2
    assert disconnected == [True]
    assert audio_logger.finalize_calls == 3


def test_run_bot_websocket_server_handles_queue_and_runner_errors(monkeypatch):
    """Normal websocket closure and runner errors remain contained."""
    ws = _Registrar()
    rtvi = _RTVI()

    class BrokenTask:
        async def queue_frames(self, frames):
            raise RuntimeError("ConnectionClosedOK 1005")

    class Runner:
        async def run(self, task):
            client = SimpleNamespace(remote_address="client")
            await ws.handlers["on_client_disconnected"](ws, client)
            raise RuntimeError("runner failed")

    monkeypatch.setattr(bot_server, "PipelineRunner", Runner)
    asyncio.run(bot_server.run_bot_websocket_server(BrokenTask(), ws, rtvi, talk_first=False))


def test_fastapi_app_and_concurrent_runner(monkeypatch):
    """HTTP discovery and the uvicorn wrapper expose their configured values."""
    app = bot_server.create_fastapi_app(8765, public_host="voice.example", ws_scheme="wss")
    with TestClient(app) as client:
        assert client.post("/connect").json() == {"ws_url": "wss://voice.example:8765"}
        with client.websocket_connect("/ws"):
            pass

    seen = {}
    monkeypatch.setattr(bot_server.uvicorn, "Config", lambda app, host, port: (app, host, port))

    class Server:
        def __init__(self, config):
            seen["config"] = config

        async def serve(self):
            seen["served"] = True

    monkeypatch.setattr(bot_server.uvicorn, "Server", Server)

    async def ws_coro():
        seen["ws"] = True

    asyncio.run(bot_server.run_bot_with_fastapi(ws_coro(), app, "0.0.0.0", 8080))
    assert seen["config"][1:] == ("0.0.0.0", 8080)
    assert seen["served"] is seen["ws"] is True


def test_audio_context_encodes_wav_and_adds_message():
    """Audio context messages contain a valid WAV and optionally persist it."""
    frames = [AudioRawFrame(b"\x00\x01" * 8, 16000, 1), AudioRawFrame(b"\x02\x03" * 4, 16000, 1)]
    message = asyncio.run(LLMContext.create_audio_message(role="developer", audio_frames=frames, text="listen"))
    assert message["role"] == "developer"
    assert message["content"][0] == {"type": "text", "text": "listen"}
    wav_bytes = base64.b64decode(message["content"][1]["input_audio"]["data"])
    with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getnchannels() == 1
        assert wav_file.getnframes() == 12

    context = LLMContext(messages=[])
    asyncio.run(context.add_audio_frames_message(audio_frames=frames, text="audio"))
    assert context.get_messages()[0]["content"][0]["text"] == "audio"


class _Params:
    def __init__(self, arguments):
        self.arguments = arguments
        self.result = None
        # Every delivery is recorded so tests can assert that
        # ``StandardSchemaTool.__call__`` hands pipecat exactly one result per
        # call — the double-delivery it used to emit wedges pipecat 1.x.
        self.results = []

    async def result_callback(self, result):
        self.result = result
        self.results.append(result)


def test_waitlist_tools_cover_join_drop_get_and_not_found():
    """Waitlist tools mutate shared state and report both drop outcomes."""
    state = {"waitlist": [{"name": "Ada", "phone": "1", "party_size": 2}]}
    summaries = []

    async def summary(value):
        summaries.append(json.loads(value))

    join = JoinWaitListTool(shared_state=state)
    join.send_scenario_summary = summary
    params = _Params({"name": "Grace", "phone": "2", "party_size": 3})
    asyncio.run(join(params))
    assert len(params.results) == 1
    assert params.result["position"] == 2
    assert join.required_properties == ["name", "phone", "party_size"]
    assert set(join.properties) == {"name", "phone", "party_size"}

    drop = DropWaitListTool(shared_state=state)
    drop.send_scenario_summary = summary
    params = _Params({"name": "Ada"})
    asyncio.run(drop(params))
    assert len(params.results) == 1
    assert params.result["success"] is True
    params = _Params({"name": "missing"})
    asyncio.run(drop(params))
    assert len(params.results) == 1
    assert params.result["success"] is False
    assert drop.required_properties == ["name"]

    getter = GetWaitlistTool(shared_state=state)
    params = _Params({})
    asyncio.run(getter(params))
    assert len(params.results) == 1
    assert params.result["total_in_waitlist"] == 1
    assert getter.properties == {}


def test_llm_text_processor_carries_the_custom_segmented_aggregator():
    """The custom TTS aggregator must reach an LLMTextProcessor, not the TTS service.

    Pipecat 1.0 removed ``TTSService(text_aggregator=...)`` and *silently
    ignores* unknown constructor kwargs, so a regression here would degrade TTS
    chunking back to plain sentence splitting without raising anything. Assert
    the wiring explicitly.
    """
    config = _config()
    config.server_config.tts = OmegaConf.create({"type": "nemo", "min_sentence_length": 9})

    processor = builders.build_llm_text_processor(config)

    assert isinstance(processor, LLMTextProcessor)
    assert isinstance(processor._text_aggregator, SimpleSegmentedTextAggregator)
    assert processor._text_aggregator._min_sentence_length == 9

    config.server_config.tts = OmegaConf.create({"type": "nemo", "use_text_aggregator": False})
    assert builders.build_llm_text_processor(config) is None


def _turn_strategies(config, turn_taking):
    """Return the user-turn strategies build_context_and_aggregators selects."""
    _, user_agg, _, _ = builders.build_context_and_aggregators(object(), config, turn_taking)
    return user_agg._params.user_turn_strategies


def test_exactly_one_component_emits_user_speaking_frames():
    """Whoever owns turn detection is the sole emitter of the user speaking frames.

    In pipecat 1.x the LLMUserAggregator — not the transport and not VAD —
    emits UserStartedSpeakingFrame / UserStoppedSpeakingFrame, gated on each
    strategy's ``enable_user_speaking_frames``. So when NeMoTurnTakingService is
    in the pipeline the aggregator must stay quiet (or both would emit), and
    when it is absent the aggregator must emit (or nothing would).
    """
    config = _config()

    # Turn-taking present: it pushes the frames, so the aggregator defers.
    with_service = _turn_strategies(config, object())
    assert isinstance(with_service, ExternalUserTurnStrategies)
    assert not any(s._enable_user_speaking_frames for s in with_service.start)
    assert not any(s._enable_user_speaking_frames for s in with_service.stop)

    # Turn-taking absent: VAD drives the turn and the aggregator emits.
    _set_turn_taking(config, type="speech_timeout")
    without_service = _turn_strategies(config, None)
    assert not isinstance(without_service, ExternalUserTurnStrategies)
    assert all(s._enable_user_speaking_frames for s in without_service.start)
    assert [type(s).__name__ for s in without_service.start] == ["VADUserTurnStartStrategy"]
    # Named explicitly so we don't inherit pipecat's default stop strategy,
    # which would pull in LocalSmartTurnAnalyzerV3.
    assert [type(s).__name__ for s in without_service.stop] == ["SpeechTimeoutUserTurnStopStrategy"]


def test_turn_taking_instance_overrides_the_config_type():
    """The passed-in service decides, not a re-derived read of turn_taking.type.

    A bot may construct its turn-taking service inline; re-deriving the answer
    from config would then silently pick the wrong strategy pair.
    """
    config = _config()
    _set_turn_taking(config, type="speech_timeout")

    assert isinstance(_turn_strategies(config, object()), ExternalUserTurnStrategies)
    assert not isinstance(_turn_strategies(config, None), ExternalUserTurnStrategies)


def test_omitted_turn_taking_type_selects_the_external_turn_strategies():
    """No ``type`` means ``nemo``, so the aggregator must defer to the service."""
    config = _set_turn_taking(_config(), max_buffer_size=2)

    assert isinstance(_turn_strategies(config, None), ExternalUserTurnStrategies)


def test_configured_user_speech_timeout_reaches_the_stop_strategy():
    """``turn_taking.user_speech_timeout`` must land on the pipecat strategy.

    The strategy waits this long *after* ``vad.stop_secs`` before ending the
    turn, so dropping the configured value silently restores pipecat's 0.6s and
    the end-of-turn latency the key exists to control.
    """
    config = _set_turn_taking(_config(), type="speech_timeout", user_speech_timeout=0.25)

    (stop,) = _turn_strategies(config, None).stop
    assert isinstance(stop, SpeechTimeoutUserTurnStopStrategy)
    assert stop._user_speech_timeout == 0.25


def test_default_user_speech_timeout_matches_pipecat():
    """The shipped default carries pipecat's own 0.6s, not a silent zero."""
    config = _set_turn_taking(_config(), type="speech_timeout")

    (stop,) = _turn_strategies(config, None).stop
    assert stop._user_speech_timeout == 0.6


def _user_turn_stop_timeout(config, turn_taking):
    """Return the watchdog timeout build_context_and_aggregators gives the user aggregator."""
    _, user_agg, _, _ = builders.build_context_and_aggregators(object(), config, turn_taking)
    return user_agg._params.user_turn_stop_timeout


def test_user_turn_stop_watchdog_is_disabled_by_default():
    """Both turn paths get no watchdog, not pipecat's 5s default.

    With the watchdog on, an offline STT reply slower than 5s arrived after the
    turn had been force-ended, and its text was prepended to the next user turn.
    """
    assert _user_turn_stop_timeout(_config(), object()) is None
    assert _user_turn_stop_timeout(_set_turn_taking(_config(), type="speech_timeout"), None) is None


def test_configured_user_turn_stop_timeout_reaches_the_aggregator():
    """A configured watchdog lands on the aggregator for either turn path."""
    config = _set_turn_taking(_config(), type="speech_timeout", user_turn_stop_timeout=30.0)
    assert _user_turn_stop_timeout(config, None) == 30.0
    assert _user_turn_stop_timeout(config, object()) == 30.0


def _turn_end_events(user_turn_stop_timeout, transcript_delay):
    """Run one VAD-driven user turn whose transcript arrives late; return the controller's events."""

    async def run():
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.0)],
            ),
            user_turn_stop_timeout=user_turn_stop_timeout,
        )
        events = []

        def record(name):
            async def handler(*_args, **_kwargs):
                events.append(name)

            return handler

        for name in ("on_user_turn_started", "on_user_turn_stopped", "on_user_turn_stop_timeout"):
            controller.add_event_handler(name, record(name))
        for name in (
            "on_push_frame",
            "on_broadcast_frame",
            "on_user_turn_inference_triggered",
            "on_reset_aggregation",
        ):
            controller.add_event_handler(name, record("ignored"))

        await controller.setup(TaskManager())
        await controller.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2))
        await controller.process_frame(VADUserStoppedSpeakingFrame(stop_secs=1.2))
        await asyncio.sleep(transcript_delay)
        events.append("transcript")
        await controller.process_frame(TranscriptionFrame("late text", "user", "now", finalized=True))
        await asyncio.sleep(0.05)
        await controller.cleanup()
        return [e for e in events if e != "ignored"]

    return asyncio.run(run())


def test_disabled_watchdog_lets_a_late_transcript_end_the_turn():
    """Pins pipecat's handling of ``user_turn_stop_timeout=None``, which the builder passes by default.

    A finite watchdog shorter than the STT reply ends the turn before the text
    arrives, which is how a slow transcript landed in the next turn. None must
    leave the turn open until the transcript ends it, not crash or fire at once.
    """
    assert _turn_end_events(0.05, transcript_delay=0.6) == [
        "on_user_turn_started",
        "on_user_turn_stop_timeout",
        "on_user_turn_stopped",
        "transcript",
    ]
    assert _turn_end_events(None, transcript_delay=0.6) == [
        "on_user_turn_started",
        "transcript",
        "on_user_turn_stopped",
    ]


@pytest.mark.parametrize(("is_omni_model", "waits"), [(False, True), (True, False)])
def test_omni_stop_strategies_do_not_wait_for_a_transcript(is_omni_model, waits):
    """UserAudioBuffer consumes every transcript in omni mode, so waiting for one would never end the turn."""
    config = _config()
    config.server_config.llm["is_omni_model"] = is_omni_model

    (external_stop,) = _turn_strategies(config, object()).stop
    _set_turn_taking(config, type="speech_timeout")
    (vad_stop,) = _turn_strategies(config, None).stop
    assert external_stop.wait_for_transcript is waits
    assert vad_stop.wait_for_transcript is waits


class _CountLLMRuns(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.runs = 0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMRunFrame):
            self.runs += 1
        await self.push_frame(frame, direction)


def test_omni_speech_timeout_turn_runs_the_llm_without_the_watchdog():
    """Regression: on omni + speech_timeout the 5s watchdog used to be the only thing that ended a turn.

    With it disabled and a stop strategy still waiting for text, the agent never
    replied, because UserAudioBuffer swallows the transcript and only runs the LLM
    on the aggregator's UserStoppedSpeakingFrame.
    """
    config = _set_turn_taking(_config(), type="speech_timeout", user_speech_timeout=0.1)
    config.server_config.llm.update(is_omni_model=True, inject_dummy_user_message=False)
    context, user_agg, _, _ = builders.build_context_and_aggregators(object(), config, None)
    audio_buffer = UserAudioBuffer(context=context, user_context_aggregator=user_agg)
    counter = _CountLLMRuns()

    def utterance(text):
        audio = InputAudioRawFrame(audio=b"\x00" * 512, sample_rate=16000, num_channels=1)
        return [
            VADUserStartedSpeakingFrame(start_secs=0.2),
            audio,
            VADUserStoppedSpeakingFrame(stop_secs=0.4),
            SleepFrame(0.05),
            TranscriptionFrame(text, "user", "now", finalized=True),
            SleepFrame(0.5),
        ]

    asyncio.run(
        run_test(
            Pipeline([audio_buffer, user_agg, counter]),
            frames_to_send=utterance("one") + utterance("two"),
        )
    )

    assert counter.runs == 2
    assert [m["role"] for m in context.get_messages()].count("user") == 2

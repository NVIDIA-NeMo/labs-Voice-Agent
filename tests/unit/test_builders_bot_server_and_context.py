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

from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pipecat.frames.frames import AudioRawFrame, EndWorkerFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
)
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies

from nemo_voice_agent.evaluation.tools.waitlist_tools import DropWaitListTool, GetWaitlistTool, JoinWaitListTool
from nemo_voice_agent.pipecat import bot_server
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
        turn_taking={"enabled": True},
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
        TURN_TAKING_MAX_BUFFER_SIZE=3,
        TURN_TAKING_BOT_STOP_DELAY=0.25,
        TURN_TAKING_BACKCHANNEL_PHRASES_PATH=["yeah"],
        SYSTEM_ROLE="system",
        SYSTEM_PROMPT="policy",
        get_vad_params=lambda: "vad-params",
        get_diar_params=lambda: "diar-params",
    )
    for name, value in overrides.items():
        setattr(config, name, value)
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
    config.server_config.turn_taking["enabled"] = False
    assert builders.build_turn_taking(config) is None


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
    # turn_taking.enabled is True in _config(), so NeMoTurnTakingService owns the
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
    """Registered handlers initialize, terminate, reset, and finalize a session."""
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
    assert sum(isinstance(frame, EndWorkerFrame) for frame in task.queued) == 2
    assert reset_service.reset_calls == 1
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

    async def result_callback(self, result):
        self.result = result


def test_waitlist_tools_cover_join_drop_get_and_not_found():
    """Waitlist tools mutate shared state and report both drop outcomes."""
    state = {"waitlist": [{"name": "Ada", "phone": "1", "party_size": 2}]}
    summaries = []

    async def summary(value):
        summaries.append(json.loads(value))

    join = JoinWaitListTool(shared_state=state)
    join.send_scenario_summary = summary
    params = _Params({"name": "Grace", "phone": "2", "party_size": 3})
    asyncio.run(join._execute(params))
    assert params.result["position"] == 2
    assert join.required_properties == ["name", "phone", "party_size"]
    assert set(join.properties) == {"name", "phone", "party_size"}

    drop = DropWaitListTool(shared_state=state)
    drop.send_scenario_summary = summary
    params = _Params({"name": "Ada"})
    asyncio.run(drop._execute(params))
    assert params.result["success"] is True
    params = _Params({"name": "missing"})
    asyncio.run(drop._execute(params))
    assert params.result["success"] is False
    assert drop.required_properties == ["name"]

    getter = GetWaitlistTool(shared_state=state)
    params = _Params({})
    asyncio.run(getter._execute(params))
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
    config.server_config.turn_taking = {"enabled": False}
    without_service = _turn_strategies(config, None)
    assert not isinstance(without_service, ExternalUserTurnStrategies)
    assert all(s._enable_user_speaking_frames for s in without_service.start)
    assert [type(s).__name__ for s in without_service.start] == ["VADUserTurnStartStrategy"]
    # Named explicitly so we don't inherit pipecat's default stop strategy,
    # which would pull in LocalSmartTurnAnalyzerV3.
    assert [type(s).__name__ for s in without_service.stop] == ["SpeechTimeoutUserTurnStopStrategy"]


def test_turn_taking_instance_overrides_the_config_flag():
    """The passed-in service decides, not a re-derived read of turn_taking.enabled.

    A bot may construct its turn-taking service inline; re-deriving the answer
    from config would then silently pick the wrong strategy pair.
    """
    config = _config()
    config.server_config.turn_taking = {"enabled": False}

    assert isinstance(_turn_strategies(config, object()), ExternalUserTurnStrategies)
    assert not isinstance(_turn_strategies(config, None), ExternalUserTurnStrategies)

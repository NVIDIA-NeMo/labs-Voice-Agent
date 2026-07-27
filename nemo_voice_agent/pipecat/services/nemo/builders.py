# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Small composable builders for the pipecat services a voice-agent bot uses.

These are thin wrappers around the existing service constructors so bot scripts
can skip the repeated boilerplate of reading ``ConfigManager`` properties. Each
builder is independent: a bot imports only what it needs. Novel services that
aren't covered here can still be constructed inline.
"""

import copy
from datetime import datetime
from typing import Optional

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.websocket.server import (
    SingleClientWebsocketServerParams,
    SingleClientWebsocketServerTransport,
)
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import (
    ExternalUserTurnStrategies,
    UserTurnStrategies,
)

from nemo_voice_agent.pipecat.services.nemo.audio_logger import AudioLogger
from nemo_voice_agent.pipecat.services.nemo.diar import NemoDiarService
from nemo_voice_agent.pipecat.services.nemo.llm import get_llm_service_from_config
from nemo_voice_agent.pipecat.services.nemo.stt import get_stt_service_from_config
from nemo_voice_agent.pipecat.services.nemo.tts import build_text_aggregator, get_tts_service_from_config
from nemo_voice_agent.pipecat.services.nemo.turn_taking import NeMoTurnTakingService
from nemo_voice_agent.utils import ConfigManager


def build_audio_logger(config_manager: ConfigManager) -> Optional[AudioLogger]:
    """Build an AudioLogger if ``transport.record_audio_data`` is enabled."""
    server_config = config_manager.server_config
    if not server_config.transport.get("record_audio_data", False):
        return None
    log_dir = server_config.transport.get("audio_log_dir", "./audio_logs")
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    audio_logger = AudioLogger(log_dir=log_dir, session_id=session_id, enabled=True)
    logger.info(f"AudioLogger initialized for session: {session_id} at {log_dir}")
    return audio_logger


def build_vad_analyzer(config_manager: ConfigManager) -> SileroVADAnalyzer:
    """Build the Silero VAD analyzer at the transport's input sample rate."""
    server_config = config_manager.server_config
    sample_rate = server_config.transport.get("audio_in_sample_rate", config_manager.SAMPLE_RATE)
    return SileroVADAnalyzer(sample_rate=sample_rate, params=config_manager.get_vad_params())


def build_vad_processor(vad_analyzer: SileroVADAnalyzer | None) -> Optional[VADProcessor]:
    """Wrap ``vad_analyzer`` in the pipeline processor that emits VAD frames.

    Pipecat 1.0 removed ``vad_analyzer`` from ``TransportParams`` — VAD is no
    longer run by the input transport. Placing a ``VADProcessor`` immediately
    after ``transport.input()`` restores the old frame ordering, so
    ``NeMoTurnTakingService`` keeps receiving ``VADUserStartedSpeakingFrame`` /
    ``VADUserStoppedSpeakingFrame`` at exactly the point it used to.

    Returns ``None`` when there is no analyzer, so callers can drop it from the
    pipeline list with the same ``if x is not None`` pattern used elsewhere.
    """
    if vad_analyzer is None:
        return None
    return VADProcessor(vad_analyzer=vad_analyzer)


def build_ws_transport(
    config_manager: ConfigManager,
    vad_analyzer: SileroVADAnalyzer | None,  # noqa: ARG001 - kept for call-site compatibility
    host: str,
    port: int,
) -> SingleClientWebsocketServerTransport:
    """Build the no-timeout websocket server transport used by all bots.

    ``vad_analyzer`` is accepted but unused: since pipecat 1.0 the transport no
    longer runs VAD. Pass the analyzer to :func:`build_vad_processor` and insert
    the result right after ``transport.input()`` instead.
    """
    server_config = config_manager.server_config
    return SingleClientWebsocketServerTransport(
        params=SingleClientWebsocketServerParams(
            serializer=ProtobufFrameSerializer(),
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            session_timeout=None,
            audio_in_sample_rate=server_config.transport.get("audio_in_sample_rate", config_manager.SAMPLE_RATE),
            audio_out_sample_rate=server_config.transport.get("audio_out_sample_rate", None),
            audio_out_10ms_chunks=config_manager.TRANSPORT_AUDIO_OUT_10MS_CHUNKS,
        ),
        host=host,
        port=port,
    )


def build_stt(config_manager: ConfigManager, audio_logger: Optional[AudioLogger] = None) -> STTService:
    """Build the NeMo STT service from config."""
    return get_stt_service_from_config(config_manager.server_config.stt, audio_logger)


def build_diar(config_manager: ConfigManager, audio_logger: Optional[AudioLogger] = None) -> Optional[NemoDiarService]:
    """Build the diarization service, or return ``None`` if ``diar.enabled`` is False."""
    if not config_manager.server_config.diar.get("enabled", False):
        return None
    return NemoDiarService(
        model=config_manager.DIAR_MODEL,
        device=config_manager.STT_DEVICE,
        params=config_manager.get_diar_params(),
        sample_rate=config_manager.SAMPLE_RATE,
        backend="legacy",
        enabled=True,
    )


def build_turn_taking(
    config_manager: ConfigManager,
    audio_logger: Optional[AudioLogger] = None,
    *,
    use_diar: Optional[bool] = None,
    use_vad: bool = True,
) -> NeMoTurnTakingService:
    """Build the turn-taking service. ``use_diar`` defaults to ``config_manager.USE_DIAR``."""
    if use_diar is None:
        use_diar = config_manager.USE_DIAR
    if not config_manager.server_config.turn_taking.get("enabled", True):
        return None
    return NeMoTurnTakingService(
        use_vad=use_vad,
        use_diar=use_diar,
        max_buffer_size=config_manager.TURN_TAKING_MAX_BUFFER_SIZE,
        bot_stop_delay=config_manager.TURN_TAKING_BOT_STOP_DELAY,
        backchannel_phrases=config_manager.TURN_TAKING_BACKCHANNEL_PHRASES_PATH,
        audio_logger=audio_logger,
    )


def build_tts(config_manager: ConfigManager, audio_logger: Optional[AudioLogger] = None) -> TTSService:
    """Build the TTS service via ``get_tts_service_from_config``."""
    return get_tts_service_from_config(config_manager.server_config.tts, audio_logger)


def build_llm_text_processor(config_manager: ConfigManager) -> Optional[LLMTextProcessor]:
    """Build the processor that segments LLM text into TTS-sized chunks.

    Pipecat 1.0 removed ``TTSService(text_aggregator=...)``; aggregation now
    belongs to an ``LLMTextProcessor`` sitting immediately upstream of the TTS
    service. Insert the result there — pipecat *silently ignores* unknown
    constructor kwargs, so passing the aggregator to the service instead would
    quietly fall back to plain sentence splitting with no error to notice.

    Returns ``None`` when ``tts.use_text_aggregator`` is False.
    """
    aggregator = build_text_aggregator(config_manager.server_config.tts)
    if aggregator is None:
        return None
    return LLMTextProcessor(text_aggregator=aggregator)


def build_llm(config_manager: ConfigManager) -> LLMService:
    """Build the LLM service via ``get_llm_service_from_config``."""
    return get_llm_service_from_config(config_manager.server_config.llm)


def build_context_and_aggregators(llm: BaseOpenAILLMService, config_manager: ConfigManager):
    """Build the ``LLMContext`` and its user/assistant aggregators.

    Returns ``(context, user_aggregator, assistant_aggregator, original_messages)``.
    ``original_messages`` is a fresh deep-copy of the initial message list, safe
    to hand to the reset/update-prompt RTVI handler factories.

    Pipecat 1.0+ made the user aggregator the owner of turn detection, so the
    strategy pair has to mirror whichever component actually decides turn
    boundaries in this config. Both branches read the VAD frames emitted by the
    ``VADProcessor`` that :func:`build_vad_processor` puts right after
    ``transport.input()`` — neither needs its own analyzer.

    - ``turn_taking.enabled: true`` (all the local-NeMo configs):
      ``NeMoTurnTakingService`` pushes ``UserStartedSpeakingFrame`` /
      ``UserStoppedSpeakingFrame`` itself, so we select
      ``ExternalUserTurnStrategies`` — the supported way to tell the aggregator
      that an upstream processor owns turn detection. This replaces the old
      ``transport.can_create_user_frames=False`` knob, which pipecat removed
      along with transport-side VAD.
    - ``turn_taking.enabled: false`` (``*_nvidia.yaml``): nothing upstream emits
      user-turn frames — these configs used to rely on the transport's VAD doing
      it, which is what their now-obsolete ``can_create_user_frames: true`` was
      for. Drive the turn off VAD directly instead. Note we name the stop
      strategy explicitly rather than taking pipecat's default, which would pull
      in ``LocalSmartTurnAnalyzerV3``.
    """
    messages = [
        {
            "role": config_manager.SYSTEM_ROLE,
            "content": config_manager.SYSTEM_PROMPT,
        }
    ]
    if config_manager.server_config.llm.get("inject_dummy_user_message", False):
        dummy_message = config_manager.server_config.llm.get("dummy_user_message", "Hello.")
        messages.append({"role": "user", "content": dummy_message})

    context = LLMContext(messages=messages)
    original_messages = copy.deepcopy(context.get_messages())

    if config_manager.server_config.turn_taking.get("enabled", True):
        user_turn_strategies = ExternalUserTurnStrategies()
    else:
        user_turn_strategies = UserTurnStrategies(
            start=[VADUserTurnStartStrategy()],
            stop=[SpeechTimeoutUserTurnStopStrategy()],
        )

    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(user_turn_strategies=user_turn_strategies),
    )
    return (
        context,
        context_aggregator.user(),
        context_aggregator.assistant(),
        original_messages,
    )


def resolve_log_file_path(
    config_manager: ConfigManager, default_name: str = "bot_server.log"
) -> tuple[str, str, bool]:
    """Read the ``server.{log_file,log_level,create_new_log,overwrite_existing_log}`` block.

    Returns ``(log_file, log_level, create_new_log)``. Callers pair this with
    ``setup_rotating_log`` from ``nemo_voice_agent.utils.misc`` to handle
    the rename-existing-log dance.
    """
    server = config_manager.server_config.server
    return (
        server.get("log_file", default_name),
        server.get("log_level", "DEBUG"),
        server.get("create_new_log", False),
    )


def overwrite_existing_log(config_manager: ConfigManager) -> bool:
    """Whether to delete (True) or rename (False) a pre-existing log file on startup."""
    return bool(config_manager.server_config.server.get("overwrite_existing_log", False))


__all__ = [
    "build_audio_logger",
    "build_vad_analyzer",
    "build_vad_processor",
    "build_ws_transport",
    "build_stt",
    "build_diar",
    "build_turn_taking",
    "build_tts",
    "build_llm_text_processor",
    "build_llm",
    "build_context_and_aggregators",
    "resolve_log_file_path",
    "overwrite_existing_log",
]

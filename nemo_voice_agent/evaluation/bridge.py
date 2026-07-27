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

"""
Voice Agent Evaluation Bridge

Connects two voice agents via WebSocket and provides:
- Bidirectional audio routing
- Response latency measurement
- Dynamic system prompt updates via RTVI actions
- Conversation monitoring and metrics
"""

import asyncio
import copy
import json
import queue
import random
import threading
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pipecat.processors.frameworks.rtvi.models as RTVI
import soxr
import websockets
from loguru import logger
from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.serializers.protobuf import MessageFrame, ProtobufFrameSerializer

from nemo_voice_agent.evaluation.tools.rtvi_control import (
    EXIT_MESSAGE_END_TAG,
    EXIT_MESSAGE_START_TAG,
    FINAL_RESPONSE_END_TAG,
    FINAL_RESPONSE_START_TAG,
)
from nemo_voice_agent.utils import setup_logging

# Import AudioStream for buffering and resampling
from nemo_voice_agent.utils.audio import AudioStream, NoiseConfig


# RTVI message type constants - automatically adapts to pipecat changes
RTVI_BOT_STOPPED_SPEAKING = RTVI.BotStoppedSpeakingMessage().type
RTVI_BOT_STARTED_SPEAKING = RTVI.BotStartedSpeakingMessage().type
RTVI_BOT_TRANSCRIPTION = RTVI.BotTranscriptionMessage(data=RTVI.TextMessageData(text="")).type
RTVI_BOT_TTS_TEXT = RTVI.BotTTSTextMessage(data=RTVI.TextMessageData(text="")).type
RTVI_BOT_SERVER_MESSAGE = RTVI.ServerMessage(data=RTVI.TextMessageData(text="")).type

# Message types whose per-event side context is already conveyed by an
# existing dedicated log line in the monitors below (e.g. ``[AGENT TTS] ...``,
# ``[AGENT STARTED SPEAKING]``). The uniform ``[SIDE EVENT] type=<X>`` tag
# emitted by ``_log_rtvi_event`` is skipped for these types so the bridge log
# doesn't double up.
_RTVI_TYPES_ALREADY_TAGGED = frozenset({RTVI_BOT_STARTED_SPEAKING, RTVI_BOT_TTS_TEXT, RTVI_BOT_STOPPED_SPEAKING})

STOP_REASON_TIMEOUT = "[TIMEOUT]"
STOP_REASON_EXIT = "[EXIT]"


@dataclass
class ResponseLatency:
    """Single response latency measurement"""

    user_stop_time: float  # When user stopped speaking
    agent_start_time: float  # When agent started responding
    latency_ms: float  # Response latency in milliseconds
    user_transcript: str = ""
    agent_transcript: str = ""


@dataclass
class SegmentEntry:
    """Entry for segLST format (segment list with timing)"""

    start_time: float  # Start time in seconds
    end_time: float  # End time in seconds
    speaker: str  # "user" or "agent"
    transcript: str  # Text content


@dataclass
class EvaluationMetrics:
    """Metrics collected during evaluation"""

    turns: list = field(default_factory=list)
    latencies: List[ResponseLatency] = field(default_factory=list)
    start_time: datetime = None
    end_time: datetime = None

    # Buffered log entries (start_time, formatted_entry) - sorted and written at end
    log_entries: List[Tuple[float, str]] = field(default_factory=list)

    # Audio timing state
    user_last_audio_time: Optional[float] = None
    agent_last_audio_time: Optional[float] = None
    waiting_for_agent_response: bool = False
    last_user_transcript: str = ""

    # Transcript accumulation (segments arrive incrementally)
    user_current_transcript: str = ""
    agent_current_transcript: str = ""

    thread_start_timestamp: Optional[float] = None  # When routing threads start (for conversation log timing)

    # Segment tracking for segLST output
    segments: List[SegmentEntry] = field(default_factory=list)
    current_user_segment: Optional[SegmentEntry] = None
    current_agent_segment: Optional[SegmentEntry] = None

    agent_final_response: List[str] = field(default_factory=list)
    agent_final_response_time: List[float] = field(default_factory=list)

    def get_latency_stats(self):
        """Calculate latency statistics"""
        if not self.latencies:
            return {
                "count": 0,
                "mean_ms": 0,
                "p50_ms": 0,
                "p95_ms": 0,
                "min_ms": 0,
                "max_ms": 0,
            }

        latencies_sorted = sorted([lat.latency_ms for lat in self.latencies])
        count = len(latencies_sorted)

        return {
            "count": count,
            "mean_ms": sum(latencies_sorted) / count,
            "p50_ms": latencies_sorted[count // 2],
            "p95_ms": latencies_sorted[int(count * 0.95)] if count > 0 else 0,
            "min_ms": latencies_sorted[0],
            "max_ms": latencies_sorted[-1],
        }

    def reset(self):
        """Reset all metrics to prepare for a new scenario"""
        self.start_time = None
        self.end_time = None

        # Reset latency tracking state
        self.user_last_audio_time = None
        self.agent_last_audio_time = None
        self.waiting_for_agent_response = False
        self.last_user_transcript = ""

        # Clear accumulated transcript segments
        self.user_current_transcript = ""
        self.agent_current_transcript = ""

        # Reset scenario-specific metrics (for multi-scenario evaluations)
        self.latencies = []
        self.turns = []
        self.segments = []
        self.log_entries = []

        self.thread_start_timestamp = None
        self.current_user_segment = None
        self.current_agent_segment = None

        self.agent_final_response = []
        self.agent_final_response_time = []


class VoiceAgentEvaluationBridge:
    """
    Evaluation bridge that connects two voice agents via WebSocket
    and provides control through RTVI actions.

    Key features:
    - Routes audio bidirectionally between agents
    - Monitors transcriptions and metrics
    - Measures response latency by tracking audio frames
    - Can send RTVI control messages to update prompts
    - Works with distributed agents
    """

    def __init__(
        self,
        user_url: str,
        agent_url: str,
        output_dir: Optional[str] = None,
        scenario_name: Optional[str] = None,
        user_output_sample_rate: int = 24000,
        agent_output_sample_rate: int = 24000,
        user_input_sample_rate: int = 16000,
        agent_input_sample_rate: int = 16000,
        output_sample_rate: int = 16000,
        audio_chunk_in_seconds: float = 0.016,
        use_burst_mode: bool = False,
        burst_size_range: Tuple[int, int] = (3, 8),
        burst_delay_ms: int = 0,
        grace_period: float = 1.0,
        turn_start_offset_secs: float = -0.0,
        turn_end_offset_secs: float = -0.3,
        noise_config: Optional[NoiseConfig] = None,
        log_level: str = "DEBUG",
    ):
        """
        Args:
            user_url: URL of the user WebSocket
            agent_url: URL of the agent WebSocket
            output_dir: Directory for all output files (conversation log, audio, segLST)
            scenario_name: Name of the scenario
            user_output_sample_rate: Sample rate of the user output
            agent_output_sample_rate: Sample rate of the agent output
            user_input_sample_rate: Sample rate of the user input
            agent_input_sample_rate: Sample rate of the agent input
            output_sample_rate: Sample rate of the output
            audio_chunk_in_seconds: Duration of the audio chunk in seconds
            use_burst_mode: Whether to use burst mode, default to steady mode with fixed interval
            burst_size_range: Range of the random burst size, used to simulate the irregular sending pattern
                of a browser.
            burst_delay_ms: Delay between the frames in the random burst, used to simulate the irregular
                sending pattern of a browser.
            grace_period: Grace period after the main duration, used to drain the websocket
            turn_start_offset_secs: Offset added to turn start times in conversation log and segLST,
                so that the latency by BOT_STARTED_SPEAKING event is mitigated. This is a workaround to the fact
                that the BOT_STARTED_SPEAKING event may come after the first audio chunk is sent.
            turn_end_offset_secs: Offset added to turn end times in conversation log and segLST,
                so that the latency by BOT_STOPPED_SPEAKING event is mitigated. This is a workaround to the fact
                that the BOT_STOPPED_SPEAKING event is sent after 0.35s silence in Pipecat output transport.
            noise_config: Noise configuration, used to configure the noise for the audio stream
        """
        self.user_url = user_url
        self.agent_url = agent_url
        self.output_dir = output_dir
        self.scenario_name = scenario_name
        self.log_file = None
        self.seglst_file = None
        self.bridge_audio_file = None
        self.user_output_sample_rate = user_output_sample_rate
        self.agent_output_sample_rate = agent_output_sample_rate
        self.user_input_sample_rate = user_input_sample_rate
        self.agent_input_sample_rate = agent_input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.audio_chunk_in_seconds = audio_chunk_in_seconds
        self.log_level = log_level

        # Random burst mode configuration (simulates browser's irregular sending pattern)
        self.use_burst_mode = use_burst_mode  # Disable burst mode by default
        self.burst_size_range = burst_size_range  # Random frames per burst
        self.burst_delay_ms = burst_delay_ms  # sleep duration between frames in burst
        # Pause calculated per burst: (burst_size × 16ms) - burst_duration
        # This maintains 16ms average per frame while varying the pattern

        # Grace period and timeout configuration for send loops
        self.grace_period = grace_period  # Extra time to drain audio after main duration
        # Settle delay after receiving ``<exit>`` from the agent and before
        # flipping ``stop_event``. Lets the bots commit any in-flight
        # function-call cycle to their LLM contexts and
        # ``shared_state["actions"]`` before the bridge pulls
        # end-of-scenario state. 500 ms is comfortably above the observed
        # commit time (~3 s in the worst case, but ~50 ms in the common
        # case once the source-side fix in ``SendExitMessageTool._execute``
        # is applied — this remains as belt-and-braces).
        self.exit_settle_delay = 0.5

        self.turn_start_offset_secs = turn_start_offset_secs
        self.turn_end_offset_secs = turn_end_offset_secs

        # Noise configuration for user channel
        self.noise_config = noise_config

        self.user_ws = None
        self.user_scenario_summary = None
        self.agent_ws = None

        # Cross-side sync state — populated by ``_setup_cross_side_sync``
        # at scenario start when the scenario opts in via ``sync_state``
        # override. Default-off for single-side domains.
        self.sync_enabled: bool = False
        self.scenario_instance = None
        self.shadow_state: Optional[dict] = None
        self.shadow_tool_map: Dict[str, Any] = {}
        self.sync_lock = asyncio.Lock()

        self.metrics = EvaluationMetrics()

        # Serializers for protobuf communication
        self.serializer = ProtobufFrameSerializer()

        # Track RTVI state
        self.user_ready = False
        self.agent_ready = False

        # Debug: accumulate sent audio chunks for analysis (only final sent audio)
        self.sent_to_agent_chunks = []  # USER→AGENT final sent chunks
        self.sent_to_user_chunks = []  # AGENT→USER final sent chunks

        # Thread-safe queues for audio routing between threads
        # Each queue passes raw audio bytes between WebSocket threads
        self.user_to_agent_queue = queue.Queue()  # User audio → Agent
        self.agent_to_user_queue = queue.Queue()  # Agent audio → User

        # Thread control
        self.stop_event = threading.Event()
        self.threads = []
        self.stop_reason = STOP_REASON_TIMEOUT

        # Bridge resamples at source (like browser client) for better quality
        # This avoids STT having to resample small chunks
        logger.info("Bridge configured to resample audio at source (simulating browser behavior)")
        logger.info(f"  User: {self.user_output_sample_rate}Hz (TTS) → {self.agent_input_sample_rate}Hz (STT)")
        logger.info(f"  Agent: {self.agent_output_sample_rate}Hz (TTS) → {self.user_input_sample_rate}Hz (STT)")

        # Log burst mode configuration
        if self.use_burst_mode:
            logger.info(
                f"Random burst mode enabled: {self.burst_size_range[0]}-{self.burst_size_range[1]} "
                f"frames per burst, {self.burst_delay_ms}ms between frames"
            )
            min_pause = (self.burst_size_range[0] * self.audio_chunk_in_seconds * 1000) - (
                (self.burst_size_range[0] - 1) * self.burst_delay_ms
            )
            max_pause = (self.burst_size_range[1] * self.audio_chunk_in_seconds * 1000) - (
                (self.burst_size_range[1] - 1) * self.burst_delay_ms
            )
            logger.info(f"  Pause range: {min_pause:.0f}-{max_pause:.0f}ms (calculated to maintain 16ms avg)")
        else:
            logger.info(f"Steady mode: sending at constant {self.audio_chunk_in_seconds * 1000:.0f}ms intervals")

        # Initialize output directory and log files
        if output_dir:
            self.init_output_dir(output_dir, scenario_name, log_level)

        self.bridge_ready = False
        self.needs_reset = False
        self.final_response_file = "final_agent_response.json"
        # Hash-only post-scenario record. The actual DB stays on the bot server;
        # only the SHA-256 of the canonicalized DB travels through the WebSocket.
        # Written to disk for traceability; the runner reads it via the
        # ``scenario_summary["db_hash"]`` field in memory.
        self.final_scenario_db_hash_file = "final_scenario_db_hash.txt"
        self.user_context_history = None
        self.agent_context_history = None
        # Pulled at end-of-scenario via the get_scenario_summary RTVI action.
        # Each bot returns its own {"actions", "db_hash", "db"?}; the bridge
        # merges the per-bot pulls into this dict, labeling by source:
        #   {"actions": [...],
        #    "db_hash": "<agent's hash>"|None,
        #    "user_db_hash": "<user's hash>"|None,  # populated when user-side pull is wired
        #    "db": {...}|None,                       # only when include_db=True
        #    "user_db": {...}|None}                  # only when include_db=True
        # ``None`` overall if the pull didn't happen.
        self.scenario_summary: Optional[dict] = None
        # Per-scenario token usage, accumulated from RTVI ``metrics`` events as
        # they arrive. Reset by ``prepare_for_scenario``. Shape:
        #   {"agent": {"n_calls": int, "prompt": int, "completion": int},
        #    "user":  {"n_calls": int, "prompt": int, "completion": int}}
        # The runner snapshots this into ``metrics.json["token_usage"]`` so the
        # canonical numbers live next to the run output, not just in the
        # bridge log. The eval-result-analyzer skill prefers this source when
        # present; falls back to bridge_log.txt parsing for older runs.
        self.token_usage: dict = self._fresh_token_usage()

    @staticmethod
    def _fresh_token_usage() -> dict:
        return {
            "agent": {"n_calls": 0, "prompt": 0, "completion": 0},
            "user": {"n_calls": 0, "prompt": 0, "completion": 0},
        }

    def init_output_dir(
        self,
        output_dir: str,
        scenario_name: Optional[str] = None,
        log_level: str = "DEBUG",
    ):
        """Initialize the output directory and all derived log/audio file paths."""
        logger.info(f"Initializing output directory: {output_dir}, session name: {scenario_name}")
        self.output_dir = output_dir
        self.scenario_name = scenario_name
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.log_file = str(Path(output_dir) / "conversation_log.txt")
        self.seglst_file = str(Path(output_dir) / "conversation_log.seglst.json")
        self.bridge_audio_file = str(Path(output_dir) / "conversation_log.wav")

        # Initialize logging for this scenario
        bridge_log_file = str(Path(output_dir) / "bridge_log.txt")
        setup_logging(log_file=bridge_log_file, log_level=log_level)  # Update logging to write to this file

        try:
            with open(self.log_file, "w") as f:
                f.write("RTVI Evaluation Bridge - Conversation Log\n")
                f.write("=" * 80 + "\n")
                f.write(f"Start Time: {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")
        except Exception as e:
            logger.error(f"Error initializing log file: {e}")
            return False
        return True

    def set_noise_config(self, noise_config: Optional[Union[NoiseConfig, dict]] = None):
        """Set the noise configuration"""
        logger.info(f"Setting noise configuration: {noise_config}")
        if noise_config is not None:
            if isinstance(noise_config, dict):
                noise_config = NoiseConfig(**noise_config)
            self.noise_config = noise_config
        else:
            self.noise_config = None

    async def prepare_for_scenario(
        self,
        scenario,
        output_dir: str,
        log_level: str = "DEBUG",
    ):
        """Prepare the bridge for a scenario.

        Args:
            scenario: A ``Scenario`` instance. The bridge calls methods
                on it directly (``get_user_prompt``, ``get_agent_prompt``,
                ``setup_shared_state``, ``initialization_actions``,
                ``sync_state``, etc.) — no intermediate dict
                serialization step. Single source of truth.
            output_dir: Per-scenario output directory.
            log_level: Pipecat log level for the bot servers.
        """
        # Initialize output directory for this scenario
        self.init_output_dir(output_dir, scenario_name=scenario.name, log_level=log_level)

        # Reset bridge before each scenario, and create connection to update the prompts
        await self.connect()

        # ``tool_domain`` tells each bot which registry namespace to
        # look up tools in. Falls back to "default" per-tool for shared
        # harness tools (EndConversationTool, etc.).
        tool_domain = getattr(scenario, "domain", "default")
        # Bridge-side bit: when True, ``_retrieve_scenario_summary``
        # asks the bot to inline the ``db`` / ``user_db`` dicts (not
        # just hashes) so the runner can evaluate
        # ``db_state_assertions`` predicates against them. Set
        # automatically from the scenario's predicate-list opt-in.
        self.include_db_in_summary = bool(getattr(scenario, "db_state_assertions", None))

        # Per-side scenario fixture data — pulled now, sent later as
        # part of ``apply_initialization`` (NOT via ``update_system_prompt``).
        # ``setup_shared_state`` populates ``state["db_path"]`` (or any
        # other scenario-specific keys) which the bot's
        # ``apply_initialization`` handler resolves into ``state["db"]``.
        user_state: dict = {}
        agent_state: dict = {}
        scenario.setup_shared_state(user_state, "user")
        scenario.setup_shared_state(agent_state, "agent")

        # 1. Prompt + tool surface only. ``update_system_prompt`` is the
        #    scenario-start lifecycle gate: it clears any prior
        #    ``shared_state`` (preserving dict identity for tool
        #    references) and stashes bot-side runtime sentinels
        #    (``__rtvi__``, ``__tool_domain__``). Scenario fixture data
        #    is NOT loaded here — that's the next step.
        await self.update_user_prompt(
            prompt=scenario.get_user_prompt(),
            tools=scenario.get_user_tools(),
            tool_domain=tool_domain,
        )
        await self.update_agent_prompt(
            prompt=scenario.get_agent_prompt(),
            tools=scenario.get_agent_tools(),
            tool_domain=tool_domain,
        )

        # 2. Full state initialization — always called, even when
        #    ``scenario.initialization_actions`` is falsy. The bot-side
        #    handler does (a) merge ``shared_state_init`` JSON into the
        #    bot's ``shared_state``, (b) resolve any ``db_path`` to an
        #    inlined ``db`` dict, then (c) apply the per-side filtered
        #    init function list. Steps (a) + (b) run for every scenario
        #    regardless of whether (c) has anything to do.
        await self._apply_initialization(
            scenario,
            user_shared_state_init=user_state,
            agent_shared_state_init=agent_state,
        )

        # Cross-side sync setup. When the scenario overrides
        # ``Scenario.sync_state`` (dual-side domains like tau2_telecom),
        # the bridge maintains in-process shadow DBs that mirror the
        # bots' state. After each write action fires on either bot, the
        # bridge replays it onto the shadow DBs, runs
        # ``scenario.sync_state(agent_db, user_db)``, and dispatches the
        # resulting per-side deltas to the corresponding bot via the
        # ``apply_sync_delta`` RTVI action. For single-side domains
        # ``sync_state`` is the no-op default — the entire pipeline is
        # skipped at scenario start.
        await self._setup_cross_side_sync(scenario)

        noise_config = getattr(scenario, "noise_config", None)
        self.set_noise_config(noise_config)

        # Disconnect the bridge to clear the WebSocket buffers
        await self.disconnect(print_stats=False)

        logger.info(f"Finished preparing for scenario: {scenario.name}")
        self.bridge_ready = True

    def _get_relative_time(self, timestamp: float) -> float:
        """
        Get time relative to scenario start (thread start time).

        Args:
            timestamp: Absolute timestamp (asyncio loop time)

        Returns:
            Time in seconds relative to thread_start_timestamp, or 0 if not set
        """
        if self.metrics.thread_start_timestamp is None:
            return 0.0
        return timestamp - self.metrics.thread_start_timestamp

    def _finalize_speaker_turn(self, speaker: str, timestamp: float) -> Optional[SegmentEntry]:
        """
        Finalize the current in-progress turn for the given speaker.

        Sets end_time, assigns transcript (or "[INTERRUPTED]" if no TTS text was received),
        appends the segment to self.metrics.segments, and clears accumulation state.

        Args:
            speaker: "user" or "agent"
            timestamp: Absolute timestamp (asyncio loop time)

        Returns:
            The finalized SegmentEntry, or None if no segment was in progress.
        """
        if speaker == "user":
            segment = self.metrics.current_user_segment
            transcript_acc = self.metrics.user_current_transcript
        else:
            segment = self.metrics.current_agent_segment
            transcript_acc = self.metrics.agent_current_transcript

        if segment is None:
            return None

        transcript = transcript_acc.strip() or "[INTERRUPTED]"
        segment.end_time = self._get_relative_time(timestamp)
        segment.transcript = transcript
        self.metrics.segments.append(segment)

        # Clear state
        if speaker == "user":
            self.metrics.current_user_segment = None
            self.metrics.user_current_transcript = ""
        else:
            self.metrics.current_agent_segment = None
            self.metrics.agent_current_transcript = ""

        logger.info(f"[{speaker.upper()}] {transcript}")
        return segment

    def _format_turn_log(
        self,
        role: str,
        text: str,
        start_time: float,
        end_time: float,
        latency_ms: float = None,
    ) -> str:
        """
        Format a turn entry for the conversation log.

        Args:
            role: "user" or "agent"
            text: Transcript text
            start_time: Turn start time (relative to scenario start)
            end_time: Turn end time (relative to scenario start)
            latency_ms: Optional response latency in milliseconds

        Returns:
            Formatted log entry string
        """
        duration = end_time - start_time
        log_entry = f"[{start_time:7.3f}s - {end_time:7.3f}s] ({duration:.3f}s) {role.upper()}: {text}\n"
        if latency_ms is not None:
            log_entry += f"  → Response latency: {latency_ms:.1f}ms\n"
        return log_entry

    async def connect(self, max_retries: int = 5, retry_delay: float = 1.0):
        """Connect to both user and agent with retry logic

        Args:
            max_retries: Maximum number of connection attempts per endpoint
            retry_delay: Initial delay between retries (doubles each retry)
        """
        # Connect to user with retries
        logger.info(f"Connecting to user at {self.user_url}")
        for attempt in range(max_retries):
            try:
                self.user_ws = await websockets.connect(
                    self.user_url, ping_interval=20, ping_timeout=10, close_timeout=10
                )
                logger.info(f"User connection established (attempt {attempt + 1})")
                break
            except (OSError, websockets.exceptions.WebSocketException) as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    logger.warning(f"User connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"User connection failed after {max_retries} attempts")
                    raise

        # Connect to agent with retries
        logger.info(f"Connecting to agent at {self.agent_url}")
        for attempt in range(max_retries):
            try:
                self.agent_ws = await websockets.connect(
                    self.agent_url, ping_interval=20, ping_timeout=10, close_timeout=10
                )
                logger.info(f"Agent connection established (attempt {attempt + 1})")
                break
            except (OSError, websockets.exceptions.WebSocketException) as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    logger.warning(f"Agent connection failed (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Agent connection failed after {max_retries} attempts")
                    raise

        # Send RTVI client-ready handshake to both agents
        await self._send_client_ready(self.user_ws)
        await self._send_client_ready(self.agent_ws)
        await self.reset()
        logger.info("Both agents connected and ready")

    async def _send_client_ready(self, ws):
        """Send RTVI client-ready handshake and wait for bot-ready"""
        client_ready_msg = {
            "label": "rtvi-ai",
            "type": "client-ready",
            "id": f"client_ready_{datetime.now().timestamp()}",
            "data": {
                "version": "1.1.0",
                "about": {"library": "evaluation-bridge", "library_version": "1.0.0"},
            },
        }

        # Serialize as MessageFrame and send
        msg_frame = MessageFrame(data=json.dumps(client_ready_msg))
        serialized = await self.serializer.serialize(msg_frame)
        await ws.send(serialized)

        logger.info("Client-ready handshake sent, waiting for bot-ready...")

        # Wait for bot-ready response
        try:
            timeout = 5.0
            start_time = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start_time < timeout:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    if isinstance(msg, bytes):
                        frame = await self.serializer.deserialize(msg)
                        if hasattr(frame, "message") and frame.message:
                            if isinstance(frame.message, str):
                                data = json.loads(frame.message)
                            else:
                                data = frame.message

                            if data.get("type") == "bot-ready":
                                logger.info("Received bot-ready response")
                                return True
                except asyncio.TimeoutError:
                    continue

            logger.warning("Timeout waiting for bot-ready response")
            return False
        except Exception as e:
            logger.error(f"Error waiting for bot-ready: {e}")
            return False

    async def update_user_prompt(
        self,
        prompt: str,
        tools: str,
        auto_reset: bool = False,
        add_suffix: bool = False,
        tool_domain: str = "default",
    ):
        """
        Update user's system prompt via RTVI action.

        Args:
            prompt: New system prompt text
            tools: New tools in json string format
            auto_reset: If True, also sends reset action after updating prompt
            add_suffix: If True, add previously configured system prompt suffix to the new prompt
            tool_domain: Registry namespace the bot server should use to look up
                tools by name. Stashed on bot-side ``shared_state["__tool_domain__"]``
                for write tools to read when emitting ``action-applied`` events.

        Scenario fixture data (``db_path``, custom keys from
        ``Scenario.setup_shared_state``) is NOT sent here — it flows via
        the subsequent ``apply_initialization`` call instead.
        """
        logger.info(f"Updating user prompt: {prompt[:100]}..., tools: {tools[:100]}..., tool_domain={tool_domain!r}")

        # Create RTVI action message
        action_msg = {
            "label": "rtvi-ai",
            "type": "client-message",
            "id": f"update_prompt_{datetime.now().timestamp()}",
            "data": {
                "t": "update_system_prompt",
                "d": {
                    "prompt": prompt,
                    "tools": tools,
                    "add_suffix": add_suffix,
                    "tool_domain": tool_domain,
                },
            },
        }

        # Serialize as MessageFrame and send
        msg_frame = MessageFrame(data=json.dumps(action_msg))
        serialized = await self.serializer.serialize(msg_frame)
        await self.user_ws.send(serialized)

        logger.info("User prompt update sent")

        if auto_reset:
            logger.info("Sending additional reset action to user...")
            await self._send_reset_action(self.user_ws, "user")

        return True

    async def update_agent_prompt(
        self,
        prompt: str,
        tools: str,
        auto_reset: bool = False,
        add_suffix: bool = False,
        tool_domain: str = "default",
    ):
        """
        Update agent's system prompt via RTVI action.

        Args:
            prompt: New system prompt text
            tools: New tools in json string format
            auto_reset: If True, also sends reset action after updating prompt
            add_suffix: If True, add previously configured system prompt suffix to the new prompt
            tool_domain: Registry namespace the bot server should use to look up
                tools by name (e.g., ``"tau2_airline"``). Falls back to
                ``"default"`` per-tool if the name isn't in the specified
                domain (with a warning logged bot-side).

        Scenario fixture data (``db_path``, custom keys from
        ``Scenario.setup_shared_state``) is NOT sent here — it flows via
        the subsequent ``apply_initialization`` call instead.
        """
        logger.info(f"Updating agent prompt: {prompt[:100]}..., tools: {tools[:100]}..., tool_domain={tool_domain!r}")

        # Create RTVI action message
        action_msg = {
            "label": "rtvi-ai",
            "type": "client-message",
            "id": f"update_prompt_{datetime.now().timestamp()}",
            "data": {
                "t": "update_system_prompt",
                "d": {
                    "prompt": prompt,
                    "tools": tools,
                    "add_suffix": add_suffix,
                    "tool_domain": tool_domain,
                },
            },
        }

        # Serialize as MessageFrame and send
        msg_frame = MessageFrame(data=json.dumps(action_msg))
        serialized = await self.serializer.serialize(msg_frame)
        await self.agent_ws.send(serialized)

        logger.info("Agent prompt update sent")

        if auto_reset:
            logger.info("Sending additional reset action to agent...")
            await self._send_reset_action(self.agent_ws, "agent")

        return True

    async def _send_reset_action(self, ws, agent_name: str):
        """
        Send RTVI reset action to clear conversation history.

        Args:
            ws: WebSocket connection
            agent_name: Name of agent (for logging)
        """
        if not ws:
            logger.info(f"[{agent_name.capitalize()}] Websocket is not connected, skipping reset")
            return

        reset_msg = {
            "label": "rtvi-ai",
            "type": "client-message",
            "id": f"reset_{datetime.now().timestamp()}",
            "data": {
                "t": "reset",
                "d": {},
            },
        }

        # Serialize as MessageFrame and send
        msg_frame = MessageFrame(data=json.dumps(reset_msg))
        serialized = await self.serializer.serialize(msg_frame)
        await ws.send(serialized)

        logger.info(f"{agent_name.capitalize()} reset action sent")

    async def reset(self):
        """
        Reset metrics and both agents' conversation history
        """
        logger.info("Resetting metrics and conversation context...")
        await self.reset_user()
        await self.reset_agent()
        # Reset all metrics
        self.metrics.reset()
        self.needs_reset = False

    async def reset_agent(self):
        """
        Reset agent's conversation history.
        Useful to clear context between evaluation scenarios.
        """
        if self.agent_ws:
            logger.info("Resetting agent...")
            await self._send_reset_action(self.agent_ws, "agent")
            logger.info("Agent reset complete")

    async def reset_user(self):
        """
        Reset user's conversation history.
        Useful to clear context between evaluation scenarios.
        """
        if self.user_ws:
            logger.info("Resetting user...")
            await self._send_reset_action(self.user_ws, "user")
            logger.info("User reset complete")

    async def send_text_to_user(self, text: str):
        """
        Send a text message to the user agent to trigger conversation.

        Args:
            text: Text to send to user agent's LLM
        """
        send_text_msg = {
            "label": "rtvi-ai",
            "type": "send-text",
            "id": f"send_text_{datetime.now().timestamp()}",
            "data": {
                "content": text,
                "options": {"run_immediately": True, "audio_response": True},
            },
        }

        msg_frame = MessageFrame(data=json.dumps(send_text_msg))
        serialized = await self.serializer.serialize(msg_frame)
        await self.user_ws.send(serialized)

        logger.info(f"Sent text to user: {text[:50]}...")

    async def send_text_to_agent(self, text: str):
        """
        Send a text message to the agent agent to trigger conversation.

        Args:
            text: Text to send to agent agent's LLM
        """
        send_text_msg = {
            "label": "rtvi-ai",
            "type": "send-text",
            "id": f"send_text_{datetime.now().timestamp()}",
            "data": {
                "content": text,
                "options": {"run_immediately": True, "audio_response": True},
            },
        }

        msg_frame = MessageFrame(data=json.dumps(send_text_msg))
        serialized = await self.serializer.serialize(msg_frame)
        await self.agent_ws.send(serialized)

        logger.info(f"Sent text to agent: {text[:50]}...")

    async def _receive_to_queue(
        self,
        ws: websockets.WebSocketClientProtocol,
        duration: float,
        direction: str,
        queue: queue.Queue,
        monitor_func: Callable,
    ):
        """
        Receive audio from websocket and put into queue.

        Args:
            ws: Source websocket to receive from
            duration: How long to run the receive loop in seconds
            direction: For logging (e.g., "USER→AGENT", "AGENT→USER")
            queue: Thread-safe queue to put audio chunks into
            monitor_func: Async monitoring function for metrics (e.g., _monitor_user_message)
        """
        logger.info(f"[{direction}] Starting receive loop")
        loop = asyncio.get_event_loop()
        start_time = loop.time()
        try:
            while not self.stop_event.is_set():
                # Use short timeout so we can check stop_event periodically
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                # Deserialize frame
                try:
                    frame = await self.serializer.deserialize(message)
                    if frame is None:
                        continue
                except Exception as e:
                    logger.error(f"[{direction}] Deserialization error: {e}")
                    continue

                current_time = loop.time()
                elapsed = current_time - start_time

                # Check if we're past the main duration
                in_grace_period = elapsed > duration
                if in_grace_period:
                    # logger.debug(f"[{direction}] In grace period, skip monitoring message: {frame}")
                    continue

                # Monitor messages
                await monitor_func(frame)

                # Check if this is audio
                if hasattr(frame, "audio") and frame.audio:
                    # Put raw audio into thread-safe queue
                    queue.put(frame.audio)
                    # logger.debug(f"[{direction}] Queued {len(frame.audio)} bytes of audio")

            if self.stop_event.is_set():
                logger.info(f"[{direction}] Stop event received, exiting receive loop")

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"[{direction}] Receive WebSocket closed: {e}")
        except Exception as e:
            logger.error(f"[{direction}] Receive error: {e}", exc_info=True)

    async def _send_audio_stream(
        self,
        audio_stream: AudioStream,
        dest_ws: websockets.WebSocketClientProtocol,
        direction: str,
        duration: int,
        source_queue: queue.Queue,
        sent_chunks_list: List[bytes],
    ):
        """
        Send audio stream at fixed intervals from AudioStream with duration and grace period.

        Args:
            audio_stream: AudioStream containing buffered and resampled audio
            dest_ws: Destination websocket to send to
            direction: For logging (e.g., "USER→AGENT", "AGENT→USER")
            duration: How long to run the send loop in seconds
            source_queue: Queue to retrieve audio chunks from
            sent_chunks_list: List to append sent chunks to for tracking
        """
        logger.info(f"[{direction}] Starting send loop")

        loop = asyncio.get_event_loop()
        start_time = loop.time()
        target_time = start_time  # Track target time incrementally for numerical stability

        try:
            while not self.stop_event.is_set():
                current_time = loop.time()
                elapsed = current_time - start_time
                in_grace_period = elapsed > duration
                if elapsed > (duration + self.grace_period):
                    logger.info(f"[{direction}] Grace period expired after {elapsed:.1f}s, stopping")
                    break

                # Empty all available audio from thread-safe queue (non-blocking)
                # This prevents stale audio buildup during burst pauses or LLM/TTS blocking
                chunks_retrieved = 0
                while True:
                    try:
                        audio_chunk = source_queue.get_nowait()
                        # Put into AudioStream for buffering/resampling
                        await audio_stream.put(audio_chunk)
                        chunks_retrieved += 1
                    except queue.Empty:
                        break

                # if chunks_retrieved > 0:
                #     logger.debug(f"[{direction}] Retrieved {chunks_retrieved} chunks from queue")
                if in_grace_period:
                    # logger.debug(f"[{direction}] In grace period, skip forwarding audio: {chunks_retrieved} chunks")
                    await asyncio.sleep(0.1)
                    continue

                # Burst sending: send N frames rapidly, then pause
                # Steady mode is just burst_size=1 (send 1 frame, pause 16ms, repeat)
                burst_size = (
                    random.randint(self.burst_size_range[0], self.burst_size_range[1]) if self.use_burst_mode else 1
                )

                # Send burst frames
                for idx in range(burst_size):
                    if idx > 0 and self.burst_delay_ms > 0:
                        # Small delay between frames in burst
                        await asyncio.sleep(self.burst_delay_ms / 1000.0)

                    # Get audio from AudioStream
                    audio_to_send, has_speech = await audio_stream.get_nowait()

                    # Track sent audio
                    sent_chunks_list.append(audio_to_send)

                    # Create frame and send
                    output_frame = OutputAudioRawFrame(
                        audio=audio_to_send,
                        sample_rate=audio_stream.output_sample_rate,
                        num_channels=1,
                    )
                    serialized = await self.serializer.serialize(output_frame)
                    await dest_ws.send(serialized)

                    # if has_speech:
                    #     logger.debug(
                    #         f"[{direction}] Sent {len(audio_to_send)} bytes "
                    #         f"({idx+1}/{burst_size}, has_speech: {has_speech})"
                    #     )

                # Time-based scheduling: increment target time from previous burst
                # This automatically compensates for processing overhead and is numerically stable
                target_time += burst_size * self.audio_chunk_in_seconds
                current_time = loop.time()
                wait_duration = max(0.001, target_time - current_time)

                if wait_duration < 0.001:
                    logger.debug(f"[{direction}] Behind schedule by {-wait_duration:.3f}s")

                # if self.use_burst_mode:
                #     logger.debug(
                #         f"[{direction}] Burst complete ({burst_size} frames), "
                #         f"waiting {wait_duration*1000:.1f}ms (target: {target_time:.3f}s)"
                #     )
                await asyncio.sleep(wait_duration)

            if self.stop_event.is_set():
                logger.info(f"[{direction}] Stop event received, exiting send loop")
            else:
                logger.info(f"[{direction}] Send loop finished")

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"[{direction}] WebSocket closed: {e}")
        except Exception as e:
            # print traceback
            import traceback

            traceback.print_exc()
            logger.error(f"[{direction}] Send error: {e}", exc_info=True)

    def user_websocket_thread(self, duration: int):
        """
        Thread 1: Handle all user WebSocket traffic (bidirectional).

        This thread:
        - Receives audio from user WebSocket
        - Puts user audio into user_to_agent_queue for agent thread
        - Gets agent audio from agent_to_user_queue
        - Sends agent audio to user WebSocket

        Args:
            duration: How long to run (seconds)
        """
        logger.info("[USER THREAD] Starting user WebSocket thread")

        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def user_loop():
            try:
                # Connect to user WebSocket
                async with websockets.connect(self.user_url, ping_timeout=None) as user_ws:
                    self.user_ws = user_ws
                    logger.info(f"[USER THREAD] Connected to user: {self.user_url}")

                    # Wait for ready handshake
                    await self._send_client_ready(user_ws)

                    # Create AudioStream for agent→user (buffering and resampling)
                    agent_to_user_stream = AudioStream(
                        chunk_size_in_seconds=self.audio_chunk_in_seconds,
                        input_sample_rate=self.agent_output_sample_rate,
                        output_sample_rate=self.user_input_sample_rate,
                        stream_resampler=False,
                        tag="AGENT→USER",
                    )

                    # Run bidirectional tasks (send loop manages timeout + grace period)
                    # Add overall timeout with grace period to stop receive loop when send loop finishes
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(
                                # Receive from user, put raw audio into queue
                                self._receive_user_to_queue(user_ws, duration),
                                # Get raw audio from queue, send to user (handles its own timeout)
                                self._send_agent_to_user(user_ws, agent_to_user_stream, duration),
                            ),
                            timeout=duration + self.grace_period,  # Extra 1s buffer for cleanup
                        )
                    except asyncio.TimeoutError:
                        logger.info("[USER THREAD] Overall timeout reached, stopping receive loop")

                    # at the end, send an RTVI message to the user to tell it to return the context history
                    self.user_context_history = await self._retrieve_context_history(user_ws)
                    # Pull the user-side scenario summary HERE (in the user
                    # thread), not from the agent thread. ``user_ws`` is owned
                    # by this event loop / thread via ``async with``; once we
                    # exit the ``with`` block the connection closes, so
                    # cross-thread access from ``agent_loop`` would race the
                    # close and fail with ``ConnectionClosedOK(1000)``. We
                    # stash the result on ``self.user_scenario_summary`` and
                    # ``run_scenario`` merges it into ``self.scenario_summary``
                    # after both threads join.
                    self.user_scenario_summary = await self._retrieve_scenario_summary(
                        user_ws, include_db=self.include_db_in_summary
                    )

            except Exception as e:
                logger.error(f"[USER THREAD] Error: {e}", exc_info=True)
            finally:
                logger.info("[USER THREAD] Exiting")

        try:
            loop.run_until_complete(user_loop())
        finally:
            loop.close()

    def agent_websocket_thread(self, duration: int):
        """
        Thread 2: Handle all agent WebSocket traffic (bidirectional).

        This thread:
        - Gets user audio from user_to_agent_queue
        - Sends user audio to agent WebSocket
        - Receives audio from agent WebSocket
        - Puts agent audio into agent_to_user_queue for user thread

        Args:
            duration: How long to run (seconds)
        """
        logger.info("[AGENT THREAD] Starting agent WebSocket thread")

        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def agent_loop():
            try:
                # Connect to agent WebSocket
                async with websockets.connect(self.agent_url, ping_timeout=None) as agent_ws:
                    self.agent_ws = agent_ws
                    logger.info(f"[AGENT THREAD] Connected to agent: {self.agent_url}")

                    # Wait for ready handshake
                    await self._send_client_ready(agent_ws)

                    # Create AudioStream for user→agent (buffering and resampling)
                    user_to_agent_stream = AudioStream(
                        chunk_size_in_seconds=self.audio_chunk_in_seconds,
                        input_sample_rate=self.user_output_sample_rate,
                        output_sample_rate=self.agent_input_sample_rate,
                        stream_resampler=False,
                        tag="USER→AGENT",
                        noise_config=self.noise_config,
                    )

                    # Send kickoff message after a delay
                    async def send_kickoff():
                        await asyncio.sleep(1)
                        logger.info("[AGENT THREAD] Sending kickoff message to agent...")
                        await self.send_text_to_agent("Hello")

                    # Run bidirectional tasks (send loop manages timeout + grace period)
                    # Add overall timeout with grace period to stop receive loop when send loop finishes
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(
                                # Get raw audio from queue, send to agent (handles its own timeout)
                                self._send_user_to_agent(agent_ws, user_to_agent_stream, duration),
                                # Receive from agent, put raw audio into queue
                                self._receive_agent_to_queue(agent_ws, duration),
                                send_kickoff(),
                            ),
                            timeout=duration + self.grace_period,
                        )
                    except asyncio.TimeoutError:
                        logger.info("[AGENT THREAD] Overall timeout reached, stopping receive loop")

                    # at the end, send RTVI messages to the agent to fetch the
                    # context history and scenario summary (actions + final DB)
                    self.agent_context_history = await self._retrieve_context_history(agent_ws)
                    self.scenario_summary = await self._retrieve_scenario_summary(
                        agent_ws, include_db=self.include_db_in_summary
                    )
                    # The user-side pull happens in ``user_websocket_thread``
                    # (each bot's WS is owned by its own thread / event
                    # loop). The merge into ``self.scenario_summary`` is
                    # done in ``run_scenario`` after both threads join.

            except Exception as e:
                logger.error(f"[AGENT THREAD] Error: {e}", exc_info=True)
            finally:
                logger.info("[AGENT THREAD] Exiting")

        try:
            loop.run_until_complete(agent_loop())
        finally:
            loop.close()

    async def _retrieve_context_history(self, ws) -> dict:
        """
        Retrieve the context history from the WebSocket. First send a message to the ws to trigger the
        `get_context_history` RTVI action, then wait for the response.
        Args:
            ws: WebSocket connection
        Returns:
            context_history: context history as a dictionary with two keys: `context` and `logs`,
                where `context` the LLM context history, and `logs` is the bot server logs.
        """
        if not ws:
            logger.warning("[CONTEXT HISTORY] WebSocket is not connected, skipping context history retrieval")
            return {}

        try:
            action_msg = {
                "label": "rtvi-ai",
                "type": "client-message",
                "id": f"get_context_history_{datetime.now().timestamp()}",
                "data": {
                    "t": "get_context_history",
                    "d": {},
                },
            }

            # Serialize as MessageFrame and send
            msg_frame = MessageFrame(data=json.dumps(action_msg))
            serialized = await self.serializer.serialize(msg_frame)
            await ws.send(serialized)

            logger.info("[CONTEXT HISTORY] Sent get_context_history action, waiting for response...")

            # Wait for the server-response with a longer timeout since log content can be large
            timeout = 15.0
            start_time = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start_time < timeout:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)

                    # Deserialize the protobuf frame
                    frame = await self.serializer.deserialize(msg)
                    if frame is None:
                        continue

                    # Extract message data from the frame
                    if not (hasattr(frame, "message") and frame.message):
                        continue

                    data = json.loads(frame.message) if isinstance(frame.message, str) else frame.message

                    if data.get("type") == "error-response":
                        logger.warning(
                            f"[CONTEXT HISTORY] bot returned an error: {(data.get('data') or {}).get('error')}"
                        )
                        break
                    if data.get("type") == "error-response":
                        logger.warning(
                            f"[SCENARIO SUMMARY] bot returned an error: {(data.get('data') or {}).get('error')}"
                        )
                        break
                    if data.get("type") == "server-response":
                        result = data.get("data", {}).get("d", {})
                        logger.info(
                            f"[CONTEXT HISTORY] Received context history "
                            f"(context: {len(result.get('context', []))} messages, "
                            f"logs: {len(result.get('logs', ''))} chars)"
                        )
                        return result
                except asyncio.TimeoutError:
                    continue

            logger.warning("[CONTEXT HISTORY] Timeout waiting for context history response")
            return {}
        except Exception as e:
            logger.warning(f"[CONTEXT HISTORY] Error retrieving context history: {e}")
            return {}

    async def _retrieve_scenario_summary(self, ws, include_db: bool = False) -> dict:
        """Retrieve the scenario summary from the bot via the
        ``get_scenario_summary`` RTVI action. Mirrors ``_retrieve_context_history``.

        Args:
            ws: WebSocket connection to the agent bot.
            include_db: When ``True``, ask the bot to inline the ``db``
                dict alongside the hash. Set by the runner via
                ``Scenario.db_state_assertions`` (predicate evaluation needs
                the actual DB, not just a hash). Off by default to preserve
                the existing hash-out behavior for retail (whose 7 MB DB
                would exceed pipecat's 1 MB WS frame cap).

        Returns:
            Dict with at minimum ``{"actions": list, "db_hash": str|None}``.
            When ``include_db=True``, also includes ``"db": dict|None``.
            Empty ``{}`` if the bot didn't register the action (legacy bot)
            or timed out.

            **One DB per pull.** This pull returns *this bot's* DB only.
            For telecom dual-state, the bridge calls this method
            separately for ``agent_ws`` and ``user_ws`` and merges the two
            results into ``self.scenario_summary`` under the ``db_hash``
            and ``user_db_hash`` keys respectively. Currently only the
            agent-side pull is wired; user-side pull is added when the
            first telecom scenario is ported.
        """
        if not ws:
            logger.warning("[SCENARIO SUMMARY] WebSocket is not connected, skipping scenario summary retrieval")
            return {}

        try:
            action_msg = {
                "label": "rtvi-ai",
                "type": "client-message",
                "id": f"get_scenario_summary_{datetime.now().timestamp()}",
                "data": {
                    "t": "get_scenario_summary",
                    "d": {
                        "include_db": include_db,
                    },
                },
            }
            msg_frame = MessageFrame(data=json.dumps(action_msg))
            serialized = await self.serializer.serialize(msg_frame)
            await ws.send(serialized)
            logger.info("[SCENARIO SUMMARY] Sent get_scenario_summary action, waiting for response...")

            timeout = 15.0
            start_time = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start_time < timeout:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    frame = await self.serializer.deserialize(msg)
                    if frame is None:
                        continue
                    if not (hasattr(frame, "message") and frame.message):
                        continue
                    data = json.loads(frame.message) if isinstance(frame.message, str) else frame.message
                    if data.get("type") == "server-response":
                        result = data.get("data", {}).get("d", {})
                        actions = result.get("actions", [])
                        # ``db`` is only inlined when the bot received
                        # ``include_db=True``. Log dict size when present;
                        # the hash-out default keeps the response tiny.
                        inline_db = result.get("db")
                        if include_db:
                            db_keys = len(inline_db) if isinstance(inline_db, dict) else 0
                            logger.info(
                                f"[SCENARIO SUMMARY] Received summary "
                                f"(actions: {len(actions)}, inline db top-level keys: {db_keys})"
                            )
                        else:
                            logger.info(
                                f"[SCENARIO SUMMARY] Received summary "
                                f"(actions: {len(actions)}, db_hash: {result.get('db_hash')})"
                            )
                        return result
                except asyncio.TimeoutError:
                    continue

            logger.warning("[SCENARIO SUMMARY] Timeout waiting for scenario summary response")
            return {}
        except Exception as e:
            logger.warning(f"[SCENARIO SUMMARY] Error retrieving scenario summary: {e}")
            return {}

    async def _apply_initialization(
        self,
        scenario,
        *,
        user_shared_state_init: dict,
        agent_shared_state_init: dict,
    ) -> None:
        """Send ``apply_initialization`` to BOTH bots — always, even with no init actions.

        Bot-side dispatch via the ``apply_initialization`` RTVI action.
        The handler does three things per bot:

          1. Merge ``shared_state_init`` JSON into the bot's
             ``shared_state`` (preserving runtime sentinels stashed by
             ``update_system_prompt``).
          2. If ``db_path`` is in the merged state, resolve it against
             ``EVAL_DATA_ROOT`` and replace with the loaded ``db`` dict.
          3. Dispatch each per-side init function record (filtered by
             ``side``) against the now-loaded ``db``.

        Always called once per bot, even when ``scenario.initialization_actions``
        is empty — steps 1 and 2 (state merge + DB load) must run for every
        tau2 scenario regardless. Single-side domains (eva / airline / retail)
        whose ``setup_shared_state`` populates ``db_path`` rely on this call
        to trigger the DB load. The bot's handler is fast (~tens of ms) when
        actions list is empty.

        **Failure handling:** any per-bot ``success: false`` response
        (invalid JSON, missing DB file, missing init function in the bot's
        registry, dispatcher exception) raises ``RuntimeError`` so the
        calling ``prepare_for_scenario`` aborts before the conversation
        starts. Partial seeding produces noisy / unscoreable runs.
        """
        actions = getattr(scenario, "initialization_actions", None) or []
        domain = getattr(scenario, "domain", "default")

        agent_actions = [a for a in actions if a.get("side") == "agent"]
        user_actions = [a for a in actions if a.get("side") == "user"]
        unsided = [a for a in actions if a.get("side") not in ("agent", "user")]
        if unsided:
            raise RuntimeError(
                f"initialization_actions contains entries with unknown side "
                f"(expected 'user' or 'agent'): {unsided}. Check the scenario "
                f"translation boundary — upstream tau2 uses env_type ∈ "
                f"{{'user', 'assistant'}}, which must be renamed to "
                f"side ∈ {{'user', 'agent'}} before the dict reaches the bridge."
            )

        logger.info(
            f"[APPLY INIT] domain={domain!r}, total={len(actions)} action(s) "
            f"(agent_side={len(agent_actions)}, user_side={len(user_actions)}); "
            f"sending to both bots for state initialization."
        )

        # Always call both — even with empty per-side action lists. The
        # handler does state merge + DB load regardless of whether there
        # are init functions to dispatch.
        await self._send_apply_initialization(self.agent_ws, "agent", domain, agent_shared_state_init, agent_actions)
        await self._send_apply_initialization(self.user_ws, "user", domain, user_shared_state_init, user_actions)

    async def _send_apply_initialization(
        self,
        ws,
        side_label: str,
        domain: str,
        shared_state_init: dict,
        actions: List[dict],
    ) -> None:
        """Send one ``apply_initialization`` action and wait for the result.

        ``side_label`` is the bot label (``"agent"`` or ``"user"``) used only
        in log lines and error messages — the bot itself doesn't need to know
        which side it is, since each action in the payload already carries
        its own ``side`` field that the dispatcher routes by.

        Raises:
            RuntimeError: if the bot returns ``success: false`` or the request
                times out. The caller (``_apply_initialization``) propagates
                this up so ``prepare_for_scenario`` aborts cleanly.
        """
        if not ws:
            raise RuntimeError(
                f"[APPLY INIT] {side_label}_ws is not connected; cannot "
                f"send apply_initialization (shared_state_init keys: "
                f"{list(shared_state_init.keys())}, actions: {len(actions)})."
            )

        action_id = f"apply_init_{side_label}_{datetime.now().timestamp()}"
        action_msg = {
            "label": "rtvi-ai",
            "type": "client-message",
            "id": action_id,
            "data": {
                "t": "apply_initialization",
                "d": {
                    "domain": domain,
                    "shared_state_init": json.dumps(shared_state_init),
                    "actions": actions,
                },
            },
        }
        msg_frame = MessageFrame(data=json.dumps(action_msg))
        serialized = await self.serializer.serialize(msg_frame)
        await ws.send(serialized)
        logger.info(
            f"[APPLY INIT] Sent apply_initialization to {side_label} bot "
            f"(shared_state_init keys: {list(shared_state_init.keys())}, "
            f"{len(actions)} action(s)); waiting for response (id={action_id!r})..."
        )

        # Wait pattern mirrors ``_retrieve_scenario_summary``. 15s budget is
        # generous since init dispatch is a pure-Python dict-mutation loop
        # (no LLM, no I/O).
        #
        # ID-match required: this call happens right after
        # ``update_system_prompt`` which has ``result="bool"``. Without
        # ID-matching, we'd grab the update_system_prompt response (bool
        # truthy) instead of ours (dict {"success", "errors"}) and the
        # ``result.get("success")`` access below would raise
        # ``AttributeError: 'bool' object has no attribute 'get'``.
        timeout = 15.0
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < timeout:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                frame = await self.serializer.deserialize(msg)
                if frame is None:
                    continue
                if not (hasattr(frame, "message") and frame.message):
                    continue
                data = json.loads(frame.message) if isinstance(frame.message, str) else frame.message
                if data.get("type") not in ("server-response", "error-response"):
                    continue
                # Only consume responses whose id matches the request we sent.
                # Skip stragglers (e.g. update_system_prompt server-response
                # that races us at scenario start).
                if data.get("id") != action_id:
                    logger.debug(f"[APPLY INIT] Skipping unrelated response id={data.get('id')!r}")
                    continue
                # A handler that raised produces error-response, not a
                # success=False payload. Fail loudly rather than sitting here
                # until the timeout expires.
                if data.get("type") == "error-response":
                    raise RuntimeError(
                        f"[APPLY INIT] {side_label} bot returned an error: {(data.get('data') or {}).get('error')}"
                    )
                result = data.get("data", {}).get("d", {})
                if result.get("success"):
                    logger.info(f"[APPLY INIT] {side_label} bot applied {len(actions)} action(s) successfully")
                    return
                raise RuntimeError(
                    f"[APPLY INIT] {side_label} bot reported failure on "
                    f"{len(actions)} action(s): {result.get('errors')}"
                )
            except asyncio.TimeoutError:
                continue

        raise RuntimeError(
            f"[APPLY INIT] Timeout ({timeout}s) waiting for {side_label} bot "
            f"to apply {len(actions)} initialization action(s)."
        )

    # ---------------------------------------------------------------------
    # Cross-side sync (mirrors upstream tau2's Environment.sync_tools())
    # ---------------------------------------------------------------------

    async def _setup_cross_side_sync(self, scenario) -> None:
        """Prepare shadow DBs + tool map for cross-side state propagation.

        Called from ``prepare_for_scenario`` once per scenario,
        **after** ``_apply_initialization`` so the bot-side
        live state is already at its post-init starting point. We
        mirror that here by replaying the same init actions onto the
        shadow DBs, then run a one-shot ``sync_state`` to propagate
        any cross-side state that should be coherent at conversation
        start (e.g. agent-side ``set_data_usage(15.1)`` flipping
        user-side ``surroundings.mobile_data_usage_exceeded`` to True).

        No-op when the scenario's ``sync_state`` is the inherited
        default (single-side domains).
        """
        # Reset per-scenario sync state regardless of whether the
        # scenario opts in — old shadow DBs from a previous scenario
        # must not leak into this one.
        self.sync_enabled = False
        self.scenario_instance = None
        self.shadow_state = None
        self.shadow_tool_map = {}

        # Detect override by walking the MRO. ``sync_state`` lives on
        # ``Scenario`` as a default no-op; any override on a subclass
        # (currently only ``Tau2TelecomBaseScenario``) signals opt-in.
        from nemo_voice_agent.evaluation.scenarios.classes import Scenario as _BaseScenario

        if type(scenario).sync_state is _BaseScenario.sync_state:
            # Default no-op; nothing to do.
            return

        # Deep-copy both DBs so live mutations during replay don't
        # leak into ``scenario``'s cached_property results (which are
        # shared across the scenario lifetime).
        try:
            agent_db = copy.deepcopy(scenario.db) if hasattr(scenario, "db") else {}
            user_db = copy.deepcopy(scenario.user_db) if hasattr(scenario, "user_db") else {}
        except Exception as exc:
            logger.warning(
                f"[SYNC SETUP] Couldn't deepcopy scenario DBs: {type(exc).__name__}: {exc}. "
                f"Cross-side sync disabled for this scenario."
            )
            return

        # Replay ``initialization_actions`` onto the shadow DBs so they
        # mirror the bots' post-init state. Init functions don't fire
        # ``action-applied`` events (they aren't WriteScenarioTool
        # subclasses), so this is the only place we can bring shadow
        # state into alignment with what the bots already have.
        domain = getattr(scenario, "domain", "default")
        init_actions = getattr(scenario, "initialization_actions", None) or []
        if init_actions:
            from nemo_voice_agent.evaluation.initialization_functions import (
                apply_initialization_actions as _apply_init,
            )

            for side, target_db in (("agent", agent_db), ("user", user_db)):
                side_actions = [a for a in init_actions if a.get("side") == side]
                if not side_actions:
                    continue
                result = _apply_init(domain=domain, actions=side_actions, db=target_db)
                if not result["success"]:
                    logger.warning(
                        f"[SYNC SETUP] Shadow init replay failed on {side} side: "
                        f"{result['errors']}. Cross-side sync disabled."
                    )
                    return

        self.scenario_instance = scenario
        self.shadow_state = {
            "db": agent_db,
            "user_db": user_db,
            "actions": [],
        }
        try:
            self.shadow_tool_map = scenario._build_tool_map(self.shadow_state)
        except Exception as exc:
            logger.warning(
                f"[SYNC SETUP] scenario._build_tool_map raised: {type(exc).__name__}: {exc}. Cross-side sync disabled."
            )
            self.shadow_state = None
            return

        self.sync_enabled = True
        logger.info(
            f"[SYNC SETUP] Cross-side sync enabled for scenario {scenario.name!r} "
            f"(shadow tool map: {len(self.shadow_tool_map)} entries)"
        )

        # One-shot post-init sync. Cross-side state that's coherent
        # from initial-state alone (e.g. agent ``data_used_gb=15.1`` →
        # user ``mobile_data_usage_exceeded=True``) gets propagated to
        # the bots BEFORE the conversation starts. Without this, the
        # user-sim's first ``check_*`` / ``run_speed_test`` would see
        # stale defaults and the conversation would diverge from the
        # task's intended starting state.
        try:
            deltas = scenario.sync_state(
                agent_db=self.shadow_state["db"],
                user_db=self.shadow_state["user_db"],
            )
        except Exception as exc:
            logger.warning(
                f"[SYNC SETUP] Initial sync_state raised: {type(exc).__name__}: {exc}. "
                f"Conversation starts without initial propagation."
            )
            return

        agent_delta = deltas.get("agent") or {}
        user_delta = deltas.get("user") or {}
        if agent_delta or user_delta:
            logger.info(
                f"[SYNC SETUP] Initial sync dispatch: agent_delta={list(agent_delta)}, user_delta={list(user_delta)}"
            )
            tasks = []
            if agent_delta:
                tasks.append(self._send_apply_sync_delta(self.agent_ws, "agent", domain, agent_delta))
            if user_delta:
                tasks.append(self._send_apply_sync_delta(self.user_ws, "user", domain, user_delta))
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _propagate_cross_side_sync(self, action: dict, source_side: str) -> None:
        """Replay one action onto the shadow DBs, run scenario.sync_state, dispatch deltas.

        Called from ``_monitor_agent_message`` / ``_monitor_user_message``
        when an ``action-applied`` RTVI server message arrives. ``source_side``
        is which side's monitor saw it.

        Steps:
          1. Replay ``action`` onto the shadow state via the shadow
             tool map (uses each tool's sync ``invoke``).
          2. Call ``scenario.sync_state(agent_db, user_db)`` — returns
             ``{"agent": delta_dict, "user": delta_dict}``.
          3. For each non-empty side delta, send an ``apply_sync_delta``
             RTVI action to that bot's WebSocket.

        No-op when ``sync_enabled`` is False (single-side scenarios).
        """
        if not self.sync_enabled or self.shadow_state is None:
            return

        async with self.sync_lock:
            name = action.get("name")
            args = action.get("arguments") or {}
            tool = self.shadow_tool_map.get(name)
            if tool is None:
                logger.warning(
                    f"[SYNC] No shadow tool for action name={name!r} (source_side={source_side!r}); "
                    f"shadow state will drift. Available: {list(self.shadow_tool_map)[:10]}..."
                )
                return
            try:
                tool.invoke(**args)
            except Exception as exc:
                logger.warning(
                    f"[SYNC] Shadow replay failed for {name!r}: {type(exc).__name__}: {exc}. Skipping sync this turn."
                )
                return

            try:
                deltas = self.scenario_instance.sync_state(
                    agent_db=self.shadow_state["db"],
                    user_db=self.shadow_state["user_db"],
                )
            except Exception as exc:
                logger.warning(f"[SYNC] sync_state raised: {type(exc).__name__}: {exc}. Skipping.")
                return

            agent_delta = deltas.get("agent") or {}
            user_delta = deltas.get("user") or {}
            if not agent_delta and not user_delta:
                return

            domain = type(self.scenario_instance).domain
            logger.info(
                f"[SYNC] Propagating after {source_side}-side {name!r}: "
                f"agent_delta={list(agent_delta)}, user_delta={list(user_delta)}"
            )
            tasks = []
            if agent_delta:
                tasks.append(self._send_apply_sync_delta(self.agent_ws, "agent", domain, agent_delta))
            if user_delta:
                tasks.append(self._send_apply_sync_delta(self.user_ws, "user", domain, user_delta))
            # Both pushes can run concurrently — they target different bots.
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_apply_sync_delta(self, ws, side_label: str, domain: str, delta: dict) -> None:
        """Send a single ``apply_sync_delta`` RTVI action to a bot.

        Mirrors ``_send_apply_initialization``: builds an action
        message with a unique id, awaits the response, logs success or
        failure. Failures are warnings — sync drift is recoverable as
        long as the next propagation cycle eventually catches up.
        """
        if not ws:
            logger.warning(f"[SYNC] {side_label}_ws is not connected; cannot push {len(delta)}-key sync delta.")
            return

        action_id = f"apply_sync_delta_{side_label}_{datetime.now().timestamp()}"
        action_msg = {
            "label": "rtvi-ai",
            "type": "client-message",
            "id": action_id,
            "data": {
                "t": "apply_sync_delta",
                "d": {
                    "domain": domain,
                    "delta": delta,
                },
            },
        }
        msg_frame = MessageFrame(data=json.dumps(action_msg))
        serialized = await self.serializer.serialize(msg_frame)
        try:
            await ws.send(serialized)
            logger.debug(f"[SYNC] Sent apply_sync_delta to {side_label} bot ({len(delta)} key(s)); id={action_id!r}")
        except Exception as exc:
            # The connection may have closed between turns; log + continue.
            logger.warning(f"[SYNC] Failed to send apply_sync_delta to {side_label}: {type(exc).__name__}: {exc}")

    async def _receive_user_to_queue(self, user_ws, duration: float):
        """Receive audio from user WebSocket and put into queue for agent thread."""
        return await self._receive_to_queue(
            ws=user_ws,
            duration=duration,
            direction="USER→AGENT",
            queue=self.user_to_agent_queue,
            monitor_func=self._monitor_user_message,
        )

    async def _send_agent_to_user(self, user_ws, audio_stream: AudioStream, duration: int):
        """Get audio from queue, process through AudioStream, send to user WebSocket."""
        return await self._send_audio_stream(
            audio_stream=audio_stream,
            dest_ws=user_ws,
            direction="AGENT→USER",
            duration=duration,
            source_queue=self.agent_to_user_queue,
            sent_chunks_list=self.sent_to_user_chunks,
        )

    async def _send_user_to_agent(self, agent_ws, audio_stream: AudioStream, duration: int):
        """Get audio from queue, process through AudioStream, send to agent WebSocket."""
        return await self._send_audio_stream(
            audio_stream=audio_stream,
            dest_ws=agent_ws,
            direction="USER→AGENT",
            duration=duration,
            source_queue=self.user_to_agent_queue,
            sent_chunks_list=self.sent_to_agent_chunks,
        )

    async def _receive_agent_to_queue(self, agent_ws, duration: float):
        """Receive audio from agent WebSocket and put into queue for user thread."""
        return await self._receive_to_queue(
            ws=agent_ws,
            duration=duration,
            direction="AGENT→USER",
            queue=self.agent_to_user_queue,
            monitor_func=self._monitor_agent_message,
        )

    async def run_scenario(self, duration: int = 300):
        """
        Route audio between agents and monitor conversation.
        Uses separate threads per WebSocket to eliminate asyncio contention.

        Args:
            duration: Duration of the evaluation in seconds
        """
        if not self.bridge_ready:
            raise RuntimeError("[RUN SCENARIO] Bridge is not ready, please call `bridge.prepare_for_scenario()` first")
        if self.needs_reset:
            raise RuntimeError(
                "Bridge needs reset before running a new scenario, "
                "please call `bridge.reset()` or `bridge.prepare_for_scenario()` first"
            )

        logger.info(f"[RUN SCENARIO] Running scenario for {duration} seconds...")
        self.metrics.start_time = datetime.now()
        self.metrics.end_time = None
        # Clear state for this run
        self.stop_event.clear()
        self.stop_reason = STOP_REASON_TIMEOUT
        self.sent_to_agent_chunks = []
        self.sent_to_user_chunks = []
        self.user_context_history = None
        self.agent_context_history = None
        self.scenario_summary = None
        self.user_scenario_summary = None
        self.token_usage = self._fresh_token_usage()

        # Clear thread-safe queues
        self.user_to_agent_queue = queue.Queue()
        self.agent_to_user_queue = queue.Queue()

        # Create and start threads
        user_thread = threading.Thread(
            target=self.user_websocket_thread,
            args=(duration,),
            name="UserWebSocketThread",
        )
        agent_thread = threading.Thread(
            target=self.agent_websocket_thread,
            args=(duration,),
            name="AgentWebSocketThread",
        )

        # Start both threads
        logger.info("[RUN SCENARIO] Starting threads for user and agent...")

        # Set thread start timestamp for conversation log timing (aligns with bridge_audio_log.wav)
        loop = asyncio.get_event_loop()
        self.metrics.thread_start_timestamp = loop.time()

        user_thread.start()
        agent_thread.start()

        # Wait for both threads to complete (in async context)
        logger.info("[RUN SCENARIO] Waiting for threads to complete...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, user_thread.join)
        await loop.run_in_executor(None, agent_thread.join)

        logger.info("[RUN SCENARIO] Both user and agent threads completed")

        # Merge the user-thread-pulled scenario summary into the
        # agent-side one. Done here (post-join), not inside either thread,
        # because each ``ws`` is owned by its own thread/event loop and
        # closes when the thread's ``async with`` block exits. The agent
        # thread populated ``self.scenario_summary`` from agent_ws; the
        # user thread populated ``self.user_scenario_summary`` from
        # user_ws. We label the user payload under parallel keys
        # (``user_db_hash`` / ``user_db``) and stamp side="user" on its
        # action records before appending to the merged list. For
        # single-side domains (eva / airline / retail) the user bot has
        # no DB seeded so ``user_scenario_summary`` is
        # ``{actions: [], db_hash: None}`` — harmless no-op merge.
        user_summary = self.user_scenario_summary or {}
        if self.scenario_summary is None:
            self.scenario_summary = {}
        self.scenario_summary["user_db_hash"] = user_summary.get("db_hash")
        if self.include_db_in_summary:
            self.scenario_summary["user_db"] = user_summary.get("db")
        user_actions = user_summary.get("actions") or []
        if user_actions:
            agent_actions = self.scenario_summary.get("actions") or []
            self.scenario_summary["actions"] = list(agent_actions) + [{**a, "side": "user"} for a in user_actions]

        self.metrics.end_time = datetime.now()
        # Finalize any in-progress turns at end of scenario
        loop = asyncio.get_event_loop()
        timestamp = loop.time()
        self._finalize_speaker_turn("user", timestamp)
        self._finalize_speaker_turn("agent", timestamp)

        # Write conversation log with post-hoc latency calculation
        self._save_final_response()
        self._save_scenario_db()
        self._save_conversation_log()
        self._save_audio_log()
        self._save_seglst()
        self._save_user_agent_history()
        logger.info(f"[RUN SCENARIO] Saved audio and logs to: {Path(self.log_file).parent}")
        self.needs_reset = True
        self.bridge_ready = False

    def _build_conversation_log(self):
        """
        Build conversation log entries from finalized segments with computed latencies.

        Called after all segments are finalized so that latency calculation has access
        to all user and agent segments. For each agent segment, latency is computed as:
            agent.start_time - previous_user.end_time
        Positive = normal response delay, negative = agent interrupted/barged in early.

        Applies turn_start_offset_secs and turn_end_offset_secs to match seglst timestamps.
        """
        sorted_segments = sorted(self.metrics.segments, key=lambda s: s.start_time)

        self.metrics.log_entries = []
        last_user_end = None

        for seg in sorted_segments:
            start = seg.start_time + self.turn_start_offset_secs
            end = seg.end_time + self.turn_end_offset_secs
            # Ensure offsets don't produce negative duration
            if end <= start:
                start = seg.start_time
                end = seg.end_time

            if seg.speaker == "user":
                last_user_end = end
                latency_ms = None
            else:  # agent
                if last_user_end is not None:
                    latency_ms = (start - last_user_end) * 1000
                else:
                    latency_ms = None

            log_entry = self._format_turn_log(seg.speaker, seg.transcript, start, end, latency_ms)
            self.metrics.log_entries.append((start, log_entry))

    def _save_conversation_log(self):
        """Build and write conversation log entries sorted by start time, with computed latencies."""
        if not self.log_file or not self.metrics.segments:
            return

        # Build log entries from finalized segments with post-hoc latency
        self._build_conversation_log()

        try:
            sorted_entries = sorted(self.metrics.log_entries, key=lambda x: x[0])
            with open(self.log_file, "a") as f:
                for _start_time, log_entry in sorted_entries:
                    f.write(log_entry)
                f.write("\n\n" + "=" * 80 + "\n")
                f.write(f"End time: {self.metrics.end_time.isoformat()}\n")
                f.write(f"Stop reason: {self.stop_reason}\n")
                f.write("=" * 80 + "\n")
            logger.info(f"[LOG] Wrote {len(sorted_entries)} conversation turns to log file (sorted by time)")
        except Exception as e:
            logger.error(f"[LOG] Error writing sorted log entries: {e}")

    @staticmethod
    def _resample_audio(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Resample audio array using soxr. Returns int16 array."""
        if from_rate == to_rate or len(audio) == 0:
            return audio
        return soxr.resample(audio, from_rate, to_rate, quality="VHQ").astype(np.int16)

    def _save_audio_log(self):
        """Save final sent audio chunks to disk as stereo WAV for debugging."""
        if not self.bridge_audio_file:
            logger.warning("[DEBUG] No bridge_audio_file to save audio")
            return

        output_path = Path(self.bridge_audio_file)

        if not self.sent_to_agent_chunks and not self.sent_to_user_chunks:
            logger.info("[DEBUG] No audio chunks to save")
            return

        logger.info(f"[DEBUG] Saving bridge audio log to {output_path}")
        # Convert audio chunks to numpy arrays
        # Channel 0 (Left): USER→AGENT audio at agent_input_sample_rate
        # Channel 1 (Right): AGENT→USER audio at user_input_sample_rate

        channel0 = np.array([], dtype=np.int16)
        channel1 = np.array([], dtype=np.int16)

        if self.sent_to_agent_chunks:
            audio_data = b"".join(self.sent_to_agent_chunks)
            channel0 = np.frombuffer(audio_data, dtype=np.int16)

        if self.sent_to_user_chunks:
            audio_data = b"".join(self.sent_to_user_chunks)
            channel1 = np.frombuffer(audio_data, dtype=np.int16)

        # Resample both channels to output_sample_rate (typically 16kHz)
        target_rate = self.output_sample_rate

        channel0 = self._resample_audio(channel0, self.agent_input_sample_rate, target_rate)
        channel1 = self._resample_audio(channel1, self.user_input_sample_rate, target_rate)

        # Pad shorter channel with silence to match longer one
        max_length = max(len(channel0), len(channel1))

        if len(channel0) < max_length:
            channel0 = np.pad(
                channel0,
                (0, max_length - len(channel0)),
                mode="constant",
                constant_values=0,
            )

        if len(channel1) < max_length:
            channel1 = np.pad(
                channel1,
                (0, max_length - len(channel1)),
                mode="constant",
                constant_values=0,
            )

        # Interleave channels for stereo: [L, R, L, R, ...]
        stereo_data = np.empty(max_length * 2, dtype=np.int16)
        stereo_data[0::2] = channel0  # Left channel (USER→AGENT)
        stereo_data[1::2] = channel1  # Right channel (AGENT→USER)

        # Save as stereo WAV
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(2)  # Stereo
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(target_rate)
            wav_file.writeframes(stereo_data.tobytes())

        duration = max_length / target_rate
        logger.info(f"[DEBUG] Saved stereo bridge audio: {output_path}")
        logger.info(f"        Left (USER→AGENT): {len(self.sent_to_agent_chunks)} chunks")
        logger.info(f"        Right (AGENT→USER): {len(self.sent_to_user_chunks)} chunks")
        logger.info(f"        Duration: {duration:.2f}s, Sample rate: {target_rate}Hz")

    def _save_bot_server_history(self, output_dir: Union[str, Path], context_history: dict, role: str = ""):
        """Save the bot server context history to a JSON file under the output directory."""
        if not output_dir:
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        context = eval(context_history.get("context", "[]"))
        if isinstance(context, str):
            try:
                context = json.loads(context)
            except Exception as e:
                logger.error(f"Error loading context into json object: {e}. Context: {context}")

        file_name = f"llm_context_{role}.json" if role else "llm_context.json"
        context_file = output_dir / file_name
        with open(context_file, "w") as f:
            json.dump(context, f, indent=2)

    def _save_user_agent_history(self):
        """Save the user and agent context history to a JSON file under the output directory."""
        if not self.output_dir:
            return

        output_dir_user = Path(self.output_dir) / "bot_logs_user"
        output_dir_agent = Path(self.output_dir) / "bot_logs_agent"
        self._save_bot_server_history(output_dir_user, self.user_context_history)
        self._save_bot_server_history(output_dir_agent, self.agent_context_history)

    def _log_rtvi_event(self, side: str, message_type: str, data: dict) -> None:
        """Emit a uniform side-tagged DEBUG log line for an RTVI event.

        Each monitor (``_monitor_user_message`` / ``_monitor_agent_message``) is
        bound to one side by construction, so we know which side emitted the
        event without time-correlation. This helper writes a one-line tag plus
        expanded payload lines for the message types that carry analytically
        useful structured data (TTFB values, token counts, transcription text,
        RTVI action names).

        Skipped types (``_RTVI_TYPES_ALREADY_TAGGED``): the monitors already
        emit dedicated side-tagged lines like ``[AGENT TTS] <text>`` for these,
        so adding ``[AGENT EVENT] type=bot-tts-text`` on top would just
        duplicate. Their expanded payload (the TTS text) is already in the
        existing line.

        Skipping is by *event-tag emission only* — the underlying RTVI message
        still flows through pipecat's ``ProtobufFrameSerializer:deserialize``
        DEBUG log, so nothing is lost from the bridge log.
        """
        if message_type not in _RTVI_TYPES_ALREADY_TAGGED:
            logger.debug(f"[{side} EVENT] type={message_type}")

        body = data.get("data") or {}
        if message_type == "metrics":
            for ttfb in body.get("ttfb") or []:
                try:
                    value_str = f"{float(ttfb.get('value', 0)):.3f}s"
                except (TypeError, ValueError):
                    value_str = str(ttfb.get("value"))
                logger.debug(f"[{side} METRICS] ttfb processor={ttfb.get('processor')} value={value_str}")
            for tok in body.get("tokens") or []:
                prompt_n = int(tok.get("prompt_tokens") or 0)
                completion_n = int(tok.get("completion_tokens") or 0)
                logger.debug(
                    f"[{side} METRICS] tokens prompt={tok.get('prompt_tokens')} "
                    f"completion={tok.get('completion_tokens')}"
                )
                # Accumulate per-side counters so the runner can pull a
                # canonical total into metrics.json after the scenario ends.
                # The side label here ("AGENT"/"USER") was already established
                # by the caller of _log_rtvi_event from which monitor produced
                # it, so the bucket lookup can't get crossed.
                bucket = self.token_usage.get(side.lower())
                if bucket is not None:
                    bucket["n_calls"] += 1
                    bucket["prompt"] += prompt_n
                    bucket["completion"] += completion_n
        elif message_type == "user-transcription":
            # tau2 retail run on 2026-06-03 showed STT collapsing spelled-out
            # digits back to numerals; final transcripts let us diff against
            # bot_logs_user/llm_context.json without reading both files.
            # Partials surface STT instability / mid-utterance corrections.
            text = body.get("text", "")
            kind = "final" if body.get("final") else "partial"
            logger.debug(f"[{side} STT {kind}] {text!r}")
        elif message_type == "action":
            action = body.get("action") or data.get("action")
            logger.debug(f"[{side} ACTION] {action}")

    async def _monitor_user_message(self, frame):
        """
        Monitor user messages for timing and transcripts.

        Turn lifecycle: BOT_STARTED_SPEAKING → BOT_TTS_TEXT (accumulate) → BOT_STOPPED_SPEAKING (finalize).
        """
        timestamp = asyncio.get_event_loop().time()

        if frame is None:
            return

        # logger.debug(f"[USER MONITOR] Frame type: {type(frame).__name__}, has audio: {hasattr(frame, 'audio')}")

        # Handle audio frames
        if hasattr(frame, "audio") and frame.audio:
            self.metrics.user_last_audio_time = timestamp
            return

        # Handle RTVI protocol messages
        if not (hasattr(frame, "message") and frame.message):
            return

        data = json.loads(frame.message) if isinstance(frame.message, str) else frame.message
        message_type = data.get("type", "")
        self._log_rtvi_event("USER", message_type, data)

        if message_type == RTVI_BOT_STARTED_SPEAKING:
            # Defensive: close previous turn if it wasn't properly stopped
            self._finalize_speaker_turn("user", timestamp)

            # Start new turn
            relative_time = self._get_relative_time(timestamp)
            self.metrics.current_user_segment = SegmentEntry(
                start_time=relative_time,
                end_time=relative_time,
                speaker="user",
                transcript="",
            )
            self.metrics.user_current_transcript = ""
            logger.debug("[TIMING] User started speaking")

        elif message_type == RTVI_BOT_TTS_TEXT:
            text = str(data.get("data", {}).get("text", ""))
            logger.debug(f"[USER TTS] {text}")
            if text:
                self.metrics.user_current_transcript += text

        elif message_type == RTVI_BOT_STOPPED_SPEAKING:
            logger.debug("[USER STOPPED SPEAKING]")
            self.metrics.user_last_audio_time = timestamp
            self.metrics.waiting_for_agent_response = True

            segment = self._finalize_speaker_turn("user", timestamp)
            if segment:
                self.metrics.last_user_transcript = segment.transcript
                self.metrics.turns.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "role": "user",
                        "text": segment.transcript,
                    }
                )

        elif message_type == RTVI_BOT_SERVER_MESSAGE:
            # Cross-side sync trigger: a write tool on the user bot
            # emitted ``action-applied`` (e.g. telecom phone-control
            # tools). Replay onto shadow DBs and propagate any
            # cross-side delta to the agent bot.
            inner = data.get("data") or {}
            if isinstance(inner, dict) and inner.get("type") == "action-applied":
                action_record = inner.get("action") or {}
                await self._propagate_cross_side_sync(action_record, source_side="user")

    async def _monitor_agent_message(self, frame):
        """
        Monitor agent messages for timing and transcripts.

        Turn lifecycle: BOT_STARTED_SPEAKING → BOT_TTS_TEXT (accumulate) → BOT_STOPPED_SPEAKING (finalize).
        Latency is measured from user's last audio to agent's first audio frame.
        """
        timestamp = asyncio.get_event_loop().time()

        if frame is None:
            return

        # logger.debug(f"[AGENT MONITOR] Frame type: {type(frame).__name__}, has audio: {hasattr(frame, 'audio')}")

        # Handle audio frames — measure latency on first agent audio after user stops
        if hasattr(frame, "audio") and frame.audio:
            if self.metrics.waiting_for_agent_response and self.metrics.user_last_audio_time:
                latency_ms = (timestamp - self.metrics.user_last_audio_time) * 1000
                latency = ResponseLatency(
                    user_stop_time=self.metrics.user_last_audio_time,
                    agent_start_time=timestamp,
                    latency_ms=latency_ms,
                    user_transcript=self.metrics.last_user_transcript,
                )
                self.metrics.latencies.append(latency)
                self.metrics.waiting_for_agent_response = False
                logger.info(f"[LATENCY] Response latency: {latency_ms:.1f}ms")

            self.metrics.agent_last_audio_time = timestamp
            return

        # Handle RTVI protocol messages
        if not (hasattr(frame, "message") and frame.message):
            return

        data = json.loads(frame.message) if isinstance(frame.message, str) else frame.message
        message_type = data.get("type", "")
        self._log_rtvi_event("AGENT", message_type, data)

        if message_type == RTVI_BOT_STARTED_SPEAKING:
            logger.debug("[AGENT STARTED SPEAKING]")
            # Defensive: close previous turn if it wasn't properly stopped
            self._finalize_speaker_turn("agent", timestamp)

            # Start new turn
            relative_time = self._get_relative_time(timestamp)
            self.metrics.current_agent_segment = SegmentEntry(
                start_time=relative_time,
                end_time=relative_time,
                speaker="agent",
                transcript="",
            )
            self.metrics.agent_current_transcript = ""

        elif message_type == RTVI_BOT_TTS_TEXT:
            text = str(data.get("data", {}).get("text", ""))
            logger.debug(f"[AGENT TTS] {text}")
            if text:
                self.metrics.agent_current_transcript += text

        elif message_type == RTVI_BOT_STOPPED_SPEAKING:
            logger.debug("[AGENT STOPPED SPEAKING]")
            segment = self._finalize_speaker_turn("agent", timestamp)
            if segment:
                # Update the last latency measurement with agent transcript
                if self.metrics.latencies and not self.metrics.latencies[-1].agent_transcript:
                    self.metrics.latencies[-1].agent_transcript = segment.transcript

                self.metrics.turns.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "role": "agent",
                        "text": segment.transcript,
                    }
                )

        elif message_type == RTVI_BOT_SERVER_MESSAGE:
            inner = data.get("data") or {}
            # Cross-side sync trigger: a write tool on the agent bot
            # emitted ``action-applied``. Replay onto shadow DBs and
            # propagate any cross-side delta to the user bot.
            if isinstance(inner, dict) and inner.get("type") == "action-applied":
                action_record = inner.get("action") or {}
                await self._propagate_cross_side_sync(action_record, source_side="agent")
                return
            text = str(inner.get("text", "")) if isinstance(inner, dict) else ""
            if text:
                logger.info(f"[AGENT SERVER MESSAGE] {text}")
                if text.startswith(FINAL_RESPONSE_START_TAG) and text.endswith(FINAL_RESPONSE_END_TAG):
                    final_response = text[len(FINAL_RESPONSE_START_TAG) : -len(FINAL_RESPONSE_END_TAG)]
                    logger.info(f"[AGENT FINAL RESPONSE] {final_response}")
                    self.metrics.agent_final_response.append(final_response)
                    self.metrics.agent_final_response_time.append(timestamp)
                    logger.info("[AGENT] Final response saved")
                if text.startswith(EXIT_MESSAGE_START_TAG) and text.endswith(EXIT_MESSAGE_END_TAG):
                    exit_message = text[len(EXIT_MESSAGE_START_TAG) : -len(EXIT_MESSAGE_END_TAG)]
                    logger.info(
                        f"[AGENT] Exit message received; settling "
                        f"{self.exit_settle_delay}s before stopping threads. "
                        f"Exit message: {exit_message}"
                    )
                    self.stop_reason = STOP_REASON_EXIT
                    # Grace window: let the bots commit any in-flight
                    # tool-call cycle (assistant + tool messages) to
                    # their LLM contexts and ``shared_state["actions"]``
                    # before the receive loops exit and the bridge pulls
                    # ``get_context_history`` / ``get_scenario_summary``.
                    # Without this, the bridge can race the commit and
                    # snapshot stale state (most visibly: ``EndConversationTool``
                    # missing from ``bot_logs_agent/llm_context.json`` even
                    # though it fired — see ``SendExitMessageTool._execute``
                    # for the paired source-side fix).
                    await asyncio.sleep(self.exit_settle_delay)
                    self.stop_event.set()
                    self.metrics.end_time = datetime.now()

    def _save_final_response(self):
        """Save the agent's final response to a JSON file under the output directory.

        Two sources, in priority order:
          1. **Pull** (``self.scenario_summary["actions"]``) — the bridge-pulled
             auto-aggregated action list. Used when the bot registered the
             ``get_scenario_summary`` action (post-commit-3 bots) and returned
             a non-empty actions list.
          2. **Push** (``self.metrics.agent_final_response``) — ``<final_response>``
             text messages captured during the conversation. Used by domains that
             still have an LLM-callable summary tool (restaurant / customer_service
             / qa) or as a fallback when pull returned empty.

        Output is always list-wrapped (``[{"actions": ...}]``) for shape compat
        with the existing strict comparator and downstream consumers.
        """
        if not self.output_dir:
            return

        # Pull path
        if self.scenario_summary and self.scenario_summary.get("actions") is not None:
            # Default each pulled action's ``side`` to "agent", but PRESERVE
            # any existing ``side`` key on the entry. The merge step in
            # ``run_scenario`` already stamps ``side="user"`` on the
            # user-bot-pulled actions before they land in
            # ``scenario_summary["actions"]``; we must not overwrite that.
            # The dict-merge ``{"side": "agent", **a}`` puts the
            # default first so any ``side`` already in ``a`` wins (later
            # keys override earlier ones in dict literal expansion).
            actions = [{"side": "agent", **a} for a in self.scenario_summary.get("actions", [])]
            results = [{"actions": actions}]
            source = "pull"
        else:
            # Push fallback
            results = []
            for final_response in self.metrics.agent_final_response:
                try:
                    response_obj = json.loads(final_response)
                except (json.JSONDecodeError, TypeError):
                    response_obj = {"message": final_response}
                results.append(response_obj)
            source = "push"

        output_path = Path(self.output_dir) / self.final_response_file
        try:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Final agent response saved (source={source}): {output_path}")
        except Exception as e:
            logger.error(f"Error saving final agent response: {e}")

    def _save_scenario_db(self):
        """Save the post-run scenario DB hash(es) to ``final_scenario_db_hash.txt``.

        Sourced from the bridge-pulled ``scenario_summary["db_hash"]`` (and
        ``["user_db_hash"]`` for telecom). Skipped if no pull happened (legacy
        bots) or the hash is None. Used by the runner's DB-state matching
        when ``scenario.expected_scenario_db`` is set — the runner compares
        the expected-DB hash (computed in-process from its own gold replay)
        against the bot's reported hash.

        Hash-only design: the full DB stays on the bot server. See
        ``create_get_scenario_summary_action`` for the rationale (WebSocket
        frame size limit; tau2's DB is 7 MB while pipecat's default frame
        cap is 1 MB).
        """
        if not self.output_dir:
            return
        if not self.scenario_summary:
            return
        db_hash = self.scenario_summary.get("db_hash")
        user_db_hash = self.scenario_summary.get("user_db_hash")
        if db_hash is None and user_db_hash is None:
            return

        output_path = Path(self.output_dir) / self.final_scenario_db_hash_file
        try:
            lines = []
            if db_hash is not None:
                lines.append(f"db_hash: {db_hash}")
            if user_db_hash is not None:
                lines.append(f"user_db_hash: {user_db_hash}")
            output_path.write_text("\n".join(lines) + "\n")
            logger.info(f"Final scenario DB hash(es) saved: {output_path}")
        except Exception as e:
            logger.error(f"Error saving final scenario DB hash: {e}")

    def _save_seglst(self):
        """Save segLST transcript file with offset-adjusted timestamps."""
        if not self.seglst_file or not self.metrics.segments:
            return

        try:
            session_id = self.scenario_name or "evaluation"
            segments_json = []
            sorted_segments = sorted(self.metrics.segments, key=lambda s: s.start_time)

            for seg in sorted_segments:
                start = seg.start_time + self.turn_start_offset_secs
                end = seg.end_time + self.turn_end_offset_secs
                if end <= start:
                    start = seg.start_time
                    end = seg.end_time

                segments_json.append(
                    {
                        "session_id": session_id,
                        "words": seg.transcript,
                        "speaker": seg.speaker,
                        "start_time": start,
                        "end_time": end,
                    }
                )

            with open(self.seglst_file, "w") as f:
                json.dump(segments_json, f, indent=2)

            logger.info(f"segLST saved: {self.seglst_file} ({len(sorted_segments)} segments)")

        except Exception as e:
            logger.error(f"Error saving segLST: {e}")
            import traceback

            traceback.print_exc()

    async def disconnect(self, print_stats: bool = False):
        """
        Disconnect from both user and agent.

        Args:
            print_stats: If True, print final latency statistics (default: True)
                        Set to False when disconnecting during scenario resets
        """
        if print_stats:
            self.metrics.end_time = datetime.now()

        if self.user_ws:
            await self.user_ws.close()
        if self.agent_ws:
            await self.agent_ws.close()

        logger.info("Disconnected from user and agent")

        # Log final statistics only if requested
        if print_stats:
            latency_stats = self.metrics.get_latency_stats()
            if latency_stats["count"] > 0:
                logger.info("\nFinal Latency Statistics:")
                logger.info(f"  Measurements: {latency_stats['count']}")
                logger.info(f"  Mean: {latency_stats['mean_ms']:.1f}ms")
                logger.info(f"  P50: {latency_stats['p50_ms']:.1f}ms")
                logger.info(f"  P95: {latency_stats['p95_ms']:.1f}ms")
                logger.info(f"  Min: {latency_stats['min_ms']:.1f}ms")
                logger.info(f"  Max: {latency_stats['max_ms']:.1f}ms")

    def get_metrics(self):
        """Get evaluation metrics"""
        duration = 0
        if self.metrics.start_time and self.metrics.end_time:
            duration = (self.metrics.end_time - self.metrics.start_time).total_seconds()

        latency_stats = self.metrics.get_latency_stats()

        return {
            "total_turns": len(self.metrics.turns),
            "duration_seconds": duration,
            "turns": self.metrics.turns,
            "latency_stats": latency_stats,
            "latencies": [
                {
                    "user_transcript": lat.user_transcript,
                    "agent_transcript": lat.agent_transcript,
                    "latency_ms": lat.latency_ms,
                }
                for lat in self.metrics.latencies
            ],
        }

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

"""Model-free functional coverage for runtime components that do real IO/actions."""

import asyncio
import json
import wave
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from nemo_voice_agent.pipecat.frames.action import (
    FinishedPresenceUserActionFrame,
    StartedPresenceUserActionFrame,
    new_uid,
    now_timestamp,
)
from nemo_voice_agent.pipecat.processors.frameworks.rtvi_actions import (
    SharedStateRef,
    TaskRef,
    create_apply_initialization_action,
    create_apply_sync_delta_action,
    create_get_context_history_action,
    create_get_scenario_summary_action,
    create_reset_context_action,
    create_update_system_prompt_action,
    sanitize_context_for_transport,
)
from nemo_voice_agent.pipecat.services.nemo.audio_logger import AudioLogger


pytestmark = pytest.mark.functional


class _Aggregator:
    """Small context aggregator fake used to exercise real RTVI action handlers."""

    def __init__(self, messages=None):
        """Initialize context state and reset counter."""
        self._context = self
        self._messages = list(messages or [])
        self.reset_count = 0
        self.tools = []
        self.has_function_calls_in_progress = False

    def reset(self):
        """Record a reset call."""
        self.reset_count += 1

    def set_messages(self, messages):
        """Replace the fake context messages."""
        self._messages = list(messages)

    def get_messages(self):
        """Return the fake context messages."""
        return self._messages


class _Service:
    """Resettable service fake used by context reset/update actions."""

    def __init__(self):
        """Initialize reset count."""
        self.reset_count = 0

    def reset(self):
        """Record one service reset."""
        self.reset_count += 1


class _Task:
    """PipelineTask fake that captures queued frames."""

    def __init__(self):
        """Initialize frame capture."""
        self.frames = []

    async def queue_frames(self, frames):
        """Capture queued frames."""
        self.frames.extend(frames)


def test_action_frames_create_unique_ids_and_timezone_aware_timestamps():
    """Action frame dataclasses create unique IDs and UTC-aware lifecycle timestamps."""
    action_id = new_uid()
    started = StartedPresenceUserActionFrame(action_id=action_id, user_id="user-1")
    finished = FinishedPresenceUserActionFrame(action_id=action_id, user_id="user-1", is_success=False)

    assert action_id != new_uid()
    assert now_timestamp().tzinfo is UTC
    assert started.action_id == action_id
    assert started.user_id == "user-1"
    assert started.action_started_at.tzinfo is UTC
    assert finished.action_id == action_id
    assert finished.is_success is False
    assert finished.action_finished_at.tzinfo is UTC


def test_audio_logger_writes_user_agent_and_stereo_artifacts(tmp_path):
    """AudioLogger writes real WAV/JSON artifacts without requiring model inference."""
    logger = AudioLogger(
        log_dir=tmp_path,
        session_id="functional",
        user_audio_sample_rate=16000,
        pre_roll_time_sec=0.0,
        round_precision=3,
    )
    base_time = datetime.now()
    logger.first_audio_timestamp = base_time
    logger.increment_turn_index("user")

    user_audio = np.array([0, 1000, -1000, 0] * 200, dtype=np.int16).tobytes()
    logger.append_continuous_user_audio(user_audio)
    staged = logger.stage_user_audio(
        timestamp_now=base_time,
        transcription="hello runtime",
        is_first_frame=True,
        additional_metadata={"source": "functional"},
    )
    logger.save_user_audio()

    logger.increment_turn_index("agent")
    logger._agent_turn_start_time = 0.05
    agent_audio = np.array([0.0, 0.25, -0.25, 0.0], dtype=np.float32)
    logged = logger.log_agent_audio(
        agent_audio,
        text="hello back",
        sample_rate=16000,
        additional_metadata={"voice": "fake"},
    )
    logger.set_agent_cutoff_time(0.051)
    logger.finalize_session()

    assert staged is not None
    assert logged is not None
    assert (tmp_path / "functional" / "user" / Path(staged["audio_file"]).name).exists()
    assert (tmp_path / "functional" / "agent" / Path(logged["metadata_file"]).name).exists()
    metadata = json.loads((tmp_path / "functional" / "session_metadata.json").read_text())
    assert metadata["total_user_entries"] == 1
    assert metadata["total_agent_segments"] == 1
    assert metadata["user_entries"][0]["transcription"] == "hello runtime"
    assert metadata["agent_entries"][0][0]["text"] == "hello back"
    assert metadata["agent_entries"][0][0]["cutoff_time"] == 0.051
    with wave.open(str(tmp_path / "functional" / "conversation_stereo.wav"), "rb") as wav_file:
        assert wav_file.getnchannels() == 2
        assert wav_file.getframerate() == 16000


def test_rtvi_context_actions_reset_update_history_and_summary():
    """RTVI context actions run real handlers against fake aggregators and shared state."""

    async def _run():
        """Execute the async RTVI action handlers."""
        task = _Task()
        task_ref = TaskRef(task=task, running=True)
        original_messages = [{"role": "system", "content": "original"}]
        user_aggregator = _Aggregator(original_messages)
        assistant_aggregator = _Aggregator()
        service = _Service()
        shared_state_ref = SharedStateRef({"actions": [{"name": "existing"}], "db": {"customer": {"status": "old"}}})
        registered = []

        def _tool_factory(tool_name, **kwargs):
            """Build a simple fake schema tool record."""
            return {"tool_name": tool_name, "kwargs": kwargs}

        def _register_schema_tools(**kwargs):
            """Capture schema registration inputs."""
            registered.append(kwargs)

        reset = create_reset_context_action(
            task_ref,
            user_aggregator,
            assistant_aggregator,
            original_messages,
            [service],
        )
        update = create_update_system_prompt_action(
            task_ref,
            user_aggregator,
            assistant_aggregator,
            original_messages,
            [service],
            system_role="system",
            system_prompt_suffix="suffix",
            enable_tool_calling=True,
            llm=object(),
            context=object(),
            rtvi=object(),
            tool_factory=_tool_factory,
            register_schema_tools=_register_schema_tools,
            shared_state_ref=shared_state_ref,
        )
        history = create_get_context_history_action(task_ref, assistant_aggregator)
        summary = create_get_scenario_summary_action(task_ref, shared_state_ref)

        assert await reset.handler(None, "context", {}) is True
        assert await update.handler(
            None,
            "context",
            {
                "prompt": "new prompt",
                "tools": json.dumps({"ToolA": {"arg": "value"}}),
                "tool_domain": "demo",
            },
        )
        assistant_aggregator.set_messages(
            [
                {
                    "role": "user",
                    "content": [{"type": "input_audio", "input_audio": {"data": b"raw", "format": "wav"}}],
                }
            ]
        )
        history_result = await history.handler(None, "context", {})
        summary_result = await summary.handler(None, "context", {"include_db": True})

        return (
            task,
            user_aggregator,
            assistant_aggregator,
            service,
            shared_state_ref,
            registered,
            history_result,
            summary_result,
        )

    (
        task,
        user_aggregator,
        assistant_aggregator,
        service,
        shared_state_ref,
        registered,
        history_result,
        summary_result,
    ) = asyncio.run(_run())

    assert len(task.frames) >= 2
    assert user_aggregator._messages == [{"role": "system", "content": "new prompt\nsuffix"}]
    assert assistant_aggregator._messages[0]["role"] == "user"
    assert service.reset_count == 2
    assert shared_state_ref.state["__tool_domain__"] == "demo"
    assert registered[0]["tools"][0]["tool_name"] == "ToolA"
    assert "<audio>" in history_result["context"]
    assert summary_result["actions"] == []
    assert summary_result["db"] is None


def test_rtvi_initialization_and_sync_delta_actions_mutate_shared_state():
    """Initialization and sync-delta handlers mutate shared state without model services."""

    async def _run():
        """Execute initialization and sync handlers."""
        shared_state_ref = SharedStateRef({})
        init_action = create_apply_initialization_action(shared_state_ref)
        sync_action = create_apply_sync_delta_action(shared_state_ref)

        init_result = await init_action.handler(
            None,
            "context",
            {
                "domain": "default",
                "shared_state_init": json.dumps({"db": {"customer": {"status": "old"}}}),
                "actions": [],
            },
        )
        sync_result = await sync_action.handler(
            None,
            "context",
            {
                "domain": "default",
                "delta": {"customer.status": "new", "customer.plan": "premium"},
            },
        )
        bad_sync_result = await sync_action.handler(None, "context", {"domain": "default", "delta": "bad"})
        return shared_state_ref, init_result, sync_result, bad_sync_result

    shared_state_ref, init_result, sync_result, bad_sync_result = asyncio.run(_run())

    assert init_result == {"success": True, "errors": []}
    assert sync_result == {"success": True, "errors": []}
    assert shared_state_ref.state["db"]["customer"] == {"status": "new", "plan": "premium"}
    assert bad_sync_result["success"] is False


def test_sanitize_context_for_transport_replaces_nested_media_payloads():
    """Context sanitization strips bytes and data URIs while preserving metadata."""
    payload = {
        "audio": {"data": b"abc", "transcript": "hello"},
        "image_url": {"url": "data:image/png;base64,abc", "detail": "low"},
        "file": {"file_data": "abc", "filename": "report.pdf"},
        "plain": {"data": "kept"},
    }

    sanitized = sanitize_context_for_transport(payload)

    assert sanitized == {
        "audio": {"data": "<audio>", "transcript": "hello"},
        "image_url": {"url": "<image>", "detail": "low"},
        "file": {"file_data": "<file>", "filename": "report.pdf"},
        "plain": {"data": "kept"},
    }

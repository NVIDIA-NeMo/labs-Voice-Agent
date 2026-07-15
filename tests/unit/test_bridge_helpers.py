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

"""Unit tests for side-effect-light helpers in the evaluation bridge."""

import asyncio
import json
import queue
import wave
from datetime import datetime
from types import SimpleNamespace

import numpy as np

from nemo_voice_agent.evaluation.bridge import (
    RTVI_BOT_SERVER_MESSAGE,
    RTVI_BOT_STARTED_SPEAKING,
    RTVI_BOT_STOPPED_SPEAKING,
    RTVI_BOT_TTS_TEXT,
    EvaluationMetrics,
    ResponseLatency,
    SegmentEntry,
    VoiceAgentEvaluationBridge,
)


def _bridge(tmp_path):
    """Build a bridge configured for local helper tests without connecting sockets."""
    return VoiceAgentEvaluationBridge(
        "ws://user.invalid",
        "ws://agent.invalid",
        output_dir=str(tmp_path),
        scenario_name="unit_bridge",
        user_input_sample_rate=16000,
        agent_input_sample_rate=16000,
        output_sample_rate=16000,
        log_level="ERROR",
    )


class _FakeSerializer:
    """Serializer that stores MessageFrame JSON data directly on the fake websocket."""

    async def serialize(self, frame):
        """Return the frame data string without protobuf encoding."""
        return getattr(frame, "data", frame)

    async def deserialize(self, message):
        """Return an object with the message payload expected by bridge helpers."""
        if isinstance(message, bytes):
            message = json.loads(message.decode())
        return SimpleNamespace(message=message)


class _FakeWebSocket:
    """Async fake websocket with scripted recv messages and captured sends."""

    def __init__(self, messages=()):
        """Initialize the fake with a finite receive queue."""
        self.messages = list(messages)
        self.sent = []

    async def send(self, message):
        """Capture a serialized outbound message."""
        self.sent.append(message)

    async def recv(self):
        """Return the next scripted inbound message or time out."""
        if not self.messages:
            raise asyncio.TimeoutError()
        item = self.messages.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeAudioStream:
    """Minimal audio stream for exercising bridge send-loop behavior."""

    output_sample_rate = 16000

    def __init__(self):
        """Initialize queued input and fixed output chunks."""
        self.put_chunks = []
        self.outputs = [(b"\x01\x02", True)]

    async def put(self, audio_chunk):
        """Record chunks drained from the source queue."""
        self.put_chunks.append(audio_chunk)

    async def get_nowait(self):
        """Return one audio output and then ask the bridge loop to stop."""
        return self.outputs.pop(0) if self.outputs else (b"\x00\x00", False)


class _FakeShadowTool:
    """Shadow sync tool that records invoke calls and can optionally fail."""

    def __init__(self, *, fail=False):
        """Configure whether invoke should raise."""
        self.fail = fail
        self.calls = []

    def invoke(self, **kwargs):
        """Record sync replay arguments or raise a configured error."""
        if self.fail:
            raise RuntimeError("shadow failed")
        self.calls.append(kwargs)


class _SyncScenario:
    """Scenario-like object that opts into bridge cross-side sync."""

    name = "sync__scenario"
    domain = "sync_domain"
    db = {"agent": {"value": 1}}
    user_db = {"user": {"value": 2}}
    initialization_actions = []

    def __init__(self, *, agent_delta=None, user_delta=None, build_fails=False, sync_fails=False):
        """Configure sync deltas and failure modes."""
        self.agent_delta = agent_delta or {}
        self.user_delta = user_delta or {}
        self.build_fails = build_fails
        self.sync_fails = sync_fails
        self.tool = _FakeShadowTool()

    def _build_tool_map(self, state):
        """Build a fake shadow tool map from the supplied shadow state."""
        if self.build_fails:
            raise RuntimeError("no tool map")
        self.shadow_state_seen = state
        return {"sync_tool": self.tool}

    def sync_state(self, agent_db, user_db):
        """Return configured per-side deltas after recording the supplied DBs."""
        if self.sync_fails:
            raise RuntimeError("sync failed")
        self.synced_agent_db = agent_db
        self.synced_user_db = user_db
        return {"agent": self.agent_delta, "user": self.user_delta}


def test_evaluation_metrics_latency_stats_and_reset():
    """Latency statistics are computed from recorded latencies and reset clears scenario state."""
    metrics = EvaluationMetrics(
        turns=[{"role": "user", "text": "hi"}],
        latencies=[
            ResponseLatency(0.0, 0.1, 100.0),
            ResponseLatency(0.0, 0.2, 200.0),
            ResponseLatency(0.0, 0.3, 300.0),
        ],
        user_current_transcript="hello",
        agent_current_transcript="agent",
        current_user_segment=SegmentEntry(0.0, 0.0, "user", ""),
        current_agent_segment=SegmentEntry(0.0, 0.0, "agent", ""),
    )

    stats = metrics.get_latency_stats()
    metrics.reset()

    assert stats == {
        "count": 3,
        "mean_ms": 200.0,
        "p50_ms": 200.0,
        "p95_ms": 300.0,
        "min_ms": 100.0,
        "max_ms": 300.0,
    }
    assert metrics.turns == []
    assert metrics.latencies == []
    assert metrics.user_current_transcript == ""
    assert metrics.current_user_segment is None
    assert metrics.current_agent_segment is None


def test_finalize_speaker_turn_records_transcript_and_clears_state(tmp_path):
    """Finalizing a speaker turn stores a segment with relative timing and clears buffers."""
    bridge = _bridge(tmp_path)
    bridge.metrics.thread_start_timestamp = 10.0
    bridge.metrics.current_user_segment = SegmentEntry(0.5, 0.5, "user", "")
    bridge.metrics.user_current_transcript = " hello "

    segment = bridge._finalize_speaker_turn("user", 12.0)

    assert segment == SegmentEntry(0.5, 2.0, "user", "hello")
    assert bridge.metrics.segments == [segment]
    assert bridge.metrics.current_user_segment is None
    assert bridge.metrics.user_current_transcript == ""


def test_finalize_speaker_turn_marks_interrupted_empty_transcript(tmp_path):
    """A turn with no accumulated TTS text is marked as interrupted instead of blank."""
    bridge = _bridge(tmp_path)
    bridge.metrics.thread_start_timestamp = 0.0
    bridge.metrics.current_agent_segment = SegmentEntry(1.0, 1.0, "agent", "")

    segment = bridge._finalize_speaker_turn("agent", 2.0)

    assert segment.transcript == "[INTERRUPTED]"
    assert bridge.metrics.agent_current_transcript == ""


def test_build_conversation_log_computes_latency_between_user_and_agent(tmp_path):
    """Conversation logs are sorted and agent latency is computed from the prior user turn."""
    bridge = _bridge(tmp_path)
    bridge.turn_start_offset_secs = 0.0
    bridge.turn_end_offset_secs = 0.0
    bridge.metrics.segments = [
        SegmentEntry(start_time=2.0, end_time=3.0, speaker="agent", transcript="response"),
        SegmentEntry(start_time=0.5, end_time=1.0, speaker="user", transcript="question"),
    ]

    bridge._build_conversation_log()

    assert len(bridge.metrics.log_entries) == 2
    assert "USER: question" in bridge.metrics.log_entries[0][1]
    assert "AGENT: response" in bridge.metrics.log_entries[1][1]
    assert "Response latency: 1000.0ms" in bridge.metrics.log_entries[1][1]


def test_log_rtvi_event_accumulates_token_usage(tmp_path):
    """Metrics RTVI events increment prompt/completion token counters for the emitting side."""
    bridge = _bridge(tmp_path)
    event = {
        "type": "metrics",
        "data": {
            "tokens": [
                {"prompt_tokens": 10, "completion_tokens": 3},
                {"prompt_tokens": "2", "completion_tokens": None},
            ],
            "ttfb": [{"processor": "llm", "value": "0.125"}],
        },
    }

    bridge._log_rtvi_event("AGENT", "metrics", event)

    assert bridge.token_usage["agent"] == {"n_calls": 2, "prompt": 12, "completion": 3}
    assert bridge.token_usage["user"] == {"n_calls": 0, "prompt": 0, "completion": 0}


def test_monitor_agent_message_captures_final_response_and_exit(tmp_path):
    """Server messages update final-response buffers and mark clean exit without sockets."""
    bridge = _bridge(tmp_path)
    bridge.exit_settle_delay = 0.0

    async def _run():
        """Feed final-response and exit messages through the agent monitor."""
        await bridge._monitor_agent_message(
            SimpleNamespace(
                message={"type": RTVI_BOT_SERVER_MESSAGE, "data": {"text": "<final_response>{}</final_response>"}}
            )
        )
        await bridge._monitor_agent_message(
            SimpleNamespace(message={"type": RTVI_BOT_SERVER_MESSAGE, "data": {"text": "<exit>done</exit>"}})
        )

    import asyncio

    asyncio.run(_run())

    assert bridge.metrics.agent_final_response == ["{}"]
    assert bridge.stop_reason == "[EXIT]"
    assert bridge.stop_event.is_set()


def test_monitor_user_and_agent_messages_build_turns_and_latency(tmp_path):
    """Fake RTVI frames exercise monitor turn lifecycle and first-audio latency capture."""
    bridge = _bridge(tmp_path)
    bridge.metrics.thread_start_timestamp = 0.0

    async def _run():
        """Feed user and agent monitor events in the order produced by a short exchange."""
        await bridge._monitor_user_message(SimpleNamespace(message={"type": RTVI_BOT_STARTED_SPEAKING, "data": {}}))
        await bridge._monitor_user_message(
            SimpleNamespace(message={"type": RTVI_BOT_TTS_TEXT, "data": {"text": "hello"}})
        )
        await bridge._monitor_user_message(SimpleNamespace(message={"type": RTVI_BOT_STOPPED_SPEAKING, "data": {}}))
        await bridge._monitor_agent_message(SimpleNamespace(audio=b"\x01\x02"))
        await bridge._monitor_agent_message(SimpleNamespace(message={"type": RTVI_BOT_STARTED_SPEAKING, "data": {}}))
        await bridge._monitor_agent_message(
            SimpleNamespace(message=json.dumps({"type": RTVI_BOT_TTS_TEXT, "data": {"text": "hi"}}))
        )
        await bridge._monitor_agent_message(SimpleNamespace(message={"type": RTVI_BOT_STOPPED_SPEAKING, "data": {}}))

    import asyncio

    asyncio.run(_run())

    assert bridge.metrics.turns[0]["role"] == "user"
    assert bridge.metrics.turns[0]["text"] == "hello"
    assert bridge.metrics.turns[1]["role"] == "agent"
    assert bridge.metrics.turns[1]["text"] == "hi"
    assert len(bridge.metrics.latencies) == 1
    assert bridge.metrics.latencies[0].user_transcript == "hello"
    assert bridge.metrics.latencies[0].agent_transcript == "hi"


def test_monitor_user_message_routes_action_applied_to_sync(tmp_path):
    """User-side action-applied server messages are forwarded to cross-side sync propagation."""
    bridge = _bridge(tmp_path)
    seen = []

    async def _fake_propagate(action, source_side):
        """Capture sync propagation arguments."""
        seen.append((action, source_side))

    bridge._propagate_cross_side_sync = _fake_propagate

    async def _run():
        """Feed an action-applied server message through the user monitor."""
        await bridge._monitor_user_message(
            SimpleNamespace(
                message={
                    "type": RTVI_BOT_SERVER_MESSAGE,
                    "data": {
                        "type": "action-applied",
                        "action": {"name": "toggle_data"},
                    },
                }
            )
        )

    import asyncio

    asyncio.run(_run())

    assert seen == [({"name": "toggle_data"}, "user")]


def test_wait_for_action_response_accepts_success_dict_and_rejects_bool_result(tmp_path):
    """The generic action waiter treats success dicts as successful responses."""
    bridge = _bridge(tmp_path)
    dict_ws = _FakeWebSocket(
        [
            json.dumps({"data": {"message_type": "ignored", "result": {"success": True}}}),
            json.dumps({"data": {"message_type": "action-response", "result": {"success": True}}}),
        ]
    )
    bool_ws = _FakeWebSocket([json.dumps({"data": {"message_type": "action-response", "result": True}})])

    dict_result = asyncio.run(bridge._wait_for_action_response(dict_ws, timeout=0.01))
    bool_result = asyncio.run(bridge._wait_for_action_response(bool_ws, timeout=0.01))

    assert dict_result is True
    assert bool_result is False


def test_client_ready_prompt_reset_and_send_text_helpers_emit_expected_actions(tmp_path):
    """Handshake, prompt update, reset, and send-text helpers serialize expected RTVI payloads."""
    bridge = _bridge(tmp_path)
    bridge.serializer = _FakeSerializer()
    user_ws = _FakeWebSocket([b'{"type": "bot-ready"}'])
    agent_ws = _FakeWebSocket([b'{"type": "bot-ready"}'])
    bridge.user_ws = user_ws
    bridge.agent_ws = agent_ws

    assert asyncio.run(bridge._send_client_ready(user_ws)) is True
    asyncio.run(bridge.update_user_prompt("user prompt", "{}", auto_reset=True, add_suffix=True, tool_domain="demo"))
    asyncio.run(bridge.update_agent_prompt("agent prompt", "{}", auto_reset=True, tool_domain="demo"))
    asyncio.run(bridge.send_text_to_user("hello user"))
    asyncio.run(bridge.send_text_to_agent("hello agent"))

    user_payloads = [json.loads(m) for m in user_ws.sent]
    agent_payloads = [json.loads(m) for m in agent_ws.sent]

    assert user_payloads[0]["type"] == "client-ready"
    assert user_payloads[1]["data"]["action"] == "update_system_prompt"
    assert user_payloads[1]["data"]["arguments"] == [
        {"name": "prompt", "value": "user prompt"},
        {"name": "tools", "value": "{}"},
        {"name": "add_suffix", "value": True},
        {"name": "tool_domain", "value": "demo"},
    ]
    assert user_payloads[2]["data"]["action"] == "reset"
    assert user_payloads[3]["type"] == "send-text"
    assert user_payloads[3]["data"]["content"] == "hello user"
    assert agent_payloads[0]["data"]["action"] == "update_system_prompt"
    assert agent_payloads[1]["data"]["action"] == "reset"
    assert agent_payloads[2]["type"] == "send-text"
    assert agent_payloads[2]["data"]["content"] == "hello agent"


def test_client_ready_returns_false_on_timeout_and_bad_payload(tmp_path):
    """Client-ready waits only for bot-ready bytes and returns False on timeout or bad JSON."""
    bridge = _bridge(tmp_path)
    bridge.serializer = _FakeSerializer()

    assert asyncio.run(bridge._send_client_ready(_FakeWebSocket([]))) is False
    assert asyncio.run(bridge._send_client_ready(_FakeWebSocket([b"not-json"]))) is False


def test_reset_calls_user_and_agent_reset_and_clears_metrics(tmp_path):
    """reset() sends reset actions to both sockets and clears scenario metrics."""
    bridge = _bridge(tmp_path)
    bridge.serializer = _FakeSerializer()
    bridge.user_ws = _FakeWebSocket()
    bridge.agent_ws = _FakeWebSocket()
    bridge.metrics.turns = [{"role": "user", "text": "old"}]
    bridge.needs_reset = True

    asyncio.run(bridge.reset())

    assert json.loads(bridge.user_ws.sent[0])["data"]["action"] == "reset"
    assert json.loads(bridge.agent_ws.sent[0])["data"]["action"] == "reset"
    assert bridge.metrics.turns == []
    assert bridge.needs_reset is False


def test_wait_for_action_response_returns_false_on_bad_json(tmp_path):
    """Malformed websocket text is caught and converted into a False action result."""
    bridge = _bridge(tmp_path)
    ws = _FakeWebSocket(["not-json"])

    assert asyncio.run(bridge._wait_for_action_response(ws, timeout=0.01)) is False


def test_retrieve_context_history_sends_action_and_returns_result(tmp_path):
    """Context-history retrieval sends the correct RTVI action and returns the action result."""
    bridge = _bridge(tmp_path)
    bridge.serializer = _FakeSerializer()
    ws = _FakeWebSocket(
        [
            {"type": "not-action-response", "data": {}},
            {"type": "action-response", "data": {"result": {"context": [{"role": "assistant"}], "logs": "ok"}}},
        ]
    )

    result = asyncio.run(bridge._retrieve_context_history(ws))
    sent = json.loads(ws.sent[0])

    assert sent["data"]["action"] == "get_context_history"
    assert result == {"context": [{"role": "assistant"}], "logs": "ok"}


def test_retrieve_context_history_returns_empty_without_websocket(tmp_path):
    """A missing websocket short-circuits context-history retrieval."""
    bridge = _bridge(tmp_path)

    assert asyncio.run(bridge._retrieve_context_history(None)) == {}


def test_retrieve_scenario_summary_sends_include_db_argument(tmp_path):
    """Scenario-summary retrieval includes the include_db argument and returns the summary result."""
    bridge = _bridge(tmp_path)
    bridge.serializer = _FakeSerializer()
    ws = _FakeWebSocket(
        [
            {
                "type": "action-response",
                "data": {
                    "result": {
                        "actions": [{"name": "done"}],
                        "db_hash": "hash",
                        "db": {"state": "ok"},
                    }
                },
            }
        ]
    )

    result = asyncio.run(bridge._retrieve_scenario_summary(ws, include_db=True))
    sent = json.loads(ws.sent[0])

    assert sent["data"]["action"] == "get_scenario_summary"
    assert sent["data"]["arguments"] == [{"name": "include_db", "value": True}]
    assert result["actions"] == [{"name": "done"}]
    assert result["db"] == {"state": "ok"}


def test_send_apply_initialization_ignores_unrelated_response_and_raises_on_failure(tmp_path):
    """Initialization sender ID-matches responses and raises when the matching result fails."""
    bridge = _bridge(tmp_path)
    bridge.serializer = _FakeSerializer()
    ws = _FakeWebSocket(
        [
            {"type": "action-response", "id": "old-response", "data": {"result": True}},
            {
                "type": "action-response",
                "id": "placeholder",
                "data": {"result": {"success": False, "errors": ["bad init"]}},
            },
        ]
    )

    async def _run():
        """Patch recv to match the generated action id on the second response."""
        original_recv = ws.recv

        async def _recv():
            """Return scripted responses, replacing placeholder with the outbound action id."""
            message = await original_recv()
            if message.get("id") == "placeholder":
                message["id"] = json.loads(ws.sent[0])["id"]
            return message

        ws.recv = _recv
        await bridge._send_apply_initialization(
            ws,
            "agent",
            "demo",
            {"db_path": "fixture.json"},
            [{"func_name": "seed", "arguments": {}, "side": "agent"}],
        )

    try:
        asyncio.run(_run())
    except RuntimeError as exc:
        assert "bad init" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for failed initialization response")


def test_send_apply_initialization_success_payload(tmp_path):
    """Initialization sender serializes domain, shared_state_init JSON, and side-filtered actions."""
    bridge = _bridge(tmp_path)
    bridge.serializer = _FakeSerializer()
    ws = _FakeWebSocket([{"type": "action-response", "id": "placeholder", "data": {"result": {"success": True}}}])

    async def _run():
        """Patch recv so the fake response matches the generated action id."""
        original_recv = ws.recv

        async def _recv():
            """Replace placeholder id after the send happens."""
            message = await original_recv()
            message["id"] = json.loads(ws.sent[0])["id"]
            return message

        ws.recv = _recv
        await bridge._send_apply_initialization(ws, "user", "demo", {"x": 1}, [{"func_name": "seed"}])

    asyncio.run(_run())
    sent = json.loads(ws.sent[0])

    assert sent["data"]["action"] == "apply_initialization"
    assert sent["data"]["arguments"][0] == {"name": "domain", "value": "demo"}
    assert sent["data"]["arguments"][1] == {"name": "shared_state_init", "value": json.dumps({"x": 1})}
    assert sent["data"]["arguments"][2] == {"name": "actions", "value": [{"func_name": "seed"}]}


def test_apply_initialization_rejects_unknown_sides_before_sending(tmp_path):
    """Bridge-side initialization rejects untranslated upstream side values."""
    bridge = _bridge(tmp_path)
    scenario = SimpleNamespace(domain="demo", initialization_actions=[{"side": "assistant", "func_name": "seed"}])

    try:
        asyncio.run(
            bridge._apply_initialization(
                scenario,
                user_shared_state_init={},
                agent_shared_state_init={},
            )
        )
    except RuntimeError as exc:
        assert "unknown side" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for unknown initialization side")


def test_send_apply_sync_delta_serializes_delta_and_ignores_send_errors(tmp_path):
    """Sync delta sending constructs the RTVI action and swallows closed-connection send failures."""
    bridge = _bridge(tmp_path)
    bridge.serializer = _FakeSerializer()
    ws = _FakeWebSocket()

    asyncio.run(bridge._send_apply_sync_delta(ws, "agent", "demo", {"path": "value"}))
    sent = json.loads(ws.sent[0])

    assert sent["data"]["action"] == "apply_sync_delta"
    assert sent["data"]["arguments"] == [
        {"name": "domain", "value": "demo"},
        {"name": "delta", "value": {"path": "value"}},
    ]

    class _FailingSendWebSocket(_FakeWebSocket):
        """Fake websocket whose send method raises."""

        async def send(self, message):
            """Raise to simulate a closed websocket."""
            raise RuntimeError("closed")

    assert asyncio.run(bridge._send_apply_sync_delta(_FailingSendWebSocket(), "agent", "demo", {"x": 1})) is None


def test_receive_to_queue_deserializes_audio_and_calls_monitor(tmp_path):
    """The receive loop deserializes fake frames, calls the monitor, and queues audio bytes."""
    bridge = _bridge(tmp_path)
    audio_frame = SimpleNamespace(audio=b"\x01\x02")
    text_frame = SimpleNamespace(message={"type": "noop"})
    seen = []

    class _FrameSerializer:
        """Serializer that maps inbound tokens to fake frame objects."""

        async def deserialize(self, message):
            """Return fake frames for known tokens and raise on the error token."""
            if message == "audio":
                return audio_frame
            if message == "text":
                return text_frame
            if message == "bad":
                raise ValueError("bad frame")
            return None

    async def _monitor(frame):
        """Capture monitored frames and stop after audio is seen."""
        seen.append(frame)
        if getattr(frame, "audio", None):
            bridge.stop_event.set()

    bridge.serializer = _FrameSerializer()
    ws = _FakeWebSocket(["bad", "text", "audio"])
    q = queue.Queue()

    asyncio.run(bridge._receive_to_queue(ws, duration=10.0, direction="TEST", queue=q, monitor_func=_monitor))

    assert seen == [text_frame, audio_frame]
    assert q.get_nowait() == b"\x01\x02"


def test_send_audio_stream_drains_queue_sends_frame_and_tracks_chunk(tmp_path):
    """The audio send loop drains queued audio, emits one frame, and records sent bytes."""
    bridge = _bridge(tmp_path)
    bridge.serializer = _FakeSerializer()
    bridge.audio_chunk_in_seconds = 0.001
    stream = _FakeAudioStream()
    ws = _FakeWebSocket()
    source_queue = queue.Queue()
    source_queue.put(b"inbound")
    sent_chunks = []

    async def _run():
        """Run the send loop and stop it after the first outbound frame."""
        original_send = ws.send

        async def _send(message):
            """Capture one send and then stop the bridge loop."""
            await original_send(message)
            bridge.stop_event.set()

        ws.send = _send
        await bridge._send_audio_stream(stream, ws, "TEST", 10, source_queue, sent_chunks)

    asyncio.run(_run())

    assert stream.put_chunks == [b"inbound"]
    assert sent_chunks == [b"\x01\x02"]
    assert len(ws.sent) == 1


def test_propagate_cross_side_sync_replays_tool_and_dispatches_deltas(tmp_path):
    """Cross-side sync replay invokes the shadow tool and dispatches both side deltas."""
    bridge = _bridge(tmp_path)
    bridge.sync_enabled = True
    bridge.sync_lock = asyncio.Lock()
    bridge.shadow_state = {"db": {"agent": 1}, "user_db": {"user": 2}}
    scenario = _SyncScenario(agent_delta={"a": 1}, user_delta={"u": 2})
    tool = _FakeShadowTool()
    bridge.scenario_instance = scenario
    bridge.shadow_tool_map = {"sync_tool": tool}
    calls = []

    async def _fake_send(ws, side_label, domain, delta):
        """Capture sync delta dispatches."""
        calls.append((side_label, domain, delta))

    bridge._send_apply_sync_delta = _fake_send

    asyncio.run(bridge._propagate_cross_side_sync({"name": "sync_tool", "arguments": {"x": 1}}, "agent"))

    assert tool.calls == [{"x": 1}]
    assert scenario.synced_agent_db == {"agent": 1}
    assert scenario.synced_user_db == {"user": 2}
    assert calls == [
        ("agent", "sync_domain", {"a": 1}),
        ("user", "sync_domain", {"u": 2}),
    ]


def test_propagate_cross_side_sync_noops_for_disabled_missing_or_failing_tool(tmp_path):
    """Sync propagation exits cleanly when disabled, tool missing, or shadow replay fails."""
    bridge = _bridge(tmp_path)

    assert asyncio.run(bridge._propagate_cross_side_sync({"name": "sync_tool"}, "agent")) is None

    bridge.sync_enabled = True
    bridge.sync_lock = asyncio.Lock()
    bridge.shadow_state = {"db": {}, "user_db": {}}
    bridge.scenario_instance = _SyncScenario(agent_delta={"a": 1})
    bridge.shadow_tool_map = {}
    assert asyncio.run(bridge._propagate_cross_side_sync({"name": "missing"}, "agent")) is None

    bridge.shadow_tool_map = {"bad": _FakeShadowTool(fail=True)}
    assert asyncio.run(bridge._propagate_cross_side_sync({"name": "bad"}, "agent")) is None


def test_setup_cross_side_sync_enables_shadow_state_and_dispatches_initial_deltas(tmp_path):
    """Sync setup deep-copies DBs, builds tools, enables sync, and sends initial deltas."""
    bridge = _bridge(tmp_path)
    scenario = _SyncScenario(agent_delta={"agent.path": 1}, user_delta={"user.path": 2})
    calls = []

    async def _fake_send(ws, side_label, domain, delta):
        """Capture initial sync delta dispatches."""
        calls.append((side_label, domain, delta))

    bridge._send_apply_sync_delta = _fake_send

    asyncio.run(bridge._setup_cross_side_sync(scenario))

    assert bridge.sync_enabled is True
    assert bridge.scenario_instance is scenario
    assert bridge.shadow_state["db"] == scenario.db
    assert bridge.shadow_state["db"] is not scenario.db
    assert bridge.shadow_state["user_db"] == scenario.user_db
    assert bridge.shadow_tool_map == {"sync_tool": scenario.tool}
    assert calls == [
        ("agent", "sync_domain", {"agent.path": 1}),
        ("user", "sync_domain", {"user.path": 2}),
    ]


def test_setup_cross_side_sync_resets_and_noops_for_default_scenario(tmp_path):
    """A scenario using the base Scenario.sync_state disables stale sync state and returns."""
    from nemo_voice_agent.evaluation.scenarios.classes import Scenario

    bridge = _bridge(tmp_path)
    bridge.sync_enabled = True
    bridge.shadow_state = {"old": True}

    class _DefaultSyncScenario:
        """Scenario-like object that inherits the base no-op sync method."""

        sync_state = Scenario.sync_state

    scenario = _DefaultSyncScenario()

    asyncio.run(bridge._setup_cross_side_sync(scenario))

    assert bridge.sync_enabled is False
    assert bridge.scenario_instance is None
    assert bridge.shadow_state is None
    assert bridge.shadow_tool_map == {}


def test_setup_cross_side_sync_disables_on_tool_map_or_sync_failures(tmp_path):
    """Sync setup leaves sync disabled when tool-map construction fails and tolerates sync_state errors."""
    bridge = _bridge(tmp_path)
    build_fails = _SyncScenario(build_fails=True)

    asyncio.run(bridge._setup_cross_side_sync(build_fails))

    assert bridge.sync_enabled is False
    assert bridge.shadow_state is None

    sync_fails = _SyncScenario(sync_fails=True)
    asyncio.run(bridge._setup_cross_side_sync(sync_fails))

    assert bridge.sync_enabled is True
    assert bridge.shadow_state is not None


def test_save_final_response_prefers_pulled_actions_and_preserves_side(tmp_path):
    """Pulled scenario-summary actions are list-wrapped and keep existing side labels."""
    bridge = _bridge(tmp_path)
    bridge.output_dir = str(tmp_path)
    bridge.scenario_summary = {
        "actions": [
            {"name": "agent_action"},
            {"side": "user", "name": "user_action"},
        ]
    }

    bridge._save_final_response()

    saved = json.loads((tmp_path / bridge.final_response_file).read_text())
    assert saved == [
        {
            "actions": [
                {"side": "agent", "name": "agent_action"},
                {"side": "user", "name": "user_action"},
            ]
        }
    ]


def test_save_final_response_falls_back_to_pushed_text(tmp_path):
    """Pushed final responses are parsed as JSON when possible and otherwise stored as messages."""
    bridge = _bridge(tmp_path)
    bridge.output_dir = str(tmp_path)
    bridge.scenario_summary = {}
    bridge.metrics.agent_final_response = ['{"ok": true}', "plain text"]

    bridge._save_final_response()

    saved = json.loads((tmp_path / bridge.final_response_file).read_text())
    assert saved == [{"ok": True}, {"message": "plain text"}]


def test_save_scenario_db_writes_available_hashes(tmp_path):
    """Scenario DB hash artifacts include whichever agent/user hashes are available."""
    bridge = _bridge(tmp_path)
    bridge.output_dir = str(tmp_path)
    bridge.scenario_summary = {"db_hash": "abc", "user_db_hash": "def"}

    bridge._save_scenario_db()

    assert (tmp_path / bridge.final_scenario_db_hash_file).read_text() == "db_hash: abc\nuser_db_hash: def\n"


def test_save_seglst_applies_offsets_and_writes_json(tmp_path):
    """segLST output is sorted and offset-adjusted while preserving segment metadata."""
    bridge = _bridge(tmp_path)
    bridge.seglst_file = str(tmp_path / "segments.json")
    bridge.turn_start_offset_secs = 0.1
    bridge.turn_end_offset_secs = 0.2
    bridge.metrics.segments = [
        SegmentEntry(start_time=1.0, end_time=2.0, speaker="agent", transcript="hello"),
    ]

    bridge._save_seglst()

    saved = json.loads((tmp_path / "segments.json").read_text())
    assert saved == [
        {
            "session_id": "unit_bridge",
            "words": "hello",
            "speaker": "agent",
            "start_time": 1.1,
            "end_time": 2.2,
        }
    ]


def test_save_audio_log_writes_stereo_wav_with_padding(tmp_path):
    """Bridge audio saving pads the shorter side and writes an interleaved stereo WAV."""
    bridge = _bridge(tmp_path)
    bridge.bridge_audio_file = str(tmp_path / "bridge.wav")
    bridge.sent_to_agent_chunks = [np.array([1, 2], dtype=np.int16).tobytes()]
    bridge.sent_to_user_chunks = [np.array([3], dtype=np.int16).tobytes()]

    bridge._save_audio_log()

    with wave.open(str(tmp_path / "bridge.wav"), "rb") as wav_file:
        assert wav_file.getnchannels() == 2
        assert wav_file.getframerate() == 16000
        data = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
    assert data.tolist() == [1, 3, 2, 0]


def test_save_bot_server_history_parses_string_and_writes_role_file(tmp_path):
    """Bot history saving accepts stringified JSON context and writes role-specific files."""
    bridge = _bridge(tmp_path)

    bridge._save_bot_server_history(tmp_path, {"context": json.dumps([{"role": "assistant"}])}, role="agent")

    assert json.loads((tmp_path / "llm_context_agent.json").read_text()) == [{"role": "assistant"}]


def test_save_user_agent_history_writes_both_contexts(tmp_path):
    """User and agent history helpers write separate bot log directories."""
    bridge = _bridge(tmp_path)
    bridge.output_dir = str(tmp_path)
    bridge.user_context_history = {"context": "[{'role': 'user'}]"}
    bridge.agent_context_history = {"context": "[{'role': 'assistant'}]"}

    bridge._save_user_agent_history()

    assert json.loads((tmp_path / "bot_logs_user" / "llm_context.json").read_text()) == [{"role": "user"}]
    assert json.loads((tmp_path / "bot_logs_agent" / "llm_context.json").read_text()) == [{"role": "assistant"}]


def test_get_metrics_serializes_latency_records(tmp_path):
    """get_metrics returns JSON-friendly turn and latency dictionaries."""
    bridge = _bridge(tmp_path)
    bridge.metrics.start_time = datetime.fromisoformat("2026-01-01T00:00:00")
    bridge.metrics.end_time = datetime.fromisoformat("2026-01-01T00:00:02")
    bridge.metrics.turns = [{"role": "agent", "text": "hello"}]
    bridge.metrics.latencies = [ResponseLatency(0.0, 0.25, 250.0, "hi", "hello")]

    metrics = bridge.get_metrics()

    assert metrics["total_turns"] == 1
    assert metrics["duration_seconds"] == 2.0
    assert metrics["latency_stats"]["mean_ms"] == 250.0
    assert metrics["latencies"] == [{"user_transcript": "hi", "agent_transcript": "hello", "latency_ms": 250.0}]

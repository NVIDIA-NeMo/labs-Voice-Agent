# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mocked coverage for the bridge-owned websocket event loops."""

from types import SimpleNamespace

from nemo_voice_agent.evaluation import bridge as bridge_module
from nemo_voice_agent.evaluation.bridge import VoiceAgentEvaluationBridge


def _bridge(tmp_path):
    return VoiceAgentEvaluationBridge(
        "ws://user.invalid",
        "ws://agent.invalid",
        output_dir=str(tmp_path),
        scenario_name="threads",
        user_input_sample_rate=16000,
        agent_input_sample_rate=16000,
        output_sample_rate=16000,
        log_level="ERROR",
    )


class _Connection:
    def __init__(self, socket):
        self.socket = socket

    async def __aenter__(self):
        return self.socket

    async def __aexit__(self, *args):
        return None


class _Stream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


async def _noop(*args, **kwargs):
    return None


def test_user_and_agent_websocket_threads_complete_with_mocked_io(monkeypatch, tmp_path):
    bridge = _bridge(tmp_path)
    bridge.include_db_in_summary = True
    calls = []
    sockets = []

    def connect(url, ping_timeout=None):
        socket = SimpleNamespace(url=url)
        sockets.append(socket)
        return _Connection(socket)

    monkeypatch.setattr(bridge_module, "websockets", SimpleNamespace(connect=connect))
    monkeypatch.setattr(bridge_module, "AudioStream", _Stream)
    monkeypatch.setattr(bridge_module.asyncio, "sleep", _noop)
    bridge._send_client_ready = _noop
    bridge._receive_user_to_queue = _noop
    bridge._send_agent_to_user = _noop
    bridge._send_user_to_agent = _noop
    bridge._receive_agent_to_queue = _noop

    async def context(socket):
        calls.append(("context", socket.url))
        return {"url": socket.url}

    async def summary(socket, include_db=False):
        calls.append(("summary", socket.url, include_db))
        return {"actions": [], "db_hash": socket.url}

    async def kickoff(text):
        calls.append(("kickoff", text))

    bridge._retrieve_context_history = context
    bridge._retrieve_scenario_summary = summary
    bridge.send_text_to_agent = kickoff

    bridge.user_websocket_thread(duration=1)
    bridge.agent_websocket_thread(duration=1)
    assert bridge.user_context_history == {"url": bridge.user_url}
    assert bridge.agent_context_history == {"url": bridge.agent_url}
    assert bridge.user_scenario_summary["db_hash"] == bridge.user_url
    assert bridge.scenario_summary["db_hash"] == bridge.agent_url
    assert ("kickoff", "Hello") in calls and len(sockets) == 2


def test_websocket_threads_contain_connection_failures(monkeypatch, tmp_path):
    bridge = _bridge(tmp_path)

    class BrokenConnection:
        async def __aenter__(self):
            raise RuntimeError("offline")

        async def __aexit__(self, *args):
            return None

    monkeypatch.setattr(
        bridge_module, "websockets", SimpleNamespace(connect=lambda *args, **kwargs: BrokenConnection())
    )
    bridge.user_websocket_thread(duration=0)
    bridge.agent_websocket_thread(duration=0)
    assert bridge.user_ws is None and bridge.agent_ws is None

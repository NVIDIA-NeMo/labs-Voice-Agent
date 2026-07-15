# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for bridge scenario preparation and run orchestration."""

import asyncio
from types import SimpleNamespace

from nemo_voice_agent.evaluation import bridge as bridge_module
from nemo_voice_agent.evaluation.bridge import VoiceAgentEvaluationBridge


def _bridge(tmp_path):
    return VoiceAgentEvaluationBridge(
        "ws://user.invalid",
        "ws://agent.invalid",
        output_dir=str(tmp_path),
        scenario_name="coverage",
        user_input_sample_rate=16000,
        agent_input_sample_rate=16000,
        output_sample_rate=16000,
        log_level="ERROR",
    )


async def _record(calls, name, *args, **kwargs):
    calls.append((name, args, kwargs))


def test_prepare_for_scenario_runs_full_lifecycle(tmp_path):
    bridge = _bridge(tmp_path)
    calls = []
    bridge.init_output_dir = lambda *args, **kwargs: calls.append(("init-output", args, kwargs))
    bridge.connect = lambda: _record(calls, "connect")
    bridge.disconnect = lambda **kwargs: _record(calls, "disconnect", **kwargs)
    bridge.update_user_prompt = lambda **kwargs: _record(calls, "user-prompt", **kwargs)
    bridge.update_agent_prompt = lambda **kwargs: _record(calls, "agent-prompt", **kwargs)
    bridge._apply_initialization = lambda *args, **kwargs: _record(calls, "initialization", *args, **kwargs)
    bridge._setup_cross_side_sync = lambda scenario: _record(calls, "sync", scenario)

    class Scenario:
        name = "sample"
        domain = "retail"
        db_state_assertions = ["check"]
        noise_config = {"random_white_noise": False}

        @staticmethod
        def setup_shared_state(state, side):
            state["side"] = side

        get_user_prompt = staticmethod(lambda: "user")
        get_agent_prompt = staticmethod(lambda: "agent")
        get_user_tools = staticmethod(lambda: ["user-tool"])
        get_agent_tools = staticmethod(lambda: ["agent-tool"])

    asyncio.run(bridge.prepare_for_scenario(Scenario(), str(tmp_path / "scenario"), log_level="INFO"))
    assert bridge.bridge_ready and bridge.include_db_in_summary
    names = [call[0] for call in calls]
    assert names == [
        "init-output",
        "connect",
        "user-prompt",
        "agent-prompt",
        "initialization",
        "sync",
        "disconnect",
    ]
    initialization = calls[4][2]
    assert initialization["user_shared_state_init"] == {"side": "user"}
    assert initialization["agent_shared_state_init"] == {"side": "agent"}


def test_run_scenario_coordinates_threads_merges_summary_and_saves(monkeypatch, tmp_path):
    bridge = _bridge(tmp_path)
    bridge.bridge_ready = True
    bridge.needs_reset = False
    bridge.include_db_in_summary = True
    calls = []

    class Thread:
        def __init__(self, target, args, name):
            self.name = name

        def start(self):
            calls.append(("start", self.name))
            if self.name.startswith("User"):
                bridge.user_scenario_summary = {
                    "db_hash": "user-hash",
                    "db": {"user": 1},
                    "actions": [{"name": "user-action"}],
                }
            else:
                bridge.scenario_summary = {
                    "db_hash": "agent-hash",
                    "actions": [{"name": "agent-action"}],
                }

        def join(self):
            calls.append(("join", self.name))

    monkeypatch.setattr(bridge_module, "threading", SimpleNamespace(Thread=Thread))
    for name in (
        "_save_final_response",
        "_save_scenario_db",
        "_save_conversation_log",
        "_save_audio_log",
        "_save_seglst",
        "_save_user_agent_history",
    ):
        setattr(bridge, name, lambda name=name: calls.append((name,)))
    bridge._finalize_speaker_turn = lambda speaker, timestamp: calls.append(("finalize", speaker))

    asyncio.run(bridge.run_scenario(duration=1))
    assert bridge.needs_reset and not bridge.bridge_ready
    assert bridge.scenario_summary["user_db_hash"] == "user-hash"
    assert bridge.scenario_summary["user_db"] == {"user": 1}
    assert bridge.scenario_summary["actions"] == [
        {"name": "agent-action"},
        {"name": "user-action", "side": "user"},
    ]
    assert bridge.metrics.end_time is not None
    assert ("finalize", "user") in calls and ("finalize", "agent") in calls


def test_run_scenario_requires_ready_and_reset_state(tmp_path):
    bridge = _bridge(tmp_path)
    bridge.bridge_ready = False
    try:
        asyncio.run(bridge.run_scenario(1))
    except RuntimeError as error:
        assert "not ready" in str(error)
    bridge.bridge_ready = True
    bridge.needs_reset = True
    try:
        asyncio.run(bridge.run_scenario(1))
    except RuntimeError as error:
        assert "needs reset" in str(error)

    bridge.metrics.thread_start_timestamp = None
    assert bridge._get_relative_time(10.0) == 0.0
    bridge.metrics.thread_start_timestamp = 4.0
    assert bridge._get_relative_time(10.0) == 6.0

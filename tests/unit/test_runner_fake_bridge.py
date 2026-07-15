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

"""Lightweight functional-style tests for the evaluation runner using a fake bridge."""

import asyncio
import json

from nemo_voice_agent.evaluation import runner as runner_module
from nemo_voice_agent.evaluation.bridge import STOP_REASON_EXIT
from nemo_voice_agent.evaluation.db_hash import get_dict_hash
from nemo_voice_agent.evaluation.scenarios.classes import (
    Actions,
    Persona,
    Resources,
    Scenario,
    SuccessSignal,
    Task,
)


class _FakeLogger:
    """Small logger shim that captures runner info messages without writing a log file."""

    def __init__(self):
        """Create an empty log capture."""
        self.messages = []

    def info(self, message):
        """Capture one info message emitted by the runner."""
        self.messages.append(str(message))


class _CleanExitScenario(Scenario):
    """Minimal scenario scored only by clean exit for fake-bridge runner coverage."""

    name = "fake_domain__clean_exit"
    domain = "fake_domain"
    max_duration = 1
    success_signals = (SuccessSignal.CLEAN_EXIT,)

    @property
    def user_persona(self):
        """Return the user persona required by Scenario.save."""
        return Persona(role="user", name="User", background="background", personality="direct")

    @property
    def agent_persona(self):
        """Return the agent persona required by Scenario.save."""
        return Persona(role="agent", name="Agent", background="background", personality="helpful")

    @property
    def user_task(self):
        """Return the user task required by Scenario.save."""
        return Task(goal="Say hello")

    @property
    def agent_task(self):
        """Return the agent task required by Scenario.save."""
        return Task(goal="Respond and exit")

    @property
    def user_actions(self):
        """Return user actions required by Scenario.save."""
        return Actions(instructions=["Greet the agent"])

    @property
    def agent_actions(self):
        """Return agent actions required by Scenario.save."""
        return Actions(instructions=["Return a greeting"])

    @property
    def user_resources(self):
        """Return user resources required by Scenario.save."""
        return Resources()

    @property
    def agent_resources(self):
        """Return agent resources required by Scenario.save."""
        return Resources()


class _DbAndCleanExitScenario(_CleanExitScenario):
    """Minimal scenario scored by DB-state hash and clean exit."""

    name = "fake_domain__db_and_exit"
    success_signals = (SuccessSignal.DB_STATE_MATCH, SuccessSignal.CLEAN_EXIT)


class _FakeBridge:
    """Deterministic bridge replacement for exercising runner orchestration cheaply."""

    final_response_file = "final_agent_response.json"

    def __init__(self, *args, **kwargs):
        """Initialize fake bridge state matching the runner's expected attributes."""
        self.output_dir = None
        self.stop_reason = STOP_REASON_EXIT
        self.scenario_summary = {}
        self.token_usage = {
            "agent": {"n_calls": 2, "prompt": 11, "completion": 5},
            "user": {"n_calls": 1, "prompt": 7, "completion": 3},
        }
        self.prepared = []
        self.ran_for = []

    async def prepare_for_scenario(self, scenario, scenario_dir):
        """Record preparation and create the fake final response artifact."""
        self.output_dir = scenario_dir
        self.prepared.append((scenario.name, scenario_dir))
        with open(f"{scenario_dir}/{self.final_response_file}", "w") as f:
            json.dump([{"message": "hello"}], f)

    async def run_scenario(self, duration):
        """Record the duration that the runner requested."""
        self.ran_for.append(duration)

    def get_metrics(self):
        """Return deterministic metrics in the same shape as VoiceAgentEvaluationBridge."""
        return {
            "total_turns": 2,
            "duration_seconds": 1.0,
            "turns": [
                {"role": "user", "text": "hello"},
                {"role": "agent", "text": "hi"},
            ],
            "latency_stats": {
                "count": 1,
                "mean_ms": 125.0,
                "p50_ms": 125.0,
                "p95_ms": 125.0,
                "min_ms": 125.0,
                "max_ms": 125.0,
            },
            "latencies": [
                {
                    "user_transcript": "hello",
                    "agent_transcript": "hi",
                    "latency_ms": 125.0,
                }
            ],
        }


class _DbHashFakeBridge(_FakeBridge):
    """Fake bridge that reports a matching scenario DB hash."""

    def __init__(self, *args, **kwargs):
        """Initialize with a matching scenario summary hash."""
        super().__init__(*args, **kwargs)
        self.scenario_summary = {"db_hash": get_dict_hash({"state": "done"})}


class _TimeoutFakeBridge(_FakeBridge):
    """Fake bridge that simulates a non-clean timeout and too few agent calls."""

    def __init__(self, *args, **kwargs):
        """Initialize with timeout stop reason and one agent call."""
        super().__init__(*args, **kwargs)
        self.stop_reason = "[TIMEOUT]"
        self.token_usage["agent"]["n_calls"] = 1


def test_run_dynamic_evaluation_with_fake_bridge_writes_run_artifacts(monkeypatch, tmp_path):
    """The real runner can aggregate one clean-exit scenario without real sockets or GPUs."""
    monkeypatch.setattr(runner_module, "VoiceAgentEvaluationBridge", _FakeBridge)
    logger = _FakeLogger()
    scenario = _CleanExitScenario()

    results = asyncio.run(
        runner_module.run_dynamic_evaluation(
            user_url="ws://fake-user",
            agent_url="ws://fake-agent",
            output_dir=str(tmp_path),
            scenarios=[scenario],
            pause_between_scenarios=0.0,
            duration_per_scenario=1,
            logger=logger,
        )
    )

    metrics_path = tmp_path / scenario.name / "metrics.json"
    all_metrics_path = tmp_path / "all_metrics.json"
    summary_path = tmp_path / "all_summary.txt"
    latencies_path = tmp_path / "all_latencies.csv"

    assert results[0]["scenario_name"] == scenario.name
    assert results[0]["is_successful"] is True
    assert results[0]["is_task_successful"] is True
    assert results[0]["clean_exit"] is True
    assert results[0]["is_action_match"] == "N/A"
    assert results[0]["token_usage"]["agent"]["n_calls"] == 2
    assert metrics_path.exists()
    assert json.loads(all_metrics_path.read_text())[0]["scenario_name"] == scenario.name
    assert "Overall Success Rate: 100.00%" in summary_path.read_text()
    assert "Token Usage:" in summary_path.read_text()
    assert '"fake_domain__clean_exit","hello","hi",125.0' in latencies_path.read_text()


def test_run_dynamic_evaluation_scores_matching_db_hash(monkeypatch, tmp_path):
    """The runner computes expected DB hashes and marks matching bridge summaries successful."""
    monkeypatch.setattr(runner_module, "VoiceAgentEvaluationBridge", _DbHashFakeBridge)
    scenario = _DbAndCleanExitScenario(expected_scenario_db={"state": "done"})

    results = asyncio.run(
        runner_module.run_dynamic_evaluation(
            user_url="ws://fake-user",
            agent_url="ws://fake-agent",
            output_dir=str(tmp_path),
            scenarios=[scenario],
            pause_between_scenarios=0.0,
            duration_per_scenario=1,
            logger=_FakeLogger(),
        )
    )

    assert results[0]["db_state_match"] is True
    assert results[0]["is_successful"] is True
    assert "db_state_expected_hash" in results[0]
    assert results[0]["db_state_actual_hash"] == get_dict_hash({"state": "done"})


def test_run_dynamic_evaluation_counts_timeout_and_low_turns_as_failure(monkeypatch, tmp_path):
    """A timeout with too few agent turns is counted as failed and excluded from per-signal buckets."""
    monkeypatch.setattr(runner_module, "VoiceAgentEvaluationBridge", _TimeoutFakeBridge)
    scenario = _CleanExitScenario()

    results = asyncio.run(
        runner_module.run_dynamic_evaluation(
            user_url="ws://fake-user",
            agent_url="ws://fake-agent",
            output_dir=str(tmp_path),
            scenarios=[scenario],
            pause_between_scenarios=0.0,
            duration_per_scenario=1,
            logger=_FakeLogger(),
            min_agent_turns=3,
        )
    )

    summary = (tmp_path / "all_summary.txt").read_text()
    assert results[0]["stop_reason"] == "[TIMEOUT]"
    assert results[0]["clean_exit"] is False
    assert results[0]["insufficient_agent_turns"] is True
    assert results[0]["is_successful"] is False
    assert "COUNTED AS FAILURE" in summary

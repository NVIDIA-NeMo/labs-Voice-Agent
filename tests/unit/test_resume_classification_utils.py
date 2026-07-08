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

"""Unit tests for standalone resume-state classification helpers."""

import json

from nemo_voice_agent.evaluation.resume import (
    classify_scenario_resume_state,
    count_agent_llm_messages,
    count_agent_responses,
)


def _write_metrics(scenario_dir, **overrides):
    """Write a minimal metrics.json fixture and return its dict."""
    scenario_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "total_turns": 3,
        "stop_reason": "[EXIT]",
        "token_usage": {
            "agent": {"n_calls": 2},
            "user": {"n_calls": 2},
        },
    }
    metrics.update(overrides)
    (scenario_dir / "metrics.json").write_text(json.dumps(metrics))
    return metrics


def test_count_agent_llm_messages_handles_missing_malformed_and_valid_contexts(tmp_path):
    """Agent LLM message counting returns None for absent/bad files and counts assistant roles."""
    missing = tmp_path / "missing"
    malformed = tmp_path / "malformed"
    valid = tmp_path / "valid"
    (malformed / "bot_logs_agent").mkdir(parents=True)
    (valid / "bot_logs_agent").mkdir(parents=True)
    (malformed / "bot_logs_agent" / "llm_context.json").write_text("{not json")
    (valid / "bot_logs_agent" / "llm_context.json").write_text(
        json.dumps(
            [
                {"role": "assistant"},
                {"role": "user"},
                {"role": "assistant"},
                "ignored",
            ]
        )
    )

    assert count_agent_llm_messages(str(missing)) is None
    assert count_agent_llm_messages(str(malformed)) is None
    assert count_agent_llm_messages(str(valid)) == 2


def test_count_agent_responses_prefers_metrics_then_disk_then_context(tmp_path):
    """Agent response counting prefers token_usage before falling back to saved context."""
    scenario_dir = tmp_path / "scenario"
    (scenario_dir / "bot_logs_agent").mkdir(parents=True)
    (scenario_dir / "bot_logs_agent" / "llm_context.json").write_text(
        json.dumps([{"role": "assistant"}, {"role": "assistant"}])
    )
    _write_metrics(scenario_dir, token_usage={"agent": {"n_calls": 4}})

    assert count_agent_responses(str(scenario_dir), {"token_usage": {"agent": {"n_calls": 7}}}) == 7
    assert count_agent_responses(str(scenario_dir)) == 4

    (scenario_dir / "metrics.json").write_text(json.dumps({"total_turns": 3}))
    assert count_agent_responses(str(scenario_dir)) == 2


def test_classify_scenario_resume_state_covers_fresh_in_flight_and_completed(tmp_path):
    """Resume classification reports fresh, in-flight, low-turn, stalled, and completed states."""
    assert classify_scenario_resume_state(str(tmp_path / "fresh")) == ("fresh", "no subdir")

    no_metrics = tmp_path / "no_metrics"
    no_metrics.mkdir()
    assert classify_scenario_resume_state(str(no_metrics)) == ("in_flight", "no metrics.json (in-flight)")

    unreadable = tmp_path / "unreadable"
    unreadable.mkdir()
    (unreadable / "metrics.json").write_text("{not json")
    state, reason = classify_scenario_resume_state(str(unreadable))
    assert state == "in_flight"
    assert "metrics.json unreadable" in reason

    zero_turns = tmp_path / "zero_turns"
    _write_metrics(zero_turns, total_turns=0)
    assert classify_scenario_resume_state(str(zero_turns)) == (
        "in_flight",
        "0 turns (bot crashed before audio)",
    )

    stalled = tmp_path / "stalled"
    _write_metrics(stalled, token_usage={"agent": {"n_calls": 1}}, stop_reason="[TIMEOUT]")
    assert classify_scenario_resume_state(str(stalled), min_agent_turns=3) == (
        "in_flight",
        "1 agent LLM response(s) < 3 ([TIMEOUT])",
    )

    completed = tmp_path / "completed"
    _write_metrics(completed, stop_reason="[EXIT]")
    assert classify_scenario_resume_state(str(completed), min_agent_turns=2) == ("completed", "[EXIT]")

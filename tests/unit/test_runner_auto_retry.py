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

"""Tests for ``auto_retry_reason`` — the in-run automatic re-run decision.

The function separates two triggers by how certain the failure is, and the
distinction is the whole point of the feature:

- **Hard** (zero conversation turns) — the bots never exchanged audio, so no
  measurement exists and re-running cannot bias anything. On by default.
- **Soft** (agent responded, but below ``min_agent_turns``) — a heuristic.
  Because such scenarios are forced to ``is_successful=False``, retrying turns
  a guaranteed failure into a fresh draw, which inflates the success rate
  whenever the floor catches a scenario that would legitimately have passed.
  Off by default.

The measured motivation: across 500 scenarios the agent LLM-call distribution
bottoms out at exactly 3, where 21 legitimate clean exits sit (11 of them
successful). Retrying that cluster would re-roll real results.
"""

import pytest

from nemo_voice_agent.evaluation.runner import auto_retry_reason


DEFAULTS = {"on_stale": True, "on_insufficient_turns": False}


class TestHardFailureTrigger:
    """Zero conversation turns — an unambiguous infrastructure failure."""

    @pytest.mark.unit
    def test_zero_turns_retries_by_default(self):
        reason = auto_retry_reason(0, 0, 2, **DEFAULTS)
        assert reason is not None
        assert "0 conversation turns" in reason

    @pytest.mark.unit
    def test_zero_turns_respects_opt_out(self):
        assert auto_retry_reason(0, 0, 2, on_stale=False, on_insufficient_turns=False) is None

    @pytest.mark.unit
    def test_zero_turns_fires_even_when_floor_disabled(self):
        """The hard trigger is independent of ``min_agent_turns``."""
        assert auto_retry_reason(0, 0, 0, **DEFAULTS) is not None


class TestSoftHeuristicTrigger:
    """Below ``min_agent_turns`` — off by default to avoid re-rolling results."""

    @pytest.mark.unit
    def test_below_floor_does_not_retry_by_default(self):
        assert auto_retry_reason(6, 1, 2, **DEFAULTS) is None

    @pytest.mark.unit
    def test_below_floor_retries_when_opted_in(self):
        reason = auto_retry_reason(6, 1, 2, on_stale=True, on_insufficient_turns=True)
        assert reason is not None
        assert "min_agent_turns=2" in reason

    @pytest.mark.unit
    @pytest.mark.parametrize("count", [3, 4, 5, 6])
    def test_legitimate_short_scenarios_are_never_retried(self, count):
        """The measured floor of legitimate clean exits is 3 agent responses.

        With the shipped default of ``min_agent_turns=2`` those scenarios must
        survive untouched even when the soft trigger is opted into, otherwise
        21 real results per run would be re-rolled.
        """
        assert auto_retry_reason(10, count, 2, on_stale=True, on_insufficient_turns=True) is None

    @pytest.mark.unit
    def test_floor_disabled_never_fires_soft_trigger(self):
        assert auto_retry_reason(6, 1, 0, on_stale=True, on_insufficient_turns=True) is None


class TestSignalRobustness:
    """Guards against retrying on a signal that is merely unavailable."""

    @pytest.mark.unit
    def test_missing_token_usage_does_not_trigger_a_retry(self):
        """A backend that does not report token usage must not cause a retry storm.

        ``token_usage.agent.n_calls`` only increments on an RTVI token-usage
        message, so it reads 0 on a healthy run against such a backend. The
        hard trigger therefore keys on conversation turns, not on that counter.
        """
        assert auto_retry_reason(12, 0, 2, on_stale=True, on_insufficient_turns=True) is None

    @pytest.mark.unit
    def test_unavailable_agent_count_is_not_a_guess(self):
        assert auto_retry_reason(6, None, 2, on_stale=True, on_insufficient_turns=True) is None

    @pytest.mark.unit
    def test_unknown_turn_count_does_not_fire_hard_trigger(self):
        assert auto_retry_reason(None, 5, 2, **DEFAULTS) is None

    @pytest.mark.unit
    def test_healthy_scenario_is_accepted(self):
        assert auto_retry_reason(20, 12, 2, on_stale=True, on_insufficient_turns=True) is None


class TestTriggerPrecedence:
    @pytest.mark.unit
    def test_hard_trigger_wins_when_both_apply(self):
        """Zero turns implies zero agent responses; report the hard reason."""
        reason = auto_retry_reason(0, 0, 2, on_stale=True, on_insufficient_turns=True)
        assert "0 conversation turns" in reason

    @pytest.mark.unit
    def test_all_triggers_disabled_never_retries(self):
        assert auto_retry_reason(0, 0, 2, on_stale=False, on_insufficient_turns=False) is None

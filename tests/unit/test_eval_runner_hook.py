# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Smoke tests for the generic runner hook landed in commit 1.

Covers two surface-level behaviors without spinning up an LLM or a bot server:
    1. ``Scenario.setup_shared_state`` propagates per-side state through the
       bridge's per-side seeding step, with the user-side and agent-side
       states kept separate.
    2. ``check_if_task_success`` honors ``disallow_extra_items``: extras pass
       in lenient mode, fail in strict mode; pred-shorter-than-ref fails in
       both modes.
"""

import json
import sys
from pathlib import Path

import pytest

from nemo_voice_agent.evaluation import get_eval_data_root
from nemo_voice_agent.evaluation.scenarios.classes import (
    Actions,
    Persona,
    Resources,
    Scenario,
    SuccessSignal,
    Task,
)
from nemo_voice_agent.evaluation.utils import check_if_task_success


# ---------------------------------------------------------------------------
# Tiny fully-realized Scenario so we can exercise setup_shared_state in
# isolation. Every abstract @property gets a minimal stub.
# ---------------------------------------------------------------------------


class _StubPerSideStateScenario(Scenario):
    """Test scenario that writes distinct per-side state values."""

    name = "test__stub_per_side_state"
    # Not actually scored by this test; declaration satisfies the
    # Scenario.__init_subclass__ non-empty check.
    success_signals = (SuccessSignal.ACTION_MATCH,)

    def setup_shared_state(self, state: dict, side: str) -> None:
        if side == "agent":
            state["marker"] = "agent_value"
            state["db_path"] = "agent/fixture.json"
        elif side == "user":
            state["marker"] = "user_value"

    @property
    def user_persona(self) -> Persona:
        return Persona(role="user", name="U", background="b", personality="p")

    @property
    def agent_persona(self) -> Persona:
        return Persona(role="agent", name="A", background="b", personality="p")

    @property
    def user_task(self) -> Task:
        return Task(goal="g")

    @property
    def agent_task(self) -> Task:
        return Task(goal="g")

    @property
    def user_actions(self) -> Actions:
        return Actions(instructions=["i"])

    @property
    def agent_actions(self) -> Actions:
        return Actions(instructions=["i"])

    @property
    def user_resources(self) -> Resources:
        return Resources()

    @property
    def agent_resources(self) -> Resources:
        return Resources()


# ---------------------------------------------------------------------------
# 1. setup_shared_state propagation
# ---------------------------------------------------------------------------


def test_setup_shared_state_default_is_noop():
    """Base-class default leaves state untouched; existing domains unaffected."""
    state: dict = {}

    class _Plain(Scenario):
        name = "test__plain"
        success_signals = (SuccessSignal.ACTION_MATCH,)

        @property
        def user_persona(self) -> Persona:
            return Persona(role="user", name="U", background="b", personality="p")

        @property
        def agent_persona(self) -> Persona:
            return Persona(role="agent", name="A", background="b", personality="p")

        @property
        def user_task(self) -> Task:
            return Task(goal="g")

        @property
        def agent_task(self) -> Task:
            return Task(goal="g")

        @property
        def user_actions(self) -> Actions:
            return Actions(instructions=["i"])

        @property
        def agent_actions(self) -> Actions:
            return Actions(instructions=["i"])

        @property
        def user_resources(self) -> Resources:
            return Resources()

        @property
        def agent_resources(self) -> Resources:
            return Resources()

    s = _Plain()
    s.setup_shared_state(state, "agent")
    s.setup_shared_state(state, "user")
    assert state == {}


def test_setup_shared_state_per_side_isolation():
    """Per-side state stays separate; runner serialization round-trips."""
    s = _StubPerSideStateScenario()
    user_state, agent_state = {}, {}
    s.setup_shared_state(user_state, "user")
    s.setup_shared_state(agent_state, "agent")

    # Per-side values are distinct
    assert user_state == {"marker": "user_value"}
    assert agent_state == {"marker": "agent_value", "db_path": "agent/fixture.json"}

    # Each side's state JSON-round-trips intact (the bridge JSON-encodes
    # them before sending to the bot via ``update_system_prompt``'s
    # ``shared_state_init`` argument).
    assert json.loads(json.dumps(user_state)) == user_state
    assert json.loads(json.dumps(agent_state)) == agent_state


def test_setup_shared_state_disallow_extra_items_default():
    """disallow_extra_items defaults to False (lenient comparator preserved)."""
    s = _StubPerSideStateScenario()
    assert s.disallow_extra_items is False


# ---------------------------------------------------------------------------
# 2. EVAL_DATA_ROOT resolution
# ---------------------------------------------------------------------------


def test_eval_data_root_falls_back_to_repo_path(monkeypatch):
    """Without env override, resolves to <repo>/nemo_voice_agent/evaluation/data.

    Was previously asserting against the old NeMo-main layout
    (``examples/voice_agent/nemo_voice_agent/evaluation/data``); the eval framework now lives
    in the standalone NeMo-Voice-Agent repo where the data root is one level
    higher (``<repo>/nemo_voice_agent/evaluation/data``). Repo *name* isn't asserted because
    users may clone into any directory. Existence + a known checked-in
    fixture path is the meaningful invariant.
    """
    monkeypatch.delenv("EVAL_DATA_ROOT", raising=False)
    root = get_eval_data_root()
    assert root.name == "data"
    assert root.parent.name == "evaluation"
    assert root.is_dir(), f"eval data root {root} does not exist"
    # eva_airline_dataset.jsonl is checked in at nemo_voice_agent/evaluation/data/eva_airline/.
    assert (root / "eva_airline" / "eva_airline_dataset.jsonl").exists(), (
        f"expected eva_airline/eva_airline_dataset.jsonl under {root}"
    )


def test_eval_data_root_honors_env_var(monkeypatch, tmp_path):
    """$EVAL_DATA_ROOT overrides the repo-default path."""
    monkeypatch.setenv("EVAL_DATA_ROOT", str(tmp_path))
    assert get_eval_data_root() == tmp_path


def test_eval_data_root_is_lazy(monkeypatch, tmp_path):
    """Function (not constant) — env-var changes after import take effect."""
    monkeypatch.delenv("EVAL_DATA_ROOT", raising=False)
    before = get_eval_data_root()
    monkeypatch.setenv("EVAL_DATA_ROOT", str(tmp_path))
    after = get_eval_data_root()
    assert before != after
    assert after == tmp_path


# ---------------------------------------------------------------------------
# 3. disallow_extra_items in the comparator
# ---------------------------------------------------------------------------


def _write_json(tmp_path: Path, name: str, payload) -> str:
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return str(p)


def test_lenient_accepts_pred_with_extras(tmp_path):
    """Default mode: pred=[A, B, C] vs ref=[A, B] passes (extras tolerated)."""
    ref = _write_json(tmp_path, "ref.json", [{"x": 1}, {"x": 2}])
    pred = _write_json(tmp_path, "pred.json", [{"x": 1}, {"x": 2}, {"x": 3}])
    assert check_if_task_success(reference=ref, prediction=pred) is True
    assert check_if_task_success(reference=ref, prediction=pred, disallow_extra_items=False) is True


def test_strict_rejects_pred_with_extras(tmp_path):
    """Strict mode: pred=[A, B, C] vs ref=[A, B] fails (length mismatch)."""
    ref = _write_json(tmp_path, "ref.json", [{"x": 1}, {"x": 2}])
    pred = _write_json(tmp_path, "pred.json", [{"x": 1}, {"x": 2}, {"x": 3}])
    assert check_if_task_success(reference=ref, prediction=pred, disallow_extra_items=True) is False


def test_pred_shorter_than_ref_fails_both_modes(tmp_path):
    """pred=[A] vs ref=[A, B] fails regardless of mode (B is unmatched)."""
    ref = _write_json(tmp_path, "ref.json", [{"x": 1}, {"x": 2}])
    pred = _write_json(tmp_path, "pred.json", [{"x": 1}])
    assert check_if_task_success(reference=ref, prediction=pred) is False
    assert check_if_task_success(reference=ref, prediction=pred, disallow_extra_items=True) is False


def test_strict_accepts_exact_bijection_unordered(tmp_path):
    """Strict mode: equal lengths + every ref matched ⇒ pass (order-independent)."""
    ref = _write_json(tmp_path, "ref.json", [{"x": 1}, {"x": 2}])
    pred = _write_json(tmp_path, "pred.json", [{"x": 2}, {"x": 1}])
    assert check_if_task_success(reference=ref, prediction=pred, disallow_extra_items=True) is True


def test_dict_reference_unaffected_by_strict(tmp_path):
    """Situation 1 (dict ref, dict pred): strict mode is a no-op (lengths are 1=1)."""
    ref = _write_json(tmp_path, "ref.json", {"x": 1})
    pred = _write_json(tmp_path, "pred.json", {"x": 1, "y": 2})
    assert check_if_task_success(reference=ref, prediction=pred) is True
    assert check_if_task_success(reference=ref, prediction=pred, disallow_extra_items=True) is True


def test_situation_2_unaffected_by_strict(tmp_path):
    """Situation 2 (dict ref, list-of-dicts pred): last pred is matched; strict no-ops."""
    ref = _write_json(tmp_path, "ref.json", {"x": 2})
    pred = _write_json(tmp_path, "pred.json", [{"x": 1}, {"x": 2}])
    # Lenient: pred is reduced to [pred[-1]] before length check, so strict still passes
    assert check_if_task_success(reference=ref, prediction=pred) is True
    assert check_if_task_success(reference=ref, prediction=pred, disallow_extra_items=True) is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

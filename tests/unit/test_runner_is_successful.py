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


"""Unit tests for the per-scenario ``success_signals`` whitelist + the
``Scenario.compute_is_successful`` composite logic.

Covers:
- ``__init_subclass__`` validates non-empty ``success_signals`` at
  class-definition time for concrete scenarios.
- ``compute_is_successful`` returns ``"N/A"`` when no whitelisted signal is
  applicable, ``True`` when every applicable signal passes, ``False`` when any
  fails.
- ``cached_property`` derivation (retail-style) returns different tuples based
  on per-scenario opt-ins.
- Unknown signal name raises ``ValueError`` at composite computation time.
"""

from functools import cached_property
from typing import Optional

import pytest

from nemo_voice_agent.evaluation.scenarios.classes import (
    Actions,
    Persona,
    Resources,
    Scenario,
    SuccessSignal,
    Task,
)


def _make_minimal_scenario(success_signals, name="test__minimal", nl_assertions=None):
    """Build a concrete Scenario subclass with the requested whitelist."""
    sig_attr = success_signals  # capture for closure

    class _T(Scenario):
        # Indirection via class-body assignment so caller can also pass a
        # ``cached_property`` (which we want to assign without triggering it).
        pass

    _T.name = name
    _T.success_signals = sig_attr
    if nl_assertions is not None:
        _T.nl_assertions = nl_assertions

    # Stub the abstract properties.
    _T.user_persona = property(lambda self: Persona(role="u", name="U", background="b", personality="p"))
    _T.agent_persona = property(lambda self: Persona(role="a", name="A", background="b", personality="p"))
    _T.user_task = property(lambda self: Task(goal="g"))
    _T.agent_task = property(lambda self: Task(goal="g"))
    _T.user_actions = property(lambda self: Actions(instructions=[]))
    _T.agent_actions = property(lambda self: Actions(instructions=[]))
    _T.user_resources = property(lambda self: Resources())
    _T.agent_resources = property(lambda self: Resources())

    return _T()


def test_concrete_scenario_without_success_signals_raises():
    """A class declaring ``name`` MUST set ``success_signals``."""
    with pytest.raises(TypeError, match="success_signals must be a non-empty tuple"):

        class _Missing(Scenario):
            name = "test__missing"
            # no success_signals → raises at class-definition time.


def test_concrete_scenario_with_empty_tuple_raises():
    """Empty tuple is also invalid — equivalent to "score nothing"."""
    with pytest.raises(TypeError, match="success_signals must be a non-empty tuple"):

        class _Empty(Scenario):
            name = "test__empty"
            success_signals = ()


def test_abstract_base_without_name_does_not_raise():
    """Domain base classes (no ``name``) are skipped — they declare on a
    sibling property or leave ``success_signals=None`` for subclasses to set."""

    class _AbstractBase(Scenario):
        pass  # no name, no success_signals — valid (abstract).

    # No assertion needed; class-definition completes.
    assert _AbstractBase.success_signals is None


def test_compute_returns_na_when_no_whitelisted_signal_applicable():
    """All whitelisted signals are None → ``"N/A"``."""
    scenario = _make_minimal_scenario((SuccessSignal.JUDGE_PASSED,))
    signals = {s: None for s in SuccessSignal}
    assert scenario.compute_is_successful(signals) == "N/A"


def test_compute_returns_true_when_all_applicable_pass():
    scenario = _make_minimal_scenario((SuccessSignal.DB_STATE_MATCH, SuccessSignal.NL_ASSERTION))
    signals = {
        SuccessSignal.ACTION_MATCH: None,
        SuccessSignal.DB_STATE_MATCH: True,
        SuccessSignal.DB_STATE_ASSERTION: None,
        SuccessSignal.NL_ASSERTION: True,
        SuccessSignal.JUDGE_PASSED: None,
    }
    assert scenario.compute_is_successful(signals) is True


def test_compute_returns_false_when_any_applicable_fails():
    scenario = _make_minimal_scenario((SuccessSignal.DB_STATE_MATCH, SuccessSignal.NL_ASSERTION))
    signals = {
        SuccessSignal.ACTION_MATCH: True,  # not in whitelist — excluded
        SuccessSignal.DB_STATE_MATCH: True,
        SuccessSignal.DB_STATE_ASSERTION: None,
        SuccessSignal.NL_ASSERTION: False,  # this fails the verdict
        SuccessSignal.JUDGE_PASSED: True,  # not in whitelist
    }
    assert scenario.compute_is_successful(signals) is False


def test_non_whitelisted_signal_values_dont_affect_verdict():
    """Even when a non-whitelisted signal returns False, the verdict is
    determined only by whitelisted ones."""
    scenario = _make_minimal_scenario((SuccessSignal.DB_STATE_MATCH,))
    signals = {
        SuccessSignal.ACTION_MATCH: False,  # excluded — should not gate
        SuccessSignal.DB_STATE_MATCH: True,
        SuccessSignal.DB_STATE_ASSERTION: None,
        SuccessSignal.NL_ASSERTION: None,
        SuccessSignal.JUDGE_PASSED: False,  # excluded
    }
    assert scenario.compute_is_successful(signals) is True


def test_retail_style_cached_property_derives_from_nl_assertions():
    """The retail rule: NL assertions present → 2 signals; absent → 1."""

    class _RetailLike(Scenario):
        nl_assertions: Optional[list] = None

        @cached_property
        def success_signals(self):
            if self.nl_assertions:
                return (SuccessSignal.DB_STATE_MATCH, SuccessSignal.NL_ASSERTION)
            return (SuccessSignal.DB_STATE_MATCH,)

        @property
        def user_persona(self):
            return Persona(role="u", name="U", background="b", personality="p")

        @property
        def agent_persona(self):
            return Persona(role="a", name="A", background="b", personality="p")

        @property
        def user_task(self):
            return Task(goal="g")

        @property
        def agent_task(self):
            return Task(goal="g")

        @property
        def user_actions(self):
            return Actions(instructions=[])

        @property
        def agent_actions(self):
            return Actions(instructions=[])

        @property
        def user_resources(self):
            return Resources()

        @property
        def agent_resources(self):
            return Resources()

    class _RetailAction(_RetailLike):
        name = "test__retail_action_only"
        # no nl_assertions

    class _RetailWithNL(_RetailLike):
        name = "test__retail_with_nl"
        nl_assertions = ["Agent should mention the cancellation policy."]

    action_inst = _RetailAction()
    nl_inst = _RetailWithNL()

    assert action_inst.success_signals == (SuccessSignal.DB_STATE_MATCH,)
    assert nl_inst.success_signals == (
        SuccessSignal.DB_STATE_MATCH,
        SuccessSignal.NL_ASSERTION,
    )


def test_clean_exit_gates_when_in_whitelist():
    """A scenario with CLEAN_EXIT in success_signals fails if stop reason wasn't [EXIT].

    Regression: prior to the CLEAN_EXIT signal, scenarios where the agent landed on
    the correct end-state DB but TIMEOUTed (never called EndConversationTool) were
    incorrectly marked successful. The most concrete case was eva_airline
    policy-refusal scenarios where ``expected_scenario_db == initial_scenario_db``,
    making `db_state_match=True` a "win by inaction" — `CLEAN_EXIT` plugs that gap.
    """
    scenario = _make_minimal_scenario((SuccessSignal.DB_STATE_MATCH, SuccessSignal.CLEAN_EXIT))
    # DB matches BUT conversation TIMEOUTed — CLEAN_EXIT=False → composite=False.
    signals = {
        SuccessSignal.ACTION_MATCH: None,
        SuccessSignal.DB_STATE_MATCH: True,
        SuccessSignal.DB_STATE_ASSERTION: None,
        SuccessSignal.NL_ASSERTION: None,
        SuccessSignal.JUDGE_PASSED: None,
        SuccessSignal.CLEAN_EXIT: False,
    }
    assert scenario.compute_is_successful(signals) is False
    # Same DB pass + clean exit → True.
    signals[SuccessSignal.CLEAN_EXIT] = True
    assert scenario.compute_is_successful(signals) is True


def test_every_concrete_scenario_includes_clean_exit():
    """Closure discipline is universal — every domain's whitelist must include CLEAN_EXIT.

    Regression guard: if a future domain is added without CLEAN_EXIT, this test fails
    so the author has to make the choice explicitly (with a documented exception)
    rather than silently drop the gate.
    """
    import nemo_voice_agent.evaluation.scenarios.data  # populates ALL_EVAL_SCENARIOS  # noqa: F401
    from nemo_voice_agent.evaluation.scenarios import ALL_EVAL_SCENARIOS

    missing = []
    for name, cls in ALL_EVAL_SCENARIOS.items():
        inst = cls()
        signals = inst.success_signals or ()
        if SuccessSignal.CLEAN_EXIT not in signals:
            missing.append(name)
    assert not missing, (
        f"{len(missing)} scenarios do not include CLEAN_EXIT in their success_signals. "
        f"If a future domain genuinely shouldn't gate on closure (e.g., a streaming-Q&A "
        f"interaction with no explicit termination), add an opt-out comment in the "
        f"domain base class. Otherwise add CLEAN_EXIT to the whitelist. "
        f"Examples: {sorted(missing)[:5]}"
    )


def test_unknown_signal_in_whitelist_raises():
    """A whitelist entry that isn't a valid ``SuccessSignal`` value raises."""

    class _Bogus(Scenario):
        name = "test__bogus"
        success_signals = ("is_action_match", "fake_signal_name")  # second is unknown

        @property
        def user_persona(self):
            return Persona(role="u", name="U", background="b", personality="p")

        @property
        def agent_persona(self):
            return Persona(role="a", name="A", background="b", personality="p")

        @property
        def user_task(self):
            return Task(goal="g")

        @property
        def agent_task(self):
            return Task(goal="g")

        @property
        def user_actions(self):
            return Actions(instructions=[])

        @property
        def agent_actions(self):
            return Actions(instructions=[])

        @property
        def user_resources(self):
            return Resources()

        @property
        def agent_resources(self):
            return Resources()

    inst = _Bogus()
    signals = {s: True for s in SuccessSignal}
    with pytest.raises(ValueError, match="unknown"):
        inst.compute_is_successful(signals)


def test_strenum_members_serialize_to_canonical_keys():
    """StrEnum members compare equal to their string values — preserves the
    on-disk JSON shape of ``metrics.json`` after the refactor."""
    assert SuccessSignal.ACTION_MATCH == "is_action_match"
    assert SuccessSignal.DB_STATE_MATCH == "db_state_match"
    assert SuccessSignal.DB_STATE_ASSERTION == "db_state_assertion"
    assert SuccessSignal.NL_ASSERTION == "nl_assertion"
    assert SuccessSignal.JUDGE_PASSED == "judge_passed"
    # str() likewise
    assert str(SuccessSignal.ACTION_MATCH) == "is_action_match"

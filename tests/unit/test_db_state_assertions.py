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

"""Tests for the ``db_state_assertions`` framework.

Synthetic-only — uses fake predicates registered against a fake domain to
exercise the full registration + dispatch + verdict chain without
depending on any real domain port. Real ported predicates should re-run
this suite to confirm nothing structural changed.

The registry is keyed by ``(domain, func_name)`` only — predicates are
side-agnostic; ``side`` is caller-side routing metadata used by the dispatcher
to pick which pulled DB to invoke the predicate against, NOT a registry key.

Coverage:

- Registration: decorator (no side arg), double-register collision.
- Dispatch happy path: ``passed=True`` when predicate output == ``assert_value``.
- Dispatch ``passed=False``: predicate returns opposite of ``assert_value``.
- Side routing: ``side="agent"`` picks ``db``; ``side="user"`` picks ``user_db``.
- Missing predicate (typo / not registered): ``passed=False`` with explanatory
  ``error``, never raises.
- Wrong / unknown ``side``: ``passed=False`` with explanatory ``error``.
- Missing target DB (e.g., user-side assertion but ``user_db=None``):
  ``passed=False`` with explanatory ``error``.
- Predicate raises an exception: ``passed=False`` with ``error`` capturing the
  exception type and message.
- Verdict shape mirrors ``nl_assertion_verdicts`` (same keys: ``passed``,
  per-record verdict, ``message`` optional) so the runner aggregation code
  path is uniform.
"""

import pytest

from nemo_voice_agent.evaluation.db_state_predicates import (
    ALL_DB_STATE_PREDICATES,
    evaluate_db_state_assertion,
    list_registered_predicates,
    register_db_state_predicate,
)


# Use a synthetic domain so this test doesn't collide with real domain
# registrations (e.g. tau2_telecom) when both are loaded.
_TEST_DOMAIN = "_test_db_state_assertions_synthetic_domain"


@pytest.fixture(autouse=True)
def _clean_registry():
    """Wipe just our synthetic-domain bucket between tests so registration
    failure cases don't pollute one another. Leave other domains alone so
    we don't disturb any predicates registered at import time."""
    yield
    ALL_DB_STATE_PREDICATES.pop(_TEST_DOMAIN, None)


def _register_count_equals():
    """Register a passing predicate."""

    @register_db_state_predicate(domain=_TEST_DOMAIN)
    def assert_count_equals(db: dict, expected: int) -> bool:
        return db.get("count") == expected

    return assert_count_equals


def _register_status_equals():
    """Register a different predicate (will be invoked via side='agent')."""

    @register_db_state_predicate(domain=_TEST_DOMAIN)
    def assert_status_equals(db: dict, expected_status: str) -> bool:
        return db.get("status") == expected_status

    return assert_status_equals


# ---------------------------------------------------------------------------
# 1. Registration
# ---------------------------------------------------------------------------


def test_register_predicate_lands_in_registry():
    _register_count_equals()
    registered = list_registered_predicates(_TEST_DOMAIN)
    assert registered == {_TEST_DOMAIN: ["assert_count_equals"]}


def test_register_two_predicates_same_domain():
    _register_count_equals()
    _register_status_equals()
    registered = list_registered_predicates(_TEST_DOMAIN)
    assert registered[_TEST_DOMAIN] == ["assert_count_equals", "assert_status_equals"]


def test_register_predicate_collision_raises():
    _register_count_equals()
    with pytest.raises(ValueError, match="already registered"):
        _register_count_equals()


# ---------------------------------------------------------------------------
# 2. Dispatch — happy path
# ---------------------------------------------------------------------------


def test_evaluate_passed_true_when_predicate_matches_assert_value():
    _register_count_equals()
    assertion = {
        "side": "user",
        "func_name": "assert_count_equals",
        "arguments": {"expected": 5},
        "assert_value": True,
    }
    verdict = evaluate_db_state_assertion(domain=_TEST_DOMAIN, assertion=assertion, db=None, user_db={"count": 5})
    assert verdict["passed"] is True
    assert verdict["actual"] is True
    assert verdict["expected"] is True
    assert verdict["error"] is None
    # Verdict shape must include these keys for downstream runner aggregation.
    for key in ("func_name", "side", "passed", "expected", "actual", "message", "error"):
        assert key in verdict


def test_evaluate_passed_false_when_assert_value_is_false_and_predicate_returns_false():
    """``assert_value: false`` flips the polarity — predicate returning False
    should *pass* in this case, but here the predicate returns True so it
    fails. Mirrors the ``assert_internet_not_excellent`` upstream pattern."""
    _register_count_equals()
    assertion = {
        "side": "user",
        "func_name": "assert_count_equals",
        "arguments": {"expected": 5},
        "assert_value": False,  # we expect the predicate to return False
    }
    verdict = evaluate_db_state_assertion(domain=_TEST_DOMAIN, assertion=assertion, db=None, user_db={"count": 5})
    # Predicate returned True, but assert_value=False → mismatch → fail.
    assert verdict["passed"] is False
    assert verdict["actual"] is True


def test_evaluate_agent_side_uses_db_not_user_db():
    """``side`` is the caller-side routing metadata that picks which DB to
    pass — agent's ``db`` vs user's ``user_db``. Same predicate, different
    DBs, depending on the assertion's ``side`` field."""
    _register_status_equals()
    assertion = {
        "side": "agent",
        "func_name": "assert_status_equals",
        "arguments": {"expected_status": "active"},
        "assert_value": True,
    }
    verdict = evaluate_db_state_assertion(
        domain=_TEST_DOMAIN,
        assertion=assertion,
        db={"status": "active"},
        user_db={"status": "inactive"},  # decoy — must not be read for agent side
    )
    assert verdict["passed"] is True
    assert verdict["actual"] is True


# ---------------------------------------------------------------------------
# 3. Dispatch — failure modes (never raise; always return verdict with error)
# ---------------------------------------------------------------------------


def test_evaluate_missing_predicate_returns_error_verdict():
    assertion = {
        "side": "user",
        "func_name": "assert_does_not_exist",
        "arguments": {},
        "assert_value": True,
    }
    verdict = evaluate_db_state_assertion(domain=_TEST_DOMAIN, assertion=assertion, db=None, user_db={"count": 1})
    assert verdict["passed"] is False
    assert verdict["error"] is not None
    assert "No predicate" in verdict["error"]
    assert "assert_does_not_exist" in verdict["error"]


def test_evaluate_unknown_side_returns_error_verdict():
    _register_count_equals()
    assertion = {
        "side": "robot",  # not a valid side
        "func_name": "assert_count_equals",
        "arguments": {"expected": 1},
        "assert_value": True,
    }
    verdict = evaluate_db_state_assertion(domain=_TEST_DOMAIN, assertion=assertion, db=None, user_db={"count": 1})
    assert verdict["passed"] is False
    assert "Unknown side" in (verdict["error"] or "")


def test_evaluate_missing_user_db_returns_error_verdict():
    _register_count_equals()
    assertion = {
        "side": "user",
        "func_name": "assert_count_equals",
        "arguments": {"expected": 1},
        "assert_value": True,
    }
    verdict = evaluate_db_state_assertion(domain=_TEST_DOMAIN, assertion=assertion, db={"count": 1}, user_db=None)
    assert verdict["passed"] is False
    assert "user DB" in (verdict["error"] or "")


def test_evaluate_predicate_raising_exception_returns_error_verdict():
    @register_db_state_predicate(domain=_TEST_DOMAIN)
    def assert_explodes(db: dict, **_) -> bool:
        raise RuntimeError("synthetic failure")

    assertion = {
        "side": "user",
        "func_name": "assert_explodes",
        "arguments": {},
        "assert_value": True,
    }
    verdict = evaluate_db_state_assertion(domain=_TEST_DOMAIN, assertion=assertion, db=None, user_db={"x": 1})
    assert verdict["passed"] is False
    assert "Predicate raised" in (verdict["error"] or "")
    assert "RuntimeError" in (verdict["error"] or "")
    assert "synthetic failure" in (verdict["error"] or "")

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

"""Tests for the M4 ``initialization_actions`` framework.

Synthetic — no real telecom toolkit yet. Covers the registration + dispatcher
(``initialization_functions.py``) and the bot-side RTVI handler
(``create_apply_initialization_actions_action``). The bridge-side helper is
exercised at integration time in M5; here we stop at the bot's handler since
the bridge talks to it over WebSocket and that's not in unit scope.

Key invariants under test:

- Registry is keyed by ``(domain, func_name)`` only — no ``side``. Function
  names are unique within a domain (16+4 in tau2 telecom), so ``side`` is
  caller-side routing metadata, not function identity.
- Dispatcher takes a single ``db`` dict; all actions in the list apply to
  it. The bridge has already filtered by side before calling.
- The bot handler reads ``side`` from the RTVI payload and uses it solely to
  pick which DB key (``"db"`` for agent, ``"user_db"`` for user) to fetch
  from its own ``shared_state`` — the per-action ``side`` field is preserved
  for upstream traceability but not consulted by the dispatcher.
"""

import asyncio

import pytest

from nemo_voice_agent.evaluation.initialization_functions import (
    ALL_INITIALIZATION_FUNCTIONS,
    apply_initialization_actions,
    list_registered_initialization_functions,
    register_initialization_function,
)
from nemo_voice_agent.pipecat.processors.frameworks.rtvi_actions import (
    SharedStateRef,
    create_apply_initialization_actions_action,
)

_TEST_DOMAIN = "_test_init_actions_synthetic_domain"


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    ALL_INITIALIZATION_FUNCTIONS.pop(_TEST_DOMAIN, None)


def _register_set_count():
    @register_initialization_function(domain=_TEST_DOMAIN)
    def set_count(db: dict, value: int) -> None:
        db["count"] = value

    return set_count


def _register_set_status():
    @register_initialization_function(domain=_TEST_DOMAIN)
    def set_status(db: dict, status: str) -> None:
        db["status"] = status

    return set_status


# ---------------------------------------------------------------------------
# 1. Registration
# ---------------------------------------------------------------------------


def test_register_init_function_lands_in_registry():
    _register_set_count()
    registered = list_registered_initialization_functions(_TEST_DOMAIN)
    assert registered == {_TEST_DOMAIN: ["set_count"]}


def test_register_two_init_functions_same_domain():
    _register_set_count()
    _register_set_status()
    registered = list_registered_initialization_functions(_TEST_DOMAIN)
    assert registered[_TEST_DOMAIN] == ["set_count", "set_status"]


def test_register_init_function_collision_raises():
    _register_set_count()
    with pytest.raises(ValueError, match="already registered"):
        _register_set_count()


# ---------------------------------------------------------------------------
# 2. Module dispatcher
# ---------------------------------------------------------------------------


def test_dispatch_mutates_db_in_place():
    _register_set_count()
    db = {}
    result = apply_initialization_actions(
        domain=_TEST_DOMAIN,
        actions=[{"side": "user", "func_name": "set_count", "arguments": {"value": 42}}],
        db=db,
    )
    assert result == {"success": True, "errors": []}
    assert db == {"count": 42}


def test_dispatch_ignores_per_action_side_field():
    """The per-action ``side`` field is preserved upstream for traceability
    but the dispatcher itself doesn't consult it. All actions in the list
    apply to the single ``db`` argument; the bridge is responsible for
    splitting beforehand.

    Concretely: if the bridge somehow forgot to filter and sent a mixed
    ``side`` list, the dispatcher still runs each function against ``db``.
    This is the intended behavior — surfacing the mismatch is the caller's
    responsibility, not the dispatcher's.
    """
    _register_set_count()
    _register_set_status()
    db = {}
    actions = [
        {"side": "user", "func_name": "set_count", "arguments": {"value": 7}},
        # Bogus side — dispatcher doesn't care, still applies to `db`.
        {"side": "totally-made-up", "func_name": "set_status", "arguments": {"status": "ok"}},
    ]
    result = apply_initialization_actions(domain=_TEST_DOMAIN, actions=actions, db=db)
    assert result["success"] is True
    assert db == {"count": 7, "status": "ok"}


def test_dispatch_continues_through_failures_collecting_errors():
    """All failures land in ``errors`` list; one bad action shouldn't abort
    the remaining good ones — partial application is acceptable here because
    the bridge will treat any failure as a framework error and abort the
    scenario anyway. Continuing maximizes the diagnostic info."""
    _register_set_count()
    actions = [
        {"side": "user", "func_name": "set_count", "arguments": {"value": 1}},
        {"side": "user", "func_name": "nonexistent", "arguments": {}},
        {"side": "user", "func_name": "set_count", "arguments": {"value": 2}},
    ]
    db = {}
    result = apply_initialization_actions(domain=_TEST_DOMAIN, actions=actions, db=db)
    assert result["success"] is False
    assert len(result["errors"]) == 1
    assert "nonexistent" in result["errors"][0]
    # First and third still ran (last write wins).
    assert db == {"count": 2}


def test_dispatch_none_db_returns_error():
    _register_set_count()
    result = apply_initialization_actions(
        domain=_TEST_DOMAIN,
        actions=[{"side": "user", "func_name": "set_count", "arguments": {"value": 1}}],
        db=None,
    )
    assert result["success"] is False
    assert "No db provided" in result["errors"][0]


def test_dispatch_function_raising_exception_reports_error():
    @register_initialization_function(domain=_TEST_DOMAIN)
    def explodes(db: dict, **_) -> None:
        raise ValueError("boom")

    result = apply_initialization_actions(
        domain=_TEST_DOMAIN,
        actions=[{"side": "user", "func_name": "explodes", "arguments": {}}],
        db={},
    )
    assert result["success"] is False
    assert "ValueError" in result["errors"][0]
    assert "boom" in result["errors"][0]


def test_dispatch_empty_actions_list_is_success():
    """No actions to apply ≠ failure. Bridge calls this even for scenarios
    that filtered all actions to the other side; should succeed silently."""
    result = apply_initialization_actions(
        domain=_TEST_DOMAIN, actions=[], db={}
    )
    assert result == {"success": True, "errors": []}


# ---------------------------------------------------------------------------
# 3. RTVI action factory + handler (side routing lives here)
# ---------------------------------------------------------------------------


def test_rtvi_action_metadata():
    action = create_apply_initialization_actions_action(SharedStateRef())
    assert action.service == "context"
    assert action.action == "apply_initialization_actions"
    # Two arguments: domain (optional string), actions (required array).
    # No ``side`` arg — each bot owns one DB at ``shared_state["db"]``; the
    # bridge has already filtered the upstream action list to entries
    # belonging to this bot before calling.
    names = {a.name for a in action.arguments}
    assert names == {"domain", "actions"}


def test_rtvi_handler_applies_to_state_db():
    """Each bot's ``shared_state["db"]`` is its own DB; the handler mutates
    that single dict. Whether it's "the agent DB" or "the user DB" is a
    bridge/runner-side concern."""
    _register_set_count()
    ref = SharedStateRef()
    ref.state = {"db": {}}
    action = create_apply_initialization_actions_action(ref)
    result = asyncio.run(
        action.handler(
            None,
            "context",
            {
                "domain": _TEST_DOMAIN,
                "actions": [
                    {"side": "user", "func_name": "set_count", "arguments": {"value": 3}}
                ],
            },
        )
    )
    assert result == {"success": True, "errors": []}
    assert ref.state["db"] == {"count": 3}


def test_rtvi_handler_per_action_side_field_is_informational_only():
    """The per-action ``side`` field is preserved from the upstream JSON
    for traceability but the handler does NOT consult it — the dispatcher
    operates on the single ``state["db"]`` regardless. This locks down the
    invariant that the bridge's pre-filter is the authoritative side split."""
    _register_set_count()
    _register_set_status()
    ref = SharedStateRef()
    ref.state = {"db": {}}
    action = create_apply_initialization_actions_action(ref)
    result = asyncio.run(
        action.handler(
            None,
            "context",
            {
                "domain": _TEST_DOMAIN,
                "actions": [
                    # Mixed side values would never reach a real bot (the
                    # bridge filters), but if they did, the handler still
                    # applies them all to its own state["db"]:
                    {"side": "user",  "func_name": "set_count",  "arguments": {"value": 7}},
                    {"side": "agent", "func_name": "set_status", "arguments": {"status": "ok"}},
                ],
            },
        )
    )
    assert result == {"success": True, "errors": []}
    assert ref.state["db"] == {"count": 7, "status": "ok"}


def test_rtvi_handler_missing_state_db_returns_error():
    """If update_system_prompt didn't seed the DB on this bot, the handler
    surfaces the mismatch explicitly rather than crashing inside the
    dispatcher."""
    _register_set_count()
    ref = SharedStateRef()
    ref.state = {}  # no db seeded
    action = create_apply_initialization_actions_action(ref)
    result = asyncio.run(
        action.handler(
            None,
            "context",
            {
                "domain": _TEST_DOMAIN,
                "actions": [
                    {"side": "user", "func_name": "set_count", "arguments": {"value": 1}}
                ],
            },
        )
    )
    assert result["success"] is False
    assert "shared_state['db']" in result["errors"][0]


def test_rtvi_handler_with_non_list_actions_payload_returns_error():
    ref = SharedStateRef()
    ref.state = {"db": {}}
    action = create_apply_initialization_actions_action(ref)
    result = asyncio.run(
        action.handler(
            None,
            "context",
            {"domain": _TEST_DOMAIN, "actions": "not a list"},
        )
    )
    assert result["success"] is False
    assert "must be a list" in result["errors"][0]

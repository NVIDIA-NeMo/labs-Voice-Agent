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

"""Tests for the tau2_telecom cross-side state-propagation pipeline.

Covers:
  - ``sync_telecom_state`` — the pure function that mirrors upstream's
    ``Environment.sync_tools()``. Each propagation path (line_active,
    roaming_allowed, mobile_data_usage_exceeded, payment_request set,
    payment_request paid → bill PAID) gets its own test.
  - ``apply_telecom_sync_delta`` — the bot-side applier. Verifies both
    dotted-path and ``bills[id].field`` shapes, plus the post-update
    ``_simulate_network_search`` trigger.
  - ``sync_appliers`` (generic) — registry dispatch + default applier.
  - ``Scenario.sync_state`` — default no-op vs telecom override.
"""

import copy
from typing import Dict

import pytest

import nemo_voice_agent.evaluation.scenarios.data  # noqa: F401 — triggers registration
import nemo_voice_agent.evaluation.tools  # noqa: F401 — registers telecom sync applier
from nemo_voice_agent.evaluation.scenarios import ALL_EVAL_SCENARIOS
from nemo_voice_agent.evaluation.scenarios.classes import Scenario
from nemo_voice_agent.evaluation.sync_appliers import (
    SYNC_APPLIERS,
    _default_sync_applier,
    apply_sync_delta,
    register_sync_applier,
)
from nemo_voice_agent.evaluation.tools.tau2_telecom_params import BillStatus, LineStatus
from nemo_voice_agent.evaluation.tools.tau2_telecom_sync import (
    apply_telecom_sync_delta,
    sync_telecom_state,
)


# ---------------------------------------------------------------------------
# Fixture helpers — minimal agent_db + user_db dicts the sync function
# walks. We deliberately don't load the real db.json/user_db.json
# here — keep the fixtures small + readable so the test assertions
# pinpoint exactly which propagation path each one exercises.
# ---------------------------------------------------------------------------


def _make_agent_db() -> dict:
    return {
        "plans": [
            {
                "plan_id": "P1",
                "data_limit_gb": 10.0,
                "data_refueling_price_per_gb": 2.0,
                "price_per_month": 50.0,
                "name": "Plan One",
            },
        ],
        "customers": [
            {
                "customer_id": "C1",
                "full_name": "Test User",
                "phone_number": "555-0001",
                "line_ids": ["L1"],
                "bill_ids": ["B1"],
            },
        ],
        "lines": [
            {
                "line_id": "L1",
                "phone_number": "555-0001",
                "status": LineStatus.ACTIVE.value,
                "plan_id": "P1",
                "roaming_enabled": False,
                "data_used_gb": 3.0,
                "data_refueling_gb": 0.0,
                "suspension_start_date": None,
            },
        ],
        "bills": [
            {
                "bill_id": "B1",
                "customer_id": "C1",
                "status": BillStatus.ISSUED.value,
                "total_due": 50.0,
            },
        ],
        "devices": [],
    }


def _make_user_db(phone_number: str = "555-0001") -> dict:
    return {
        "device": {
            "airplane_mode": False,
            "sim_card_missing": False,
            "sim_card_status": "Active",
            "data_enabled": True,
            "roaming_enabled": False,
            "data_saver_mode": False,
            "wifi_enabled": False,
            "wifi_connected": False,
            "wifi_ssid": None,
            "wifi_signal_strength": "none",
            "wifi_calling_enabled": False,
            "wifi_calling_mms_over_wifi": False,
            "vpn_enabled_setting": False,
            "vpn_connected": False,
            "vpn_details": None,
            "battery_level": 80,
            "network_mode_preference": "4g_5g_preferred",
            "network_connection_status": "connected",
            "network_technology_connected": "4G",
            "network_signal_strength": "excellent",
            "active_apn_settings": {
                "apn_name": "default",
                "mmsc_url": "http://mmsc.example.com",
                "reset_at_reboot": False,
            },
            "app_statuses": {},
        },
        "surroundings": {
            "name": "Test User",
            "phone_number": phone_number,
            "is_abroad": False,
            "roaming_allowed": False,
            "signal_strength": {"5G": "none", "4G": "excellent", "3G": "good", "2G": "fair"},
            "mobile_data_usage_exceeded": False,
            "line_active": True,
            "payment_request": None,
        },
    }


# ---------------------------------------------------------------------------
# 1. Pure sync function: sync_telecom_state
# ---------------------------------------------------------------------------


def test_sync_no_phone_number_is_noop():
    """Mirrors upstream's early return when ``surroundings.phone_number is None``."""
    agent_db = _make_agent_db()
    user_db = _make_user_db(phone_number=None)
    deltas = sync_telecom_state(agent_db, user_db)
    assert deltas == {"agent": {}, "user": {}}


def test_sync_no_matching_line_is_noop():
    """User's phone doesn't map to any agent-side line → early return."""
    agent_db = _make_agent_db()
    user_db = _make_user_db(phone_number="555-9999")  # not in agent_db
    deltas = sync_telecom_state(agent_db, user_db)
    assert deltas == {"agent": {}, "user": {}}


def test_sync_steady_state_produces_no_delta():
    """Both sides already in agreement → empty deltas (idempotent)."""
    agent_db = _make_agent_db()
    user_db = _make_user_db()
    # Pre-align surroundings.line_active matches line.status=Active
    sync_telecom_state(agent_db, user_db)
    # First call may produce deltas for fields not yet set; second call must
    # be a no-op since the sync just updated them.
    second = sync_telecom_state(agent_db, user_db)
    assert second == {"agent": {}, "user": {}}


def test_sync_line_status_propagates_to_line_active():
    """Suspending the agent's line flips user's ``surroundings.line_active`` to False."""
    agent_db = _make_agent_db()
    user_db = _make_user_db()
    # Steady-state first
    sync_telecom_state(agent_db, user_db)
    # Agent suspends the line
    agent_db["lines"][0]["status"] = LineStatus.SUSPENDED.value
    deltas = sync_telecom_state(agent_db, user_db)
    assert deltas["user"]["surroundings.line_active"] is False
    assert deltas["agent"] == {}
    assert user_db["surroundings"]["line_active"] is False


def test_sync_roaming_enabled_propagates_to_roaming_allowed():
    """Agent enabling roaming on the line flips user's ``surroundings.roaming_allowed`` to True."""
    agent_db = _make_agent_db()
    user_db = _make_user_db()
    sync_telecom_state(agent_db, user_db)
    agent_db["lines"][0]["roaming_enabled"] = True
    deltas = sync_telecom_state(agent_db, user_db)
    assert deltas["user"]["surroundings.roaming_allowed"] is True
    assert user_db["surroundings"]["roaming_allowed"] is True


def test_sync_data_usage_exceeded_propagates():
    """``line.data_used_gb`` over the limit flips ``mobile_data_usage_exceeded`` to True."""
    agent_db = _make_agent_db()
    user_db = _make_user_db()
    sync_telecom_state(agent_db, user_db)
    # Plan limit is 10 GB, refueling is 0 → use 11 to exceed
    agent_db["lines"][0]["data_used_gb"] = 11.0
    deltas = sync_telecom_state(agent_db, user_db)
    assert deltas["user"]["surroundings.mobile_data_usage_exceeded"] is True
    assert user_db["surroundings"]["mobile_data_usage_exceeded"] is True


def test_sync_data_refueling_unblocks_exceeded():
    """After refueling, ``data_used_gb`` may again be below limit + refueling → False."""
    agent_db = _make_agent_db()
    user_db = _make_user_db()
    # Start with usage exceeded
    agent_db["lines"][0]["data_used_gb"] = 11.0
    sync_telecom_state(agent_db, user_db)
    assert user_db["surroundings"]["mobile_data_usage_exceeded"] is True
    # Agent refuels 5 GB: 10 + 5 = 15 GB capacity > 11 used → not exceeded
    agent_db["lines"][0]["data_refueling_gb"] = 5.0
    deltas = sync_telecom_state(agent_db, user_db)
    assert deltas["user"]["surroundings.mobile_data_usage_exceeded"] is False
    assert user_db["surroundings"]["mobile_data_usage_exceeded"] is False


def test_sync_awaiting_payment_surfaces_to_user():
    """Agent's ``send_payment_request`` sets bill→AWAITING_PAYMENT; sync
    propagates to user's ``surroundings.payment_request`` so the user-sim's
    ``check_payment_request`` returns the request."""
    agent_db = _make_agent_db()
    user_db = _make_user_db()
    sync_telecom_state(agent_db, user_db)
    # Agent sends payment request: bill flips to AWAITING_PAYMENT
    agent_db["bills"][0]["status"] = BillStatus.AWAITING_PAYMENT.value
    deltas = sync_telecom_state(agent_db, user_db)
    pr = user_db["surroundings"]["payment_request"]
    assert pr is not None
    assert pr["bill_id"] == "B1"
    assert pr["amount_due"] == 50.0
    assert pr["paid"] is False
    assert deltas["user"]["surroundings.payment_request"]["bill_id"] == "B1"


def test_sync_user_payment_propagates_back_to_agent():
    """When user-sim's ``make_payment`` flips ``payment_request.paid`` to True,
    sync sets the agent's bill to PAID and clears the user's request."""
    agent_db = _make_agent_db()
    user_db = _make_user_db()
    # Setup: bill awaiting payment + user has a paid request
    agent_db["bills"][0]["status"] = BillStatus.AWAITING_PAYMENT.value
    user_db["surroundings"]["payment_request"] = {
        "bill_id": "B1",
        "amount_due": 50.0,
        "paid": True,
    }
    deltas = sync_telecom_state(agent_db, user_db)
    # Bill flipped to PAID on agent side
    assert agent_db["bills"][0]["status"] == BillStatus.PAID.value
    assert deltas["agent"]["bills[B1].status"] == BillStatus.PAID.value
    # User-side request cleared
    assert user_db["surroundings"]["payment_request"] is None
    assert deltas["user"]["surroundings.payment_request"] is None


def test_sync_does_not_resurface_paid_bill():
    """After a payment cycle, the next sync call must NOT re-surface the
    paid bill as a fresh payment_request (the bill is no longer
    AWAITING_PAYMENT)."""
    agent_db = _make_agent_db()
    user_db = _make_user_db()
    agent_db["bills"][0]["status"] = BillStatus.AWAITING_PAYMENT.value
    user_db["surroundings"]["payment_request"] = {"bill_id": "B1", "amount_due": 50.0, "paid": True}
    sync_telecom_state(agent_db, user_db)  # clears + sets PAID
    # Subsequent sync should be a no-op
    deltas = sync_telecom_state(agent_db, user_db)
    assert "surroundings.payment_request" not in deltas["user"]
    assert user_db["surroundings"]["payment_request"] is None


# ---------------------------------------------------------------------------
# 2. Bot-side applier: apply_telecom_sync_delta
# ---------------------------------------------------------------------------


def test_apply_telecom_dotted_path_sets_surroundings():
    user_db = _make_user_db()
    apply_telecom_sync_delta(user_db, {"surroundings.line_active": False})
    assert user_db["surroundings"]["line_active"] is False


def test_apply_telecom_bills_path_sets_bill_status():
    agent_db = _make_agent_db()
    apply_telecom_sync_delta(agent_db, {"bills[B1].status": BillStatus.PAID.value})
    assert agent_db["bills"][0]["status"] == BillStatus.PAID.value


def test_apply_telecom_bills_path_missing_bill_is_no_op():
    """Applier shouldn't raise when the bill_id doesn't exist —
    bridge log will warn but conversation continues."""
    agent_db = _make_agent_db()
    apply_telecom_sync_delta(agent_db, {"bills[NOT_EXIST].status": "Paid"})
    # No bill mutation, no exception
    assert agent_db["bills"][0]["status"] != BillStatus.PAID.value


def test_apply_telecom_surroundings_change_triggers_network_search():
    """When ``surroundings.*`` changes, ``_simulate_network_search`` must
    re-derive connection state. Test by flipping surroundings.line_active
    to False — network search should then force NO_SERVICE."""
    user_db = _make_user_db()
    # Pre-condition: connected
    assert user_db["device"]["network_connection_status"] == "connected"
    apply_telecom_sync_delta(user_db, {"surroundings.line_active": False})
    # Post-condition: network search saw line_active=False and forced NO_SERVICE
    assert user_db["device"]["network_connection_status"] == "no_service"
    assert user_db["device"]["network_technology_connected"] == "none"


def test_apply_telecom_agent_delta_does_not_crash_without_user_state():
    """When applying an agent-side delta (e.g. ``bills[B1].status``) to
    the agent's DB, there's no ``surroundings`` to simulate — the
    network-search guard must not raise."""
    agent_db = _make_agent_db()
    apply_telecom_sync_delta(agent_db, {"bills[B1].status": BillStatus.PAID.value})
    assert agent_db["bills"][0]["status"] == BillStatus.PAID.value


# ---------------------------------------------------------------------------
# 3. Generic applier registry (sync_appliers)
# ---------------------------------------------------------------------------


def test_default_applier_dotted_path():
    db = {"a": {"b": {"c": 1}}}
    _default_sync_applier(db, {"a.b.c": 99})
    assert db["a"]["b"]["c"] == 99


def test_default_applier_top_level_key():
    db = {"flag": False}
    _default_sync_applier(db, {"flag": True})
    assert db["flag"] is True


def test_apply_sync_delta_routes_to_registered_applier():
    """``apply_sync_delta`` should route to the telecom applier when
    domain='tau2_telecom'."""
    user_db = _make_user_db()
    apply_sync_delta(domain="tau2_telecom", db=user_db, delta={"surroundings.line_active": False})
    assert user_db["surroundings"]["line_active"] is False
    # Telecom applier also triggers network re-derivation
    assert user_db["device"]["network_connection_status"] == "no_service"


def test_apply_sync_delta_falls_back_to_default_for_unknown_domain():
    """Unknown domain → default applier (no domain-specific re-derivation)."""
    db = {"foo": {"bar": "old"}}
    apply_sync_delta(domain="some_future_domain", db=db, delta={"foo.bar": "new"})
    assert db["foo"]["bar"] == "new"


def test_apply_sync_delta_empty_delta_is_noop():
    db = {"x": 1}
    apply_sync_delta(domain="tau2_telecom", db=db, delta={})
    assert db == {"x": 1}


def test_apply_sync_delta_swallows_applier_exception():
    """Bad delta shape shouldn't crash the bot — applier exceptions are
    logged, not raised, so the conversation can continue."""

    @register_sync_applier(domain="__test_raising__")
    def _raising_applier(db, delta):
        raise RuntimeError("boom")

    try:
        apply_sync_delta(domain="__test_raising__", db={}, delta={"k": "v"})
        # Did not raise — test passes.
    finally:
        # Clean up the registered test applier so this test stays
        # idempotent across re-runs.
        SYNC_APPLIERS.pop("__test_raising__", None)


def test_register_sync_applier_collision_raises():
    """Same-domain double registration must raise at decoration time."""

    @register_sync_applier(domain="__test_collision__")
    def _first(db, delta):
        pass

    try:
        with pytest.raises(ValueError, match="Sync applier collision"):

            @register_sync_applier(domain="__test_collision__")
            def _second(db, delta):
                pass
    finally:
        SYNC_APPLIERS.pop("__test_collision__", None)


# ---------------------------------------------------------------------------
# 4. Scenario.sync_state contract
# ---------------------------------------------------------------------------


def test_base_scenario_sync_state_is_noop():
    """The default ``Scenario.sync_state`` returns empty deltas — never
    triggers cross-side sync. Single-side domains (eva / airline /
    retail) rely on this default."""
    s = Scenario(name="stub", description="d")
    deltas = s.sync_state(agent_db={}, user_db={})
    assert deltas == {"agent": {}, "user": {}}


def test_eva_airline_scenario_inherits_noop_sync_state():
    """A representative single-side scenario must not override sync_state."""
    eva_name = next(n for n in ALL_EVAL_SCENARIOS if n.startswith("eva_airline__"))
    inst = ALL_EVAL_SCENARIOS[eva_name]()
    deltas = inst.sync_state(agent_db={}, user_db={})
    assert deltas == {"agent": {}, "user": {}}


def test_tau2_telecom_scenario_overrides_sync_state():
    """Telecom scenarios MUST override ``sync_state`` — the override
    delegates to ``sync_telecom_state``. We confirm by exercising one
    propagation path end-to-end through the scenario method."""
    scenario_name = "tau2_telecom__mobile_data_issue__airplane_mode_on__data_mode_off"
    inst = ALL_EVAL_SCENARIOS[scenario_name]()
    agent_db = _make_agent_db()
    user_db = _make_user_db()
    # Toggle line.roaming_enabled and verify the scenario method
    # propagates it.
    agent_db["lines"][0]["roaming_enabled"] = True
    deltas = inst.sync_state(agent_db=agent_db, user_db=user_db)
    assert deltas["user"]["surroundings.roaming_allowed"] is True


def test_tau2_telecom_override_distinct_from_base():
    """Confirm the method-resolution check the bridge uses:
    ``type(scenario).sync_state is not Scenario.sync_state`` for telecom."""
    scenario_name = "tau2_telecom__mobile_data_issue__airplane_mode_on__data_mode_off"
    inst = ALL_EVAL_SCENARIOS[scenario_name]()
    assert type(inst).sync_state is not Scenario.sync_state
    # And the inverse: a single-side domain's scenario does NOT override.
    eva_name = next(n for n in ALL_EVAL_SCENARIOS if n.startswith("eva_airline__"))
    eva_inst = ALL_EVAL_SCENARIOS[eva_name]()
    assert type(eva_inst).sync_state is Scenario.sync_state

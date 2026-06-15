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

"""Tests for the tau2 telecom DB-state predicate ports.

Two suites:

1. **Direct-call tests** of each helper + each public ``assert_*``
   predicate against synthetic dict fixtures. Lets us pin down the
   precise truth tables without going through the dispatcher. Each
   public predicate gets at least one PASS and one FAIL case.

2. **Framework integration** — verifies the 6 predicates are
   registered under ``domain="tau2_telecom"`` in
   ``ALL_DB_STATE_PREDICATES``, and that ``evaluate_db_state_assertion``
   dispatches each correctly, returning a well-formed verdict dict.

Imports the predicates module to ensure decorator-driven registration
runs.
"""

import copy
import json
from pathlib import Path

import pytest

from nemo_voice_agent.evaluation import get_eval_data_root
from nemo_voice_agent.evaluation.db_state_predicates import (
    ALL_DB_STATE_PREDICATES,
    evaluate_db_state_assertion,
)
# Importing the module triggers @register_db_state_predicate decorators.
from nemo_voice_agent.evaluation.tools import tau2_telecom_predicates  # noqa: F401
from nemo_voice_agent.evaluation.tools.tau2_telecom_predicates import (
    _can_send_mms,
    _get_mobile_data_working,
    _run_speed_test,
    assert_can_send_mms,
    assert_data_refueling_amount,
    assert_internet_speed,
    assert_mobile_data_status,
    assert_no_overdue_bill,
    assert_service_status,
)


# ---------------------------------------------------------------------------
# Fixtures: load the canonical user_db / db that prepare_telecom.py produced
# ---------------------------------------------------------------------------


@pytest.fixture
def default_user_db() -> dict:
    """The full default user_db dict as produced by the Pydantic round-trip
    in ``scripts/prepare_tau2_data/prepare_telecom.py`` — ``surroundings``
    block + ``signal_strength`` table materialized, ``app_statuses``
    defaults present.

    Returns a fresh deep-copy each call so individual tests can mutate
    without affecting siblings."""
    with (get_eval_data_root() / "tau2_telecom" / "user_db.json").open() as f:
        return copy.deepcopy(json.load(f))


@pytest.fixture
def default_agent_db() -> dict:
    """Agent-facing DB (customers, plans, lines, bills, devices)."""
    with (get_eval_data_root() / "tau2_telecom" / "db.json").open() as f:
        return copy.deepcopy(json.load(f))


# ---------------------------------------------------------------------------
# 1. Helper: _get_mobile_data_working — truth table for the 6 falsy conditions
# ---------------------------------------------------------------------------


def test_helper_get_mobile_data_working_default_true(default_user_db):
    """Default state: 5G, GOOD signal, data enabled, not abroad, no usage
    cap → mobile data is working."""
    assert _get_mobile_data_working(default_user_db) is True


def test_helper_get_mobile_data_working_airplane_mode_off(default_user_db):
    default_user_db["device"]["airplane_mode"] = True
    assert _get_mobile_data_working(default_user_db) is False


def test_helper_get_mobile_data_working_no_signal(default_user_db):
    default_user_db["device"]["network_signal_strength"] = "none"
    assert _get_mobile_data_working(default_user_db) is False


def test_helper_get_mobile_data_working_no_service(default_user_db):
    default_user_db["device"]["network_connection_status"] = "no_service"
    assert _get_mobile_data_working(default_user_db) is False


def test_helper_get_mobile_data_working_abroad_without_roaming(default_user_db):
    """Two-part: user is abroad AND (device roaming off OR location disallows).
    Each independent condition flips the result false."""
    default_user_db["surroundings"]["is_abroad"] = True
    # roaming_enabled is False by default → fails
    assert _get_mobile_data_working(default_user_db) is False
    # Now enable device roaming — but location still disallows
    default_user_db["device"]["roaming_enabled"] = True
    default_user_db["surroundings"]["roaming_allowed"] = False
    assert _get_mobile_data_working(default_user_db) is False
    # Both enabled — now passes
    default_user_db["surroundings"]["roaming_allowed"] = True
    assert _get_mobile_data_working(default_user_db) is True


def test_helper_get_mobile_data_working_data_disabled(default_user_db):
    default_user_db["device"]["data_enabled"] = False
    assert _get_mobile_data_working(default_user_db) is False


def test_helper_get_mobile_data_working_usage_exceeded(default_user_db):
    default_user_db["surroundings"]["mobile_data_usage_exceeded"] = True
    assert _get_mobile_data_working(default_user_db) is False


# ---------------------------------------------------------------------------
# 2. Helper: _run_speed_test — bucketing
# ---------------------------------------------------------------------------


def test_helper_run_speed_test_default_excellent(default_user_db):
    """5G + GOOD signal + no VPN + no data-saver → ~220 Mbps → 'Excellent'.

    Exact value depends on the speed formula; checks the bucket only.
    """
    speed, desc = _run_speed_test(default_user_db)
    assert speed is not None
    assert speed > 100  # Excellent threshold
    assert desc == "Excellent"


def test_helper_run_speed_test_data_off_returns_none(default_user_db):
    default_user_db["device"]["data_enabled"] = False
    speed, desc = _run_speed_test(default_user_db)
    assert speed is None
    assert desc == "No Connection"


def test_helper_run_speed_test_data_saver_kills_speed(default_user_db):
    """Data saver multiplies base_speed_factor by 0.2 — 5G GOOD signal
    drops from ~220 to ~44 Mbps → 'Good' bucket."""
    default_user_db["device"]["data_saver_mode"] = True
    speed, desc = _run_speed_test(default_user_db)
    assert speed is not None
    assert 25 <= speed < 100
    assert desc == "Good"


def test_helper_run_speed_test_2g_very_poor(default_user_db):
    """2G with default GOOD signal (0.8 multiplier) → (0.1+0.4)/2 * 0.8 = 0.2 Mbps → 'Very Poor'."""
    default_user_db["device"]["network_technology_connected"] = "2G"
    speed, desc = _run_speed_test(default_user_db)
    assert speed is not None
    assert speed < 1
    assert desc == "Very Poor"


def test_helper_run_speed_test_poor_signal(default_user_db):
    """5G but POOR signal — 0.2 multiplier on the (50+500)/2 mid → 55 Mbps → 'Good'."""
    default_user_db["device"]["network_signal_strength"] = "poor"
    speed, desc = _run_speed_test(default_user_db)
    assert speed is not None
    assert 25 <= speed < 100
    assert desc == "Good"


# ---------------------------------------------------------------------------
# 3. Helper: _can_send_mms — 5 falsy conditions
# ---------------------------------------------------------------------------


def test_helper_can_send_mms_default_true(default_user_db):
    """Default: 5G, mobile data working, MMSC configured, messaging app
    has SMS+storage perms, no WiFi-calling-MMS conflict → True."""
    assert _can_send_mms(default_user_db) is True


def test_helper_can_send_mms_falls_through_when_mobile_data_off(default_user_db):
    default_user_db["device"]["data_enabled"] = False
    assert _can_send_mms(default_user_db) is False


def test_helper_can_send_mms_falls_through_on_2g(default_user_db):
    default_user_db["device"]["network_technology_connected"] = "2G"
    assert _can_send_mms(default_user_db) is False


def test_helper_can_send_mms_falls_through_when_wifi_calling_mms_active(default_user_db):
    default_user_db["device"]["wifi_calling_enabled"] = True
    default_user_db["device"]["wifi_calling_mms_over_wifi"] = True
    assert _can_send_mms(default_user_db) is False


def test_helper_can_send_mms_falls_through_on_missing_mmsc_url(default_user_db):
    default_user_db["device"]["active_apn_settings"]["mmsc_url"] = None
    assert _can_send_mms(default_user_db) is False


def test_helper_can_send_mms_falls_through_on_missing_sms_permission(default_user_db):
    default_user_db["device"]["app_statuses"]["messaging"]["permissions"]["sms"] = False
    assert _can_send_mms(default_user_db) is False


# ---------------------------------------------------------------------------
# 4. Public predicates — direct truth table
# ---------------------------------------------------------------------------


def test_assert_mobile_data_status_pass_and_fail(default_user_db):
    assert assert_mobile_data_status(default_user_db, expected_status=True) is True
    assert assert_mobile_data_status(default_user_db, expected_status=False) is False
    default_user_db["device"]["data_enabled"] = False
    assert assert_mobile_data_status(default_user_db, expected_status=True) is False
    assert assert_mobile_data_status(default_user_db, expected_status=False) is True


def test_assert_internet_speed_with_threshold_only(default_user_db):
    """Default 5G/GOOD ~220 Mbps. Numeric-only threshold."""
    assert assert_internet_speed(default_user_db, expected_speed=200) is True
    assert assert_internet_speed(default_user_db, expected_speed=300) is False


def test_assert_internet_speed_with_desc_constraint(default_user_db):
    """The data_mode_off scenario uses
    (expected_speed=200, expected_desc='excellent') — both must pass."""
    assert (
        assert_internet_speed(
            default_user_db, expected_speed=200, expected_desc="excellent"
        )
        is True
    )
    # Same speed but wrong description label → fails
    assert (
        assert_internet_speed(
            default_user_db, expected_speed=200, expected_desc="good"
        )
        is False
    )


def test_assert_internet_speed_zero_speed_when_data_off(default_user_db):
    default_user_db["device"]["data_enabled"] = False
    # speed treated as 0.0; only assert_speed<=0 would pass numerically
    assert assert_internet_speed(default_user_db, expected_speed=0.0) is True
    assert assert_internet_speed(default_user_db, expected_speed=1.0) is False


def test_assert_service_status(default_user_db):
    assert assert_service_status(default_user_db, expected_status="connected") is True
    assert assert_service_status(default_user_db, expected_status="no_service") is False
    default_user_db["device"]["network_connection_status"] = "no_service"
    assert assert_service_status(default_user_db, expected_status="no_service") is True


def test_assert_can_send_mms(default_user_db):
    assert assert_can_send_mms(default_user_db, expected_status=True) is True
    assert assert_can_send_mms(default_user_db, expected_status=False) is False


def test_assert_data_refueling_amount(default_agent_db):
    """Pull a real line and check refueling against its actual value."""
    line = default_agent_db["lines"][0]
    customer_id = next(
        c["customer_id"]
        for c in default_agent_db["customers"]
        if line["line_id"] in (c.get("line_ids") or [])
    )
    expected = line["data_refueling_gb"]
    assert (
        assert_data_refueling_amount(
            default_agent_db,
            customer_id=customer_id,
            line_id=line["line_id"],
            expected_amount=expected,
        )
        is True
    )
    # Different expected value → False
    assert (
        assert_data_refueling_amount(
            default_agent_db,
            customer_id=customer_id,
            line_id=line["line_id"],
            expected_amount=expected + 5.0,
        )
        is False
    )


def test_assert_data_refueling_amount_missing_line_returns_false(default_agent_db):
    """Predicates-as-bool semantic: missing line returns False, not raises."""
    assert (
        assert_data_refueling_amount(
            default_agent_db,
            customer_id="DOES_NOT_EXIST",
            line_id="L_FAKE",
            expected_amount=2.0,
        )
        is False
    )


def test_assert_no_overdue_bill_missing_bill_is_true(default_agent_db):
    """Upstream semantic: a non-existent bill ID counts as 'no overdue' = True."""
    assert (
        assert_no_overdue_bill(default_agent_db, overdue_bill_id="B_DOES_NOT_EXIST")
        is True
    )


def test_assert_no_overdue_bill_paid_bill_is_true(default_agent_db):
    """An existing bill whose status is PAID also counts as no overdue."""
    # Find a bill, force-mark it PAID
    bill = default_agent_db["bills"][0]
    bill["status"] = "Paid"
    assert assert_no_overdue_bill(default_agent_db, overdue_bill_id=bill["bill_id"]) is True


def test_assert_no_overdue_bill_existing_overdue_is_false(default_agent_db):
    bill = default_agent_db["bills"][0]
    bill["status"] = "Overdue"
    assert (
        assert_no_overdue_bill(default_agent_db, overdue_bill_id=bill["bill_id"]) is False
    )


# ---------------------------------------------------------------------------
# 5. Framework integration — registration + dispatch end-to-end
# ---------------------------------------------------------------------------


def test_all_six_predicates_registered():
    """The 6 expected predicate names must be present in
    ALL_DB_STATE_PREDICATES['tau2_telecom']. Catches if a decorator was
    accidentally removed during a future refactor."""
    bucket = ALL_DB_STATE_PREDICATES.get("tau2_telecom", {})
    expected = {
        "assert_mobile_data_status",
        "assert_internet_speed",
        "assert_service_status",
        "assert_can_send_mms",
        "assert_data_refueling_amount",
        "assert_no_overdue_bill",
    }
    missing = expected - set(bucket)
    assert not missing, f"Missing predicates: {missing}"


def test_dispatch_via_evaluate_db_state_assertion_user_side(default_user_db):
    """Dispatch a user-side predicate through the evaluator. Mirrors
    what ``runner.py`` does at scenario-completion time for any
    user-side db_state_assertion."""
    assertion = {
        "side": "user",
        "func_name": "assert_mobile_data_status",
        "arguments": {"expected_status": True},
        "assert_value": True,
    }
    verdict = evaluate_db_state_assertion(
        domain="tau2_telecom",
        assertion=assertion,
        db=None,
        user_db=default_user_db,
    )
    assert verdict["passed"] is True
    assert verdict["error"] is None
    assert verdict["actual"] is True


def test_dispatch_via_evaluate_db_state_assertion_agent_side(default_agent_db):
    """Dispatch an agent-side predicate. Validates that the side->db
    routing in evaluate_db_state_assertion correctly picks the agent DB."""
    assertion = {
        "side": "agent",
        "func_name": "assert_no_overdue_bill",
        "arguments": {"overdue_bill_id": "B_DOES_NOT_EXIST"},
        "assert_value": True,
    }
    verdict = evaluate_db_state_assertion(
        domain="tau2_telecom",
        assertion=assertion,
        db=default_agent_db,
        user_db=None,
    )
    assert verdict["passed"] is True
    assert verdict["actual"] is True


def test_dispatch_with_assert_internet_speed_full_data_mode_off_shape(default_user_db):
    """Exercise the exact assertion shape from upstream's
    ``[mobile_data_issue]data_mode_off`` task — ``{expected_speed=200,
    expected_desc='excellent'}``. Default user_db at 5G/GOOD passes."""
    assertion = {
        "side": "user",
        "func_name": "assert_internet_speed",
        "arguments": {"expected_speed": 200, "expected_desc": "excellent"},
        "assert_value": True,
    }
    verdict = evaluate_db_state_assertion(
        domain="tau2_telecom",
        assertion=assertion,
        db=None,
        user_db=default_user_db,
    )
    assert verdict["passed"] is True

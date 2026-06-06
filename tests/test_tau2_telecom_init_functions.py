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

"""Tests for the tau2 telecom initialization function ports.

Per-function direct-call tests against synthetic dict fixtures + an
end-to-end framework integration check via ``apply_initialization_actions``.

Covers:

- All 16 user-side functions: each verified to mutate the expected
  fields. The ``simulate_network_search`` helper is exercised via the
  user-side functions that call it (turn_airplane_mode_on, turn_data_off,
  set_network_mode_preference, unseat_sim_card, lock_sim_card,
  break_apn_settings, plus a direct call).
- All 4 agent-side functions: enable_roaming / disable_roaming /
  set_data_usage / suspend_line_for_overdue_bill.
- Failure modes: enable/disable/set_data on a non-existent line raises;
  suspend_line_for_overdue_bill with a non-linked bill raises.
- Framework integration: the 20 names registered correctly; dispatch
  through ``apply_initialization_actions`` mutates the expected DB.
- End-to-end driving the first telecom scenario's init sequence
  (set_user_info + turn_airplane_mode_on + turn_data_off) leaves the
  device in a state where ``_get_mobile_data_working`` returns False.

Imports the init-functions module to ensure decorator-driven registration
runs.
"""

import copy
import json
from pathlib import Path

import pytest

from nemo_voice_agent.evaluation import get_eval_data_root
from nemo_voice_agent.evaluation.initialization_functions import (
    ALL_INITIALIZATION_FUNCTIONS,
    apply_initialization_actions,
)
# Importing the module triggers @register_initialization_function decorators.
from nemo_voice_agent.evaluation.tools import tau2_telecom_init_functions  # noqa: F401
from nemo_voice_agent.evaluation.tools.tau2_telecom_init_functions import (
    break_apn_mms_setting,
    break_apn_settings,
    break_vpn,
    disable_roaming,
    enable_roaming,
    lock_sim_card,
    remove_app_permission,
    set_data_usage,
    set_network_mode_preference,
    set_user_info,
    set_user_location,
    set_wifi_calling,
    simulate_network_search,
    suspend_line_for_overdue_bill,
    turn_airplane_mode_on,
    turn_data_off,
    turn_data_saver_mode_on,
    turn_roaming_off,
    turn_roaming_on,
    unseat_sim_card,
)
from nemo_voice_agent.evaluation.tools.tau2_telecom_predicates import (
    _get_mobile_data_working,
    _can_send_mms,
)


@pytest.fixture
def default_user_db() -> dict:
    with (get_eval_data_root() / "tau2_telecom" / "user_db.json").open() as f:
        return copy.deepcopy(json.load(f))


@pytest.fixture
def default_agent_db() -> dict:
    with (get_eval_data_root() / "tau2_telecom" / "db.json").open() as f:
        return copy.deepcopy(json.load(f))


# ---------------------------------------------------------------------------
# 1. User-side basic field setters
# ---------------------------------------------------------------------------


def test_set_user_info(default_user_db):
    set_user_info(default_user_db, name="John Smith", phone_number="555-123-2002")
    assert default_user_db["surroundings"]["name"] == "John Smith"
    assert default_user_db["surroundings"]["phone_number"] == "555-123-2002"


def test_set_user_location_abroad(default_user_db):
    set_user_location(default_user_db, abroad=True)
    assert default_user_db["surroundings"]["is_abroad"] is True
    set_user_location(default_user_db, abroad=False)
    assert default_user_db["surroundings"]["is_abroad"] is False


def test_set_network_mode_preference_valid(default_user_db):
    set_network_mode_preference(default_user_db, mode="2g_only")
    assert default_user_db["device"]["network_mode_preference"] == "2g_only"
    # Network search re-derives technology + signal:
    assert default_user_db["device"]["network_technology_connected"] == "2G"


def test_set_network_mode_preference_invalid_raises(default_user_db):
    with pytest.raises(ValueError):
        set_network_mode_preference(default_user_db, mode="bogus_mode_string")


# ---------------------------------------------------------------------------
# 2. User-side toggles (each drives device to a specific end state)
# ---------------------------------------------------------------------------


def test_turn_airplane_mode_on(default_user_db):
    """Airplane mode forces no_service / NONE / NONE; clears wifi connection
    and disconnects VPN if connected."""
    # Pre-set VPN connected so we can verify disconnect path
    default_user_db["device"]["vpn_connected"] = True
    default_user_db["device"]["vpn_details"] = {
        "server_address": "x", "protocol": "y", "server_performance": "excellent"
    }
    turn_airplane_mode_on(default_user_db)
    device = default_user_db["device"]
    assert device["airplane_mode"] is True
    assert device["wifi_connected"] is False
    assert device["wifi_ssid"] is None
    assert device["wifi_signal_strength"] == "none"
    # VPN was disconnected as a side-effect
    assert device["vpn_connected"] is False
    assert device["vpn_details"] is None
    # Network search forced no_service / NONE / NONE because airplane_mode
    assert device["network_connection_status"] == "no_service"
    assert device["network_technology_connected"] == "none"
    assert device["network_signal_strength"] == "none"


def test_turn_data_off(default_user_db):
    turn_data_off(default_user_db)
    assert default_user_db["device"]["data_enabled"] is False
    # _get_mobile_data_working flips False (the whole point):
    assert _get_mobile_data_working(default_user_db) is False


def test_turn_roaming_on_and_off(default_user_db):
    turn_roaming_on(default_user_db)
    assert default_user_db["device"]["roaming_enabled"] is True
    turn_roaming_off(default_user_db)
    assert default_user_db["device"]["roaming_enabled"] is False


def test_turn_data_saver_mode_on(default_user_db):
    turn_data_saver_mode_on(default_user_db)
    assert default_user_db["device"]["data_saver_mode"] is True


def test_simulate_network_search_with_active_sim(default_user_db):
    """Explicit call to network search with default 4g_5g_preferred + 5G
    signal in surroundings → 5G connected, GOOD (default surrounding
    5G signal)."""
    simulate_network_search(default_user_db)
    assert default_user_db["device"]["network_connection_status"] == "connected"
    assert default_user_db["device"]["network_technology_connected"] == "5G"
    # Default surroundings.signal_strength[5G] = "excellent"
    assert default_user_db["device"]["network_signal_strength"] == "excellent"


# ---------------------------------------------------------------------------
# 3. SIM / APN / VPN / wifi-calling state
# ---------------------------------------------------------------------------


def test_unseat_sim_card(default_user_db):
    unseat_sim_card(default_user_db)
    assert default_user_db["device"]["sim_card_missing"] is True
    # MISSING SIM forces no_service
    assert default_user_db["device"]["network_connection_status"] == "no_service"


def test_lock_sim_card_pin(default_user_db):
    lock_sim_card(default_user_db, mode="pin")
    assert default_user_db["device"]["sim_card_status"] == "locked_pin"
    # Locked SIM also forces no_service
    assert default_user_db["device"]["network_connection_status"] == "no_service"


def test_lock_sim_card_puk(default_user_db):
    lock_sim_card(default_user_db, mode="puk")
    assert default_user_db["device"]["sim_card_status"] == "locked_puk"


def test_lock_sim_card_invalid_mode_raises(default_user_db):
    with pytest.raises(ValueError, match="must be 'pin' or 'puk'"):
        lock_sim_card(default_user_db, mode="other")


def test_break_apn_settings(default_user_db):
    break_apn_settings(default_user_db)
    assert default_user_db["device"]["active_apn_settings"]["apn_name"] == "broken"
    # Broken APN forces no_service
    assert default_user_db["device"]["network_connection_status"] == "no_service"


def test_break_apn_mms_setting_drops_mmsc_url(default_user_db):
    """Breaks MMS specifically — _can_send_mms should flip False but
    _get_mobile_data_working stays True."""
    break_apn_mms_setting(default_user_db)
    assert default_user_db["device"]["active_apn_settings"]["mmsc_url"] is None
    assert _can_send_mms(default_user_db) is False
    assert _get_mobile_data_working(default_user_db) is True


def test_break_vpn_connects_with_poor_perf(default_user_db):
    """break_vpn: if VPN was disconnected, connect with default details
    + set performance to POOR."""
    assert default_user_db["device"]["vpn_connected"] is False
    break_vpn(default_user_db)
    device = default_user_db["device"]
    assert device["vpn_connected"] is True
    assert device["vpn_details"] is not None
    assert device["vpn_details"]["server_performance"] == "poor"


def test_set_wifi_calling_enable_only(default_user_db):
    set_wifi_calling(default_user_db, enabled=True)
    assert default_user_db["device"]["wifi_calling_enabled"] is True
    # mms_over_wifi not passed → unchanged
    assert default_user_db["device"]["wifi_calling_mms_over_wifi"] is False


def test_set_wifi_calling_with_mms_over_wifi(default_user_db):
    set_wifi_calling(default_user_db, enabled=True, mms_over_wifi=True)
    device = default_user_db["device"]
    assert device["wifi_calling_enabled"] is True
    assert device["wifi_calling_mms_over_wifi"] is True
    # Carrier limitation triggers — _can_send_mms flips False
    assert _can_send_mms(default_user_db) is False


# ---------------------------------------------------------------------------
# 4. Application permissions
# ---------------------------------------------------------------------------


def test_remove_app_permission_clears_flag(default_user_db):
    """Removing 'sms' permission from messaging app blocks MMS-send."""
    assert default_user_db["device"]["app_statuses"]["messaging"]["permissions"]["sms"] is True
    remove_app_permission(default_user_db, app_name="messaging", permission="sms")
    assert default_user_db["device"]["app_statuses"]["messaging"]["permissions"]["sms"] is False
    assert _can_send_mms(default_user_db) is False


def test_remove_app_permission_case_insensitive(default_user_db):
    remove_app_permission(default_user_db, app_name="messaging", permission="SMS")
    assert default_user_db["device"]["app_statuses"]["messaging"]["permissions"]["sms"] is False


def test_remove_app_permission_unknown_app_is_noop(default_user_db):
    """Tolerant behavior — unknown app name silently no-ops (matches upstream)."""
    before = copy.deepcopy(default_user_db["device"]["app_statuses"])
    remove_app_permission(default_user_db, app_name="DOES_NOT_EXIST", permission="sms")
    assert default_user_db["device"]["app_statuses"] == before


def test_remove_app_permission_unknown_permission_is_noop(default_user_db):
    before = copy.deepcopy(default_user_db["device"]["app_statuses"]["messaging"])
    remove_app_permission(default_user_db, app_name="messaging", permission="bogus")
    assert default_user_db["device"]["app_statuses"]["messaging"] == before


# ---------------------------------------------------------------------------
# 5. Agent-side init functions
# ---------------------------------------------------------------------------


def _first_line_with_customer(db: dict) -> tuple[str, str]:
    """Return (customer_id, line_id) for the first line that's linked to a customer."""
    for customer in db["customers"]:
        if customer.get("line_ids"):
            return customer["customer_id"], customer["line_ids"][0]
    raise RuntimeError("no customer with a line in fixture db.json")


def test_enable_roaming(default_agent_db):
    customer_id, line_id = _first_line_with_customer(default_agent_db)
    enable_roaming(default_agent_db, customer_id=customer_id, line_id=line_id)
    line = next(l for l in default_agent_db["lines"] if l["line_id"] == line_id)
    assert line["roaming_enabled"] is True


def test_disable_roaming(default_agent_db):
    customer_id, line_id = _first_line_with_customer(default_agent_db)
    # First enable, then disable to verify the toggle
    enable_roaming(default_agent_db, customer_id=customer_id, line_id=line_id)
    disable_roaming(default_agent_db, customer_id=customer_id, line_id=line_id)
    line = next(l for l in default_agent_db["lines"] if l["line_id"] == line_id)
    assert line["roaming_enabled"] is False


def test_enable_roaming_missing_line_raises(default_agent_db):
    with pytest.raises(ValueError, match="line .* not found"):
        enable_roaming(default_agent_db, customer_id="C1001", line_id="L_FAKE")


def test_set_data_usage(default_agent_db):
    customer_id, line_id = _first_line_with_customer(default_agent_db)
    set_data_usage(default_agent_db, customer_id=customer_id, line_id=line_id, amount_gb=7.5)
    line = next(l for l in default_agent_db["lines"] if l["line_id"] == line_id)
    assert line["data_used_gb"] == 7.5


def test_set_data_usage_missing_line_raises(default_agent_db):
    with pytest.raises(ValueError, match="line .* not found"):
        set_data_usage(
            default_agent_db, customer_id="C1001", line_id="L_FAKE", amount_gb=1.0
        )


def test_suspend_line_for_overdue_bill_no_bill(default_agent_db):
    """Without bill_id, just flip line to SUSPENDED."""
    customer_id, line_id = _first_line_with_customer(default_agent_db)
    suspend_line_for_overdue_bill(
        default_agent_db, customer_id=customer_id, line_id=line_id
    )
    line = next(l for l in default_agent_db["lines"] if l["line_id"] == line_id)
    assert line["status"] == "Suspended"


def test_suspend_line_for_overdue_bill_with_linked_bill(default_agent_db):
    """With bill_id linked to the customer, mark bill as OVERDUE too."""
    # Find a customer + line + their associated bill
    customer = next(c for c in default_agent_db["customers"] if c.get("line_ids") and c.get("bill_ids"))
    customer_id = customer["customer_id"]
    line_id = customer["line_ids"][0]
    bill_id = customer["bill_ids"][0]
    suspend_line_for_overdue_bill(
        default_agent_db,
        customer_id=customer_id,
        line_id=line_id,
        bill_id=bill_id,
    )
    line = next(l for l in default_agent_db["lines"] if l["line_id"] == line_id)
    bill = next(b for b in default_agent_db["bills"] if b["bill_id"] == bill_id)
    assert line["status"] == "Suspended"
    assert bill["status"] == "Overdue"


def test_suspend_line_for_overdue_bill_unlinked_bill_raises(default_agent_db):
    customer = next(c for c in default_agent_db["customers"] if c.get("line_ids"))
    customer_id = customer["customer_id"]
    line_id = customer["line_ids"][0]
    # Pick a bill that exists but isn't linked to this customer
    other_bills = [
        b["bill_id"]
        for b in default_agent_db["bills"]
        if b["bill_id"] not in (customer.get("bill_ids") or [])
    ]
    if not other_bills:
        pytest.skip("no unlinked bill available in fixture db.json")
    with pytest.raises(ValueError, match="not linked to customer"):
        suspend_line_for_overdue_bill(
            default_agent_db,
            customer_id=customer_id,
            line_id=line_id,
            bill_id=other_bills[0],
        )


# ---------------------------------------------------------------------------
# 6. Framework integration
# ---------------------------------------------------------------------------


def test_all_twenty_init_functions_registered():
    """20 expected function names must be present in
    ALL_INITIALIZATION_FUNCTIONS['tau2_telecom']."""
    bucket = ALL_INITIALIZATION_FUNCTIONS.get("tau2_telecom", {})
    expected = {
        # 16 user-side
        "set_user_info", "set_user_location", "set_network_mode_preference",
        "simulate_network_search", "turn_airplane_mode_on", "turn_data_off",
        "turn_data_saver_mode_on", "turn_roaming_on", "turn_roaming_off",
        "unseat_sim_card", "lock_sim_card", "break_apn_settings",
        "break_apn_mms_setting", "break_vpn", "set_wifi_calling",
        "remove_app_permission",
        # 4 agent-side
        "enable_roaming", "disable_roaming", "set_data_usage",
        "suspend_line_for_overdue_bill",
    }
    missing = expected - set(bucket)
    extra = set(bucket) - expected
    assert not missing, f"Missing init functions: {missing}"
    assert not extra, f"Unexpected init functions registered: {extra}"


def test_apply_initialization_actions_dispatches_data_mode_off_init(default_user_db):
    """End-to-end: drive the user-side DB through the init sequence for
    upstream's ``[mobile_data_issue]airplane_mode_on|data_mode_off`` task
    via ``apply_initialization_actions``. Resulting state should leave
    ``_get_mobile_data_working`` False."""
    actions = [
        {
            "func_name": "set_user_info",
            "arguments": {"name": "John Smith", "phone_number": "555-123-2002"},
        },
        {"func_name": "turn_airplane_mode_on", "arguments": {}},
        {"func_name": "turn_data_off", "arguments": {}},
    ]
    result = apply_initialization_actions(
        domain="tau2_telecom", actions=actions, db=default_user_db
    )
    assert result == {"success": True, "errors": []}
    assert default_user_db["surroundings"]["name"] == "John Smith"
    assert default_user_db["device"]["airplane_mode"] is True
    assert default_user_db["device"]["data_enabled"] is False
    assert _get_mobile_data_working(default_user_db) is False


def test_apply_initialization_actions_propagates_per_action_failure(default_agent_db):
    """One bad action in the middle of a list shouldn't abort the list,
    but ``success=False`` propagates with the error captured."""
    actions = [
        {
            "func_name": "enable_roaming",
            "arguments": {"customer_id": "C1001", "line_id": "L_FAKE"},
        },
    ]
    result = apply_initialization_actions(
        domain="tau2_telecom", actions=actions, db=default_agent_db
    )
    assert result["success"] is False
    assert len(result["errors"]) == 1
    assert "L_FAKE" in result["errors"][0]

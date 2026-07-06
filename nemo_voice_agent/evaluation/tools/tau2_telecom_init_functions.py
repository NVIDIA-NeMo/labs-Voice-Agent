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
#
# Adapted from https://github.com/sierra-research/tau2-bench/tree/voice-user-sim-v1.0
#   src/tau2/domains/telecom/user_tools.py — user-side mutation methods
#       (set_user_info, set_user_location, turn_*_on/off, lock_sim_card,
#       unseat_sim_card, remove_app_permission, break_apn_settings,
#       break_apn_mms_setting, break_vpn, set_wifi_calling,
#       set_network_mode_preference, simulate_network_search).
#   src/tau2/domains/telecom/tools.py      — agent-side mutation methods
#       (enable_roaming, disable_roaming, set_data_usage,
#       suspend_line_for_overdue_bill).
#
# Translation notes vs upstream:
#   - Upstream uses Pydantic model attribute mutation (``self.device.x = y``);
#     we mutate plain dict fields (``db["device"]["x"] = y``). The Pydantic
#     round-trip in ``scripts/prepare_tau2_data/prepare_telecom.py``
#     materializes default fields so all keys already exist; init functions
#     assume the dict shape is well-formed.
#   - Enum values are strings (e.g. ``"4g_5g_preferred"``) because the
#     Pydantic round-trip serializes enums via their ``str.value``. Init
#     functions write string values directly; predicates compare via
#     ``Enum.value`` lookups.
#   - Upstream wraps the public ``turn_X_on/off`` methods around private
#     ``_toggle_X`` helpers that flip-then-correct-if-wrong (double-toggle
#     pattern). For our init-time usage the public methods are always
#     called from a clean default state with intent to drive to a specific
#     end state, so the simpler `set the field to True/False` form is
#     equivalent and clearer. Where upstream's tools.py would call the
#     toggle helper to bounce the device through SEARCHING during a live
#     conversation, we just write the final state.
#   - Some upstream methods also call ``simulate_network_search()`` after
#     mutating to recompute connection_status / signal_strength from the
#     new SIM/airplane/preference state. We port this helper as
#     ``_simulate_network_search(db)`` and call it from the same set of
#     init functions upstream does — preserving the post-mutation derived
#     state so predicates observe what they would on a live device.

"""Telecom initialization-function ports for the ``initialization_actions`` framework.

Twenty init functions total — sixteen user-side and four agent-side. Each
is registered under ``domain="tau2_telecom"`` in the flat
``ALL_INITIALIZATION_FUNCTIONS`` registry; the bot-side dispatcher invokes
them by ``func_name`` only (functions are side-agnostic, the bot picks
the DB based on which bot it is).

User-side init functions (mutate the user bot's
``shared_state["db"]`` = ``TelecomUserDB``):

  - ``set_user_info(name, phone_number)`` — name + phone in surroundings.
  - ``set_user_location(abroad)`` — toggles is_abroad flag.
  - ``set_network_mode_preference(mode)`` — preferred cellular tech;
    triggers a network search.
  - ``turn_airplane_mode_on()`` / ``turn_data_off()`` /
    ``turn_data_saver_mode_on()`` / ``turn_roaming_on()`` /
    ``turn_roaming_off()`` — direct field setters with a follow-up
    ``_simulate_network_search`` where upstream does the same.
  - ``simulate_network_search()`` — recomputes derived connection
    status. Used as both an init action and a helper.
  - ``unseat_sim_card()`` / ``lock_sim_card(mode)`` — SIM state.
  - ``remove_app_permission(app_name, permission)`` — clears a single
    permission flag.
  - ``break_apn_settings()`` / ``break_apn_mms_setting()`` —
    deliberately bad APN config (test setup for issue scenarios).
  - ``break_vpn()`` — connect VPN with POOR performance.
  - ``set_wifi_calling(enabled, mms_over_wifi=None)`` — wifi calling +
    optional MMS-over-wifi flag.

Agent-side init functions (mutate the agent bot's
``shared_state["db"]`` = ``TelecomDB``):

  - ``enable_roaming(customer_id, line_id)`` / ``disable_roaming(...)``
    — flips a line's ``roaming_enabled`` flag.
  - ``set_data_usage(customer_id, line_id, amount_gb)`` — overwrites
    ``data_used_gb`` on a line.
  - ``suspend_line_for_overdue_bill(customer_id, line_id, bill_id, ...)``
    — flips a line to SUSPENDED and ties an overdue bill.

Each function takes ``(db: dict, **arguments) -> None`` and mutates the
passed DB dict in place. Errors propagate; the dispatcher catches them
and returns a structured failure verdict.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from nemo_voice_agent.evaluation.initialization_functions import register_initialization_function
from nemo_voice_agent.evaluation.tools.tau2_telecom_params import (
    APNNames,
    BillStatus,
    LineStatus,
    NetworkModePreference,
    NetworkStatus,
    NetworkTechnology,
    PerformanceLevel,
    SignalStrength,
    SimStatus,
)
from nemo_voice_agent.evaluation.tools.tau2_telecom_predicates import (
    _can_send_mms,
    _get_customer_by_id,
    _get_target_line,
)


# Default VPN details used when ``break_vpn`` connects a previously-disconnected
# VPN. Mirrors upstream's ``TelecomUserTools.default_vpn_details``.
_DEFAULT_VPN_DETAILS = {
    "server_address": "192.168.1.1",
    "protocol": "OpenVPN",
    "server_performance": PerformanceLevel.EXCELLENT.value,
}


# =============================================================================
# Private helpers (not registered with the init dispatcher)
# =============================================================================


def _check_sim_status(db: Dict[str, Any]) -> str:
    """Mirror of upstream's ``TelecomUserTools._check_sim_status``."""
    device = db["device"]
    if device["sim_card_missing"]:
        return SimStatus.MISSING.value
    return device["sim_card_status"]


def _simulate_network_search(db: Dict[str, Any]) -> None:
    """Recompute derived connection state from SIM + airplane + preference.

    Mirrors upstream's ``TelecomUserTools.simulate_network_search``. Called
    after init functions that affect SIM state, airplane mode, network
    preference, APN settings, or roaming. Updates three fields:
    ``network_connection_status``, ``network_technology_connected``,
    ``network_signal_strength``.
    """
    device = db["device"]
    surroundings = db["surroundings"]
    signal_strengths_per_tech = surroundings["signal_strength"]
    sim_status = _check_sim_status(db)

    if sim_status == SimStatus.ACTIVE.value:
        device["network_connection_status"] = NetworkStatus.CONNECTED.value
        pref = device["network_mode_preference"]
        if pref == NetworkModePreference.FOUR_G_5G_PREFERRED.value:
            five_g_signal = signal_strengths_per_tech.get(NetworkTechnology.FIVE_G.value, SignalStrength.NONE.value)
            if five_g_signal == SignalStrength.NONE.value:
                device["network_technology_connected"] = NetworkTechnology.FOUR_G.value
                device["network_signal_strength"] = signal_strengths_per_tech.get(
                    NetworkTechnology.FOUR_G.value, SignalStrength.NONE.value
                )
            else:
                device["network_technology_connected"] = NetworkTechnology.FIVE_G.value
                device["network_signal_strength"] = five_g_signal
        elif pref == NetworkModePreference.FOUR_G_ONLY.value:
            device["network_technology_connected"] = NetworkTechnology.FOUR_G.value
            device["network_signal_strength"] = signal_strengths_per_tech.get(
                NetworkTechnology.FOUR_G.value, SignalStrength.NONE.value
            )
        elif pref == NetworkModePreference.THREE_G_ONLY.value:
            device["network_technology_connected"] = NetworkTechnology.THREE_G.value
            device["network_signal_strength"] = signal_strengths_per_tech.get(
                NetworkTechnology.THREE_G.value, SignalStrength.NONE.value
            )
        elif pref == NetworkModePreference.TWO_G_ONLY.value:
            device["network_technology_connected"] = NetworkTechnology.TWO_G.value
            device["network_signal_strength"] = signal_strengths_per_tech.get(
                NetworkTechnology.TWO_G.value, SignalStrength.NONE.value
            )
        else:
            # Default fallback (matches upstream)
            device["network_technology_connected"] = NetworkTechnology.FOUR_G.value
            device["network_signal_strength"] = signal_strengths_per_tech.get(
                NetworkTechnology.FOUR_G.value, SignalStrength.NONE.value
            )
    else:
        # MISSING / LOCKED_PIN / LOCKED_PUK / unknown — same shape
        device["network_connection_status"] = NetworkStatus.NO_SERVICE.value
        device["network_technology_connected"] = NetworkTechnology.NONE.value
        device["network_signal_strength"] = SignalStrength.NONE.value

    # No connection if airplane mode is on (overrides SIM-derived state)
    if device["airplane_mode"]:
        device["network_connection_status"] = NetworkStatus.NO_SERVICE.value
        device["network_technology_connected"] = NetworkTechnology.NONE.value
        device["network_signal_strength"] = SignalStrength.NONE.value

    # No connection if APN is broken
    if device["active_apn_settings"]["apn_name"] == APNNames.BROKEN.value:
        device["network_connection_status"] = NetworkStatus.NO_SERVICE.value
        device["network_technology_connected"] = NetworkTechnology.NONE.value
        device["network_signal_strength"] = SignalStrength.NONE.value

    # No connection if the user's line is inactive (e.g. suspended/closed
    # upstream — surroundings["line_active"] is the mirror flag on the
    # user side)
    if not surroundings["line_active"]:
        device["network_connection_status"] = NetworkStatus.NO_SERVICE.value
        device["network_technology_connected"] = NetworkTechnology.NONE.value
        device["network_signal_strength"] = SignalStrength.NONE.value


def _disconnect_vpn(db: Dict[str, Any]) -> None:
    """Mirror of upstream's ``_disconnect_vpn``: clear connection + details."""
    device = db["device"]
    if not device["vpn_connected"]:
        return
    device["vpn_connected"] = False
    device["vpn_details"] = None


def _connect_vpn(db: Dict[str, Any]) -> None:
    """Mirror of upstream's ``_connect_vpn``: set default VPN details if not already connected."""
    device = db["device"]
    if device["vpn_connected"]:
        return
    device["vpn_connected"] = True
    device["vpn_details"] = dict(_DEFAULT_VPN_DETAILS)


# =============================================================================
# User-side init functions registered with the dispatcher
# =============================================================================


@register_initialization_function(domain="tau2_telecom")
def set_user_info(db: Dict[str, Any], name: str, phone_number: str) -> None:
    """Set the user's name and phone number on surroundings."""
    db["surroundings"]["name"] = name
    db["surroundings"]["phone_number"] = phone_number


@register_initialization_function(domain="tau2_telecom")
def set_user_location(db: Dict[str, Any], abroad: bool) -> None:
    """Toggle the user's abroad-or-home flag on surroundings."""
    db["surroundings"]["is_abroad"] = abroad


@register_initialization_function(domain="tau2_telecom")
def set_network_mode_preference(db: Dict[str, Any], mode: str) -> None:
    """Set preferred cellular network mode; triggers a network search.

    ``mode`` is the string value of a ``NetworkModePreference`` enum
    (e.g. ``"4g_5g_preferred"``, ``"2g_only"``).
    """
    # Validate via enum lookup — raises ValueError on invalid input,
    # which the dispatcher converts to an init-time failure.
    db["device"]["network_mode_preference"] = NetworkModePreference(mode).value
    _simulate_network_search(db)


@register_initialization_function(domain="tau2_telecom")
def simulate_network_search(db: Dict[str, Any]) -> None:
    """Recompute network_connection_status / technology / signal_strength
    from current SIM + airplane + preference + APN state.

    Exposed as a public init action because some upstream tasks use it
    standalone (e.g. after manual surroundings tweaks) without bundling
    it into another mutation."""
    _simulate_network_search(db)


@register_initialization_function(domain="tau2_telecom")
def turn_airplane_mode_on(db: Dict[str, Any]) -> None:
    """Enable Airplane Mode. Disconnects wifi, VPN, and clears wifi SSID;
    triggers a network search."""
    device = db["device"]
    device["airplane_mode"] = True
    device["wifi_connected"] = False
    device["wifi_ssid"] = None
    device["wifi_signal_strength"] = SignalStrength.NONE.value
    if device["vpn_connected"]:
        _disconnect_vpn(db)
    _simulate_network_search(db)


@register_initialization_function(domain="tau2_telecom")
def unseat_sim_card(db: Dict[str, Any]) -> None:
    """Mark the SIM as physically missing; triggers a network search.

    The network-search rebuild sees ``sim_card_missing=True`` and forces
    NO_SERVICE / NONE / NONE on connection_status / tech / signal.
    """
    db["device"]["sim_card_missing"] = True
    _simulate_network_search(db)


@register_initialization_function(domain="tau2_telecom")
def lock_sim_card(db: Dict[str, Any], mode: str) -> None:
    """Lock the SIM card. ``mode`` must be ``"pin"`` or ``"puk"``."""
    if mode == "pin":
        db["device"]["sim_card_status"] = SimStatus.LOCKED_PIN.value
    elif mode == "puk":
        db["device"]["sim_card_status"] = SimStatus.LOCKED_PUK.value
    else:
        raise ValueError(f"lock_sim_card: mode must be 'pin' or 'puk', got {mode!r}")
    _simulate_network_search(db)


@register_initialization_function(domain="tau2_telecom")
def turn_data_off(db: Dict[str, Any]) -> None:
    """Turn the master Mobile Data switch OFF and re-derive network state."""
    db["device"]["data_enabled"] = False
    _simulate_network_search(db)


@register_initialization_function(domain="tau2_telecom")
def turn_roaming_on(db: Dict[str, Any]) -> None:
    """Enable Data Roaming on the device; triggers a network search."""
    db["device"]["roaming_enabled"] = True
    _simulate_network_search(db)


@register_initialization_function(domain="tau2_telecom")
def turn_roaming_off(db: Dict[str, Any]) -> None:
    """Disable Data Roaming on the device; triggers a network search."""
    db["device"]["roaming_enabled"] = False
    _simulate_network_search(db)


@register_initialization_function(domain="tau2_telecom")
def turn_data_saver_mode_on(db: Dict[str, Any]) -> None:
    """Enable Data Saver mode (reduces simulated speed by 80%)."""
    db["device"]["data_saver_mode"] = True


@register_initialization_function(domain="tau2_telecom")
def break_apn_settings(db: Dict[str, Any]) -> None:
    """Mark APN settings as broken (sets apn_name = BROKEN), then re-derive
    network state — broken APN forces NO_SERVICE."""
    db["device"]["active_apn_settings"]["apn_name"] = APNNames.BROKEN.value
    _simulate_network_search(db)


@register_initialization_function(domain="tau2_telecom")
def break_apn_mms_setting(db: Dict[str, Any]) -> None:
    """Clear the MMSC URL (forces MMS-send to fail)."""
    db["device"]["active_apn_settings"]["mmsc_url"] = None


@register_initialization_function(domain="tau2_telecom")
def set_wifi_calling(db: Dict[str, Any], enabled: bool, mms_over_wifi: Optional[bool] = None) -> None:
    """Set the Wi-Fi Calling toggle and optionally the MMS-over-Wi-Fi flag.

    When both Wi-Fi Calling and MMS-over-Wi-Fi are on, MMS-send is
    blocked (carrier limitation in the upstream simulation).
    """
    db["device"]["wifi_calling_enabled"] = enabled
    if mms_over_wifi is not None:
        db["device"]["wifi_calling_mms_over_wifi"] = mms_over_wifi


@register_initialization_function(domain="tau2_telecom")
def break_vpn(db: Dict[str, Any]) -> None:
    """Connect the VPN (if not already) and set its server_performance to
    POOR. Causes the speed-test simulator to multiply speed by 0.1."""
    _connect_vpn(db)
    db["device"]["vpn_details"]["server_performance"] = PerformanceLevel.POOR.value


@register_initialization_function(domain="tau2_telecom")
def remove_app_permission(db: Dict[str, Any], app_name: str, permission: str) -> None:
    """Clear a specific permission flag on an app.

    ``permission`` is case-insensitive ('sms', 'storage', 'phone', 'network').
    No-op if the app isn't tracked or the permission name isn't valid for
    that app — matches upstream's tolerant behavior (upstream returns a
    ``(success, message)`` tuple; we discard the message at init time).
    """
    app_status = db["device"]["app_statuses"].get(app_name)
    if app_status is None:
        return
    permission = permission.lower()
    if permission not in app_status["permissions"]:
        return
    app_status["permissions"][permission] = False


# =============================================================================
# Agent-side init functions registered with the dispatcher
# =============================================================================


@register_initialization_function(domain="tau2_telecom")
def enable_roaming(db: Dict[str, Any], customer_id: str, line_id: str) -> None:
    """Set ``roaming_enabled = True`` on the named customer's line.

    Raises ``ValueError`` if the customer or line can't be located —
    init-time data errors should fail loud (the dispatcher catches them
    and aborts the scenario).
    """
    line = _get_target_line(db, customer_id, line_id)
    if line is None:
        raise ValueError(f"enable_roaming: line {line_id!r} not found for customer {customer_id!r}")
    line["roaming_enabled"] = True


@register_initialization_function(domain="tau2_telecom")
def disable_roaming(db: Dict[str, Any], customer_id: str, line_id: str) -> None:
    """Set ``roaming_enabled = False`` on the named customer's line."""
    line = _get_target_line(db, customer_id, line_id)
    if line is None:
        raise ValueError(f"disable_roaming: line {line_id!r} not found for customer {customer_id!r}")
    line["roaming_enabled"] = False


@register_initialization_function(domain="tau2_telecom")
def set_data_usage(db: Dict[str, Any], customer_id: str, line_id: str, data_used_gb: float) -> None:
    """Overwrite ``data_used_gb`` on the named line.

    Parameter name matches upstream ``TelecomTools.set_data_usage`` so
    task ``initialization_actions`` payloads (e.g.
    ``{"data_used_gb": 15.1}``) dispatch cleanly without renaming.
    """
    line = _get_target_line(db, customer_id, line_id)
    if line is None:
        raise ValueError(f"set_data_usage: line {line_id!r} not found for customer {customer_id!r}")
    line["data_used_gb"] = float(data_used_gb)


@register_initialization_function(domain="tau2_telecom")
def suspend_line_for_overdue_bill(
    db: Dict[str, Any],
    customer_id: str,
    line_id: str,
    new_bill_id: str,
    contract_ended: bool = False,
) -> None:
    """Suspend a line and create a new OVERDUE bill (mirrors upstream).

    Faithful port of upstream's ``TelecomTools.suspend_line_for_overdue_bill``.
    Used by service_issue scenarios that need to test the agent's
    bill-payment + line-resume flow:

      1. Verify line is ACTIVE (raise if not).
      2. Create a new OVERDUE bill with id ``new_bill_id`` linked to the
         customer, charging one month's plan price.
      3. Set the line's status to SUSPENDED with today's
         ``suspension_start_date``.
      4. If ``contract_ended=True``, also set
         ``line.contract_end_date`` to the last day of the previous
         month — the agent then can NOT lift the suspension by paying,
         per policy.

    Parameter names match upstream's ``set_data_usage`` /
    ``suspend_line_for_overdue_bill`` signatures so task
    ``initialization_actions`` payloads dispatch without renaming.
    """
    from datetime import date, datetime, timedelta

    line = _get_target_line(db, customer_id, line_id)
    if line is None:
        raise ValueError(f"suspend_line_for_overdue_bill: line {line_id!r} not found for customer {customer_id!r}")
    if line["status"] != LineStatus.ACTIVE.value:
        raise ValueError(
            f"suspend_line_for_overdue_bill: line {line_id!r} must be active "
            f"to suspend for unpaid bill (got status={line['status']!r})"
        )

    customer = _get_customer_by_id(db, customer_id)
    if customer is None:
        raise ValueError(f"suspend_line_for_overdue_bill: customer {customer_id!r} not found")

    # No existing OVERDUE bill allowed (one open dispute at a time).
    for bid in customer.get("bill_ids") or []:
        bill = next((b for b in (db.get("bills") or []) if b["bill_id"] == bid), None)
        if bill is not None and bill["status"] == BillStatus.OVERDUE.value:
            raise ValueError(
                f"suspend_line_for_overdue_bill: customer {customer_id!r} already has an overdue bill ({bid})"
            )

    plan = next((p for p in (db.get("plans") or []) if p["plan_id"] == line.get("plan_id")), None)
    if plan is None:
        raise ValueError(f"suspend_line_for_overdue_bill: plan {line.get('plan_id')!r} not found")
    amount = float(plan["price_per_month"])

    # ``current_date`` on the scenario drives "today" — but init
    # functions run against the bot's live state and don't have that
    # context. Fall back to system today; the bot logs the date
    # window so scoring can attribute correctly.
    today = date.today()
    first_day_of_last_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    last_day_of_last_month = today.replace(day=1) - timedelta(days=1)

    overdue_bill = {
        "bill_id": new_bill_id,
        "customer_id": customer_id,
        "period_start": first_day_of_last_month.isoformat(),
        "period_end": last_day_of_last_month.isoformat(),
        "issue_date": first_day_of_last_month.isoformat(),
        "total_due": amount,
        "due_date": (first_day_of_last_month + timedelta(days=14)).isoformat(),
        "status": BillStatus.OVERDUE.value,
        "line_items": [
            {
                "description": f"Charge for line {line_id}",
                "amount": amount,
                "date": today.isoformat(),
                "item_type": "Charge",
            }
        ],
    }
    db.setdefault("bills", []).append(overdue_bill)
    customer.setdefault("bill_ids", []).append(new_bill_id)
    line["status"] = LineStatus.SUSPENDED.value
    line["suspension_start_date"] = today.isoformat()
    if contract_ended:
        line["contract_end_date"] = last_day_of_last_month.isoformat()

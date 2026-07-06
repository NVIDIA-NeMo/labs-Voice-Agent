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
#   src/tau2/domains/telecom/user_tools.py — user-side assert_* methods +
#       helper methods (_get_mobile_data_working, _run_speed_test, _can_send_mms).
#   src/tau2/domains/telecom/tools.py      — agent-side assert_* methods
#       (assert_data_refueling_amount, assert_no_overdue_bill) + their
#       helper lookups (_get_bill_by_id, _get_target_line, get_customer_by_id).
#
# Translation notes vs upstream:
#   - Upstream uses Pydantic model attribute access (``self.device.airplane_mode``);
#     we use plain dict access (``db["device"]["airplane_mode"]``). The
#     Pydantic round-trip in ``scripts/prepare_tau2_data/prepare_telecom.py``
#     materializes all default fields so the dict access is safe without
#     defensive ``.get(default)`` everywhere.
#   - Enum comparisons stay typed — we import the enum classes from
#     ``tau2_telecom_params`` and compare ``db["x"] == SignalStrength.NONE``.
#     Because the enums inherit from ``str``, this matches both the string
#     values in the round-tripped JSON and the enum instances themselves,
#     so behavior matches upstream and we keep the cross-reference clean.
#   - Helper functions (``_get_mobile_data_working``, ``_run_speed_test``,
#     ``_can_send_mms``, the agent-side lookups) keep their leading
#     underscore name as a hint that they're not registered with the
#     predicate dispatcher — only the public ``assert_*`` functions get
#     ``@register_db_state_predicate`` decorators.

"""Telecom DB-state predicate ports for the ``db_state_assertions`` framework.

Six predicates total — four user-side and two agent-side. Each one is
registered under ``domain="tau2_telecom"`` via the flat ``(domain, func_name)``
predicate registry; the runner dispatches by ``func_name`` only (predicates
are side-agnostic, the caller picks the right DB).

User-side predicates (read ``shared_state["db"]`` on the user bot):

  - ``assert_mobile_data_status(db, expected_status)`` — single derived bool
    summarizing whether mobile data is functional. Combines 6 device
    fields via ``_get_mobile_data_working``.
  - ``assert_internet_speed(db, expected_speed, expected_desc=None)`` —
    runs a deterministic simulated speed test against device + surroundings
    state; checks ``simulated_speed >= expected_speed`` (and optionally
    that the speed-category label matches).
  - ``assert_service_status(db, expected_status)`` — direct check on
    ``device.network_connection_status``.
  - ``assert_can_send_mms(db, expected_status)`` — multi-field check
    (mobile data + network tech + MMSC config + messaging-app permissions).

Agent-side predicates (read ``shared_state["db"]`` on the agent bot):

  - ``assert_data_refueling_amount(db, customer_id, line_id, expected_amount)``
    — checks the line's ``data_refueling_gb`` field against an expected
    value (within float tolerance).
  - ``assert_no_overdue_bill(db, overdue_bill_id)`` — true if the named
    bill is either absent from the DB or marked PAID.

Each predicate returns ``bool`` and never raises (the dispatcher catches,
but we don't rely on that — upstream's semantics for these is "false on
unexpected conditions"). The matching ``assert_value`` in the upstream
task JSON is what determines pass/fail; the predicate just produces the
underlying signal.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from nemo_voice_agent.evaluation.db_state_predicates import register_db_state_predicate
from nemo_voice_agent.evaluation.tools.tau2_telecom_params import (
    BillStatus,
    NetworkStatus,
    NetworkTechnology,
    PerformanceLevel,
    SignalStrength,
)


# =============================================================================
# User-side helpers (private; not registered with the predicate dispatcher)
# =============================================================================


def _get_mobile_data_working(db: Dict[str, Any]) -> bool:
    """Return True if mobile data is currently functional, False otherwise.

    Mirrors upstream's ``TelecomUserTools._get_mobile_data_working``. Mobile
    data is NOT working when any of the following are true (early-return
    short-circuit, in this order):

      - Airplane mode is on
      - Network signal strength is NONE
      - Network connection status is NO_SERVICE
      - User is abroad AND (roaming not enabled on device OR roaming not allowed by location)
      - Master data switch is off
      - Surroundings flag says data usage exceeded
    """
    device = db["device"]
    surroundings = db["surroundings"]

    if device["airplane_mode"] or device["network_signal_strength"] == SignalStrength.NONE:
        return False

    if device["network_connection_status"] == NetworkStatus.NO_SERVICE:
        return False

    if surroundings["is_abroad"]:
        if not device["roaming_enabled"] or not surroundings["roaming_allowed"]:
            return False

    if not device["data_enabled"]:
        return False

    if surroundings["mobile_data_usage_exceeded"]:
        return False

    return True


def _run_speed_test(db: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    """Simulate a deterministic speed test against current device state.

    Mirrors upstream's ``TelecomUserTools._run_speed_test``. Returns
    ``(speed_mbps, description)`` where:

      - ``speed_mbps`` is ``None`` when mobile data isn't working
        (description = "No Connection" in that case).
      - Otherwise speed is computed as
        ``(min + max) / 2 * signal_factor * base_speed_factor`` with:
          - ``(min, max)`` from the network-technology speed-range table
          - ``signal_factor`` from the signal-strength multiplier table
          - ``base_speed_factor`` reduced by VPN-with-poor-performance (×0.1)
            and data-saver-mode (×0.2)
      - ``description`` is a verbal label ("Very Poor" / "Poor" / "Fair" /
        "Good" / "Excellent") bucketed from the computed speed.
    """
    if not _get_mobile_data_working(db):
        return None, "No Connection"

    device = db["device"]

    # Base speed factor: 0.1 if VPN-with-poor-performance, else 1.0
    if (
        device["vpn_connected"]
        and device.get("vpn_details") is not None
        and device["vpn_details"]["server_performance"] == PerformanceLevel.POOR
    ):
        base_speed_factor = 0.1
    else:
        base_speed_factor = 1.0

    if device["data_saver_mode"]:
        base_speed_factor *= 0.2

    # Network technology → (min, max) Mbps range
    tech_speed_map = {
        NetworkTechnology.TWO_G.value: (0.1, 0.4),
        NetworkTechnology.THREE_G.value: (1.0, 5.0),
        NetworkTechnology.FOUR_G.value: (10.0, 100.0),
        NetworkTechnology.FIVE_G.value: (50.0, 500.0),
        NetworkTechnology.NONE.value: (0.0, 0.0),
    }
    min_speed, max_speed = tech_speed_map.get(device["network_technology_connected"], (0.0, 0.0))

    # Signal strength → multiplier
    signal_factor_map = {
        SignalStrength.POOR.value: 0.2,
        SignalStrength.FAIR.value: 0.5,
        SignalStrength.GOOD.value: 0.8,
        SignalStrength.EXCELLENT.value: 1.0,
        SignalStrength.NONE.value: 0.0,
    }
    signal_factor = signal_factor_map.get(device["network_signal_strength"], 0.0)

    simulated_speed = (min_speed + max_speed) / 2.0 * signal_factor * base_speed_factor
    simulated_speed = round(simulated_speed, 2)

    # Bucket the speed into a verbal description
    if simulated_speed < 1:
        desc = "Very Poor"
    elif simulated_speed < 5:
        desc = "Poor"
    elif simulated_speed < 25:
        desc = "Fair"
    elif simulated_speed < 100:
        desc = "Good"
    else:
        desc = "Excellent"
    return simulated_speed, desc


def _can_send_mms(db: Dict[str, Any]) -> bool:
    """Mirror of upstream's ``TelecomUserTools._can_send_mms``.

    MMS sending requires *all* of:
      - mobile data working (delegates to ``_get_mobile_data_working``)
      - network technology is 3G or higher (NOT 2G)
      - WiFi-calling-with-MMS-over-WiFi is OFF (carrier limitation)
      - MMSC URL is configured (non-None)
      - The ``messaging`` app exists in app_statuses and has both
        ``storage`` and ``sms`` permissions
    """
    if not _get_mobile_data_working(db):
        return False

    device = db["device"]
    if device["network_technology_connected"] == NetworkTechnology.TWO_G.value:
        return False

    if device["wifi_calling_enabled"] and device["wifi_calling_mms_over_wifi"]:
        return False

    if device["active_apn_settings"]["mmsc_url"] is None:
        return False

    msg_app = device["app_statuses"].get("messaging")
    if msg_app is None:
        return False
    permissions = msg_app["permissions"]
    return permissions["storage"] and permissions["sms"]


# =============================================================================
# Agent-side helpers (private; not registered with the predicate dispatcher)
# =============================================================================


def _get_bill_by_id(db: Dict[str, Any], bill_id: str) -> Optional[Dict[str, Any]]:
    """Iterate ``db["bills"]`` looking for ``bill_id``. Returns ``None`` if not found.

    Upstream raises ``ValueError`` here; we return ``None`` because
    ``assert_no_overdue_bill`` treats "bill not found" as success and the
    None-check is simpler than try/except for that semantic.
    """
    for bill in db.get("bills") or []:
        if bill["bill_id"] == bill_id:
            return bill
    return None


def _get_customer_by_id(db: Dict[str, Any], customer_id: str) -> Optional[Dict[str, Any]]:
    """Iterate ``db["customers"]`` looking for ``customer_id``. Returns
    ``None`` if not found (upstream raises). Same rationale as
    ``_get_bill_by_id`` — caller decides the missing-customer semantic."""
    for customer in db.get("customers") or []:
        if customer["customer_id"] == customer_id:
            return customer
    return None


def _get_line_by_id(db: Dict[str, Any], line_id: str) -> Optional[Dict[str, Any]]:
    """Iterate ``db["lines"]`` looking for ``line_id``."""
    for line in db.get("lines") or []:
        if line["line_id"] == line_id:
            return line
    return None


def _get_target_line(db: Dict[str, Any], customer_id: str, line_id: str) -> Optional[Dict[str, Any]]:
    """Lookup a line scoped to a customer.

    Mirrors upstream's ``_get_target_line``: verifies ``line_id`` is in
    ``customer.line_ids`` before returning the line. Returns ``None`` if
    either the customer or the line isn't found, or if the line isn't
    associated with the customer.
    """
    customer = _get_customer_by_id(db, customer_id)
    if customer is None:
        return None
    if line_id not in (customer.get("line_ids") or []):
        return None
    return _get_line_by_id(db, line_id)


# =============================================================================
# Predicates registered with the db_state_assertions framework
# =============================================================================


@register_db_state_predicate(domain="tau2_telecom")
def assert_mobile_data_status(db: Dict[str, Any], expected_status: bool) -> bool:
    """Assert that mobile data working status matches ``expected_status``."""
    return _get_mobile_data_working(db) == expected_status


@register_db_state_predicate(domain="tau2_telecom")
def assert_internet_speed(
    db: Dict[str, Any],
    expected_speed: float,
    expected_desc: Optional[str] = None,
) -> bool:
    """Assert that the simulated speed meets a minimum threshold.

    Args:
        expected_speed: Minimum acceptable Mbps. The simulated speed must be
            ``>= expected_speed`` for the assertion to pass. When the
            speed test fails entirely (mobile data not working), the speed
            is treated as 0.0 — assertion passes only if ``expected_speed``
            is also <= 0.
        expected_desc: Optional verbal-bucket constraint. When provided,
            the simulated speed's category label (case-insensitive) must
            equal this value AND the numeric threshold must also be met.
            Common values from upstream tasks: ``"excellent"``, ``"good"``.
    """
    speed, desc = _run_speed_test(db)
    speed = speed or 0.0
    if expected_desc is None:
        return speed >= expected_speed
    # Both numeric threshold AND label must match
    return speed >= expected_speed and (desc or "").lower() == expected_desc.lower()


@register_db_state_predicate(domain="tau2_telecom")
def assert_service_status(db: Dict[str, Any], expected_status: str) -> bool:
    """Assert that ``device.network_connection_status`` matches ``expected_status``.

    ``expected_status`` is the string value of a ``NetworkStatus`` enum
    (e.g. ``"connected"``, ``"no_service"``).
    """
    return db["device"]["network_connection_status"] == NetworkStatus(expected_status)


@register_db_state_predicate(domain="tau2_telecom")
def assert_can_send_mms(db: Dict[str, Any], expected_status: bool) -> bool:
    """Assert that the messaging-app MMS-send capability matches ``expected_status``."""
    return _can_send_mms(db) == expected_status


@register_db_state_predicate(domain="tau2_telecom")
def assert_data_refueling_amount(db: Dict[str, Any], customer_id: str, line_id: str, expected_amount: float) -> bool:
    """Assert that the named line's ``data_refueling_gb`` matches ``expected_amount``.

    Float-tolerance compare (``abs(diff) < 1e-6``). Returns ``False`` if the
    line cannot be located (vs upstream which raises) — predicate-as-bool
    semantics are simpler for the dispatcher.
    """
    line = _get_target_line(db, customer_id, line_id)
    if line is None:
        return False
    return abs(line["data_refueling_gb"] - expected_amount) < 1e-6


@register_db_state_predicate(domain="tau2_telecom")
def assert_no_overdue_bill(db: Dict[str, Any], overdue_bill_id: str) -> bool:
    """Assert that ``overdue_bill_id`` is either absent or paid.

    Mirrors upstream: returns ``True`` when (a) the bill isn't in the DB
    at all (missing means "no longer overdue"), or (b) the bill exists
    and its status is ``PAID``. Returns ``False`` otherwise (e.g. still
    overdue, draft, awaiting payment, disputed).
    """
    bill = _get_bill_by_id(db, overdue_bill_id)
    if bill is None:
        return True
    return bill["status"] == BillStatus.PAID

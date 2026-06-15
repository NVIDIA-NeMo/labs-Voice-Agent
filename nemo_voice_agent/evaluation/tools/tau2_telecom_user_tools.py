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
#   src/tau2/domains/telecom/user_tools.py — user-side LLM-callable methods
#       decorated with ``@is_tool(...)``. This first cut ports the minimum
#       4 tools the simplest telecom scenario needs:
#         - toggle_data (write; flips data_enabled, re-derives network state)
#         - toggle_airplane_mode (write; flips airplane_mode + wifi/VPN side
#           effects, re-derives network state)
#         - check_status_bar (read; formatted phone status-bar string)
#         - run_speed_test (read; simulated download speed + advice)
#       More tools will be ported incrementally as additional scenarios
#       require them.
#
# Translation notes vs upstream:
#   - These are LLM-callable tools the user-sim's LLM calls during the
#     conversation; they're distinct from M5c's initialization functions
#     (which run before the conversation starts via the
#     ``apply_initialization`` RTVI path). Some tools share helper
#     logic with init functions (network search, status checks); we reuse
#     those helpers across the two layers via plain Python imports.
#   - Write tools subclass ``WriteScenarioTool`` so successful mutations
#     auto-record into ``shared_state["actions"]`` for downstream scoring.
#     The ``action_type`` strings match upstream's @is_tool method names
#     (``toggle_data``, ``toggle_airplane_mode``) so the bridge-pulled
#     action list aligns with upstream task references.

"""Tau2-telecom user-side LLM tool ports.

The user simulator's LLM has these tools available to control the mock
phone state during a conversation. Same architecture as the retail tool
port: dict-form DB in ``self.state["db"]``, ``_Tau2InvokeMixin`` for
sync/async dual-routing, ``WriteScenarioTool`` subclass auto-records each
mutation.
"""

# pylint: disable=line-too-long
# flake8: noqa: E501

from typing import Any, ClassVar, Dict, List, Optional, Type

from pipecat.services.llm_service import FunctionCallParams
from pydantic import BaseModel, ValidationError

from nemo_voice_agent.evaluation.tools import register_schema_tool_for_eval
from nemo_voice_agent.evaluation.tools._write_tool_base import WriteScenarioTool
from nemo_voice_agent.evaluation.tools.tau2_telecom_init_functions import (
    _DEFAULT_VPN_DETAILS,
    _disconnect_vpn,
    _simulate_network_search,
)
from nemo_voice_agent.evaluation.tools.tau2_telecom_params import (
    APNNames,
    APNSettings,
    CanSendMmsParams,
    CheckApnSettingsParams,
    CheckAppPermissionsParams,
    CheckAppStatusParams,
    CheckDataRestrictionStatusParams,
    CheckInstalledAppsParams,
    CheckNetworkModePreferenceParams,
    CheckNetworkStatusParams,
    CheckPaymentRequestParams,
    CheckSimStatusParams,
    CheckStatusBarParams,
    CheckVpnStatusParams,
    CheckWifiCallingStatusParams,
    CheckWifiStatusParams,
    ConnectVpnParams,
    DisconnectVpnParams,
    GrantAppPermissionParams,
    MakePaymentParams,
    NetworkModePreference,
    NetworkStatus,
    NetworkTechnology,
    PerformanceLevel,
    RebootDeviceParams,
    ResetApnSettingsParams,
    ReseatSimCardParams,
    RunSpeedTestParams,
    SetApnSettingsParams,
    SetNetworkModePreferenceParams,
    SignalStrength,
    SimStatus,
    ToggleAirplaneModeParams,
    ToggleDataParams,
    ToggleDataSaverModeParams,
    ToggleRoamingParams,
    ToggleWifiCallingParams,
    ToggleWifiParams,
    validation_error_response,
)
from nemo_voice_agent.evaluation.tools.tau2_telecom_predicates import _run_speed_test
from nemo_voice_agent.utils.tool_calling import StandardSchemaTool


# =============================================================================
# Action-type vocabulary
# =============================================================================
#
# Matches upstream's @is_tool method names 1:1. The bridge stamps
# side="user" on user-bot mutations at pull time; combined with these
# action_type strings, the bridge-pulled action list aligns with upstream
# tasks' ``evaluation_criteria.actions`` references for cross-checking.


TAU2_TELECOM_USER_ACTION_TYPES: List[str] = [
    # Phone radio / network
    "toggle_airplane_mode",
    "toggle_data",
    "toggle_roaming",
    "toggle_data_saver_mode",
    "toggle_wifi",
    "toggle_wifi_calling",
    "set_network_mode_preference",
    # SIM
    "reseat_sim_card",
    # APN
    "set_apn_settings",
    "reset_apn_settings",
    # VPN
    "connect_vpn",
    "disconnect_vpn",
    # App permissions
    "grant_app_permission",
    # Device-level
    "reboot_device",
    # Billing
    "make_payment",
    # Read-only tools (``check_*``, ``can_send_mms``, ``run_speed_test``,
    # ``check_payment_request``) don't produce action records.
]


# =============================================================================
# Helper: _check_status_bar
# =============================================================================
#
# Build the user-facing status-bar string. Mirrors upstream's
# ``TelecomUserTools._check_status_bar`` — same emoji indicator vocabulary,
# same precedence (airplane mode overrides signal display, etc.).


_SIGNAL_DISPLAY = {
    SignalStrength.NONE.value: "📵 No Signal",
    SignalStrength.POOR.value: "📶¹ Poor",
    SignalStrength.FAIR.value: "📶² Fair",
    SignalStrength.GOOD.value: "📶³ Good",
    SignalStrength.EXCELLENT.value: "📶⁴ Excellent",
}


def _check_status_bar(db: dict) -> str:
    """Build a human-readable status-bar indicator string."""
    device = db["device"]
    indicators: List[str] = []

    if device["airplane_mode"]:
        indicators.append("✈️ Airplane Mode")
    else:
        # Signal strength
        indicators.append(_SIGNAL_DISPLAY.get(device["network_signal_strength"], "📵 No Signal"))
        # Network technology
        tech = device["network_technology_connected"]
        if tech != NetworkTechnology.NONE.value:
            indicators.append(tech)
        # Data enabled indicator
        if device["data_enabled"] and tech != NetworkTechnology.NONE.value:
            indicators.append("📱 Data Enabled")
            if device["data_saver_mode"]:
                indicators.append("🔽 Data Saver")
        else:
            indicators.append("📵 Data Disabled")

    # WiFi
    if device["wifi_enabled"] and device["wifi_connected"]:
        if device.get("wifi_ssid"):
            indicators.append(f"📡 Connected to {device['wifi_ssid']}")
        else:
            indicators.append("📡 Enabled")

    # VPN
    if device["vpn_connected"]:
        indicators.append("🔒 VPN Connected")

    # Battery
    indicators.append(f"🔋 {device['battery_level']}%")
    return " | ".join(indicators)


# =============================================================================
# Helper: _toggle_data / _toggle_airplane_mode
# =============================================================================
#
# Distinct from M5c's ``turn_data_off`` / ``turn_airplane_mode_on`` init
# actions which drive to a specific end state. These toggle helpers flip
# the current value (the LLM-tool semantic) and re-derive network state.


def _toggle_data(db: dict) -> bool:
    """Toggle ``device.data_enabled``; trigger network re-derivation. Returns new state."""
    new_state = not db["device"]["data_enabled"]
    db["device"]["data_enabled"] = new_state
    _simulate_network_search(db)
    return new_state


def _toggle_airplane_mode(db: dict) -> bool:
    """Toggle airplane mode; clear wifi connection, disconnect VPN if turning ON.

    Returns the new airplane_mode state.

    Mirrors upstream's ``_toggle_airplane_mode``: when turning ON, also
    disconnects VPN (mode physically can't keep it alive); when turning
    OFF, drops the connection status to SEARCHING so the network search
    re-derives it. Wifi connection is cleared in both directions.
    """
    device = db["device"]
    was_on = device["airplane_mode"]
    device["airplane_mode"] = not was_on

    if was_on:
        # Turning OFF — start searching
        device["network_connection_status"] = NetworkStatus.SEARCHING.value
        if device["wifi_enabled"]:
            device["wifi_connected"] = False
            device["wifi_ssid"] = None
            device["wifi_signal_strength"] = SignalStrength.NONE.value
    else:
        # Turning ON — drop wifi + VPN
        device["wifi_connected"] = False
        device["wifi_ssid"] = None
        device["wifi_signal_strength"] = SignalStrength.NONE.value
        if device["vpn_connected"]:
            _disconnect_vpn(db)

    _simulate_network_search(db)
    return device["airplane_mode"]


# =============================================================================
# Helper: _db_not_initialized
# =============================================================================


def _db_not_initialized() -> dict:
    return {
        "status": "error",
        "error_type": "db_not_initialized",
        "message": (
            "Scenario user-side database not loaded. This indicates a fixture-loading bug."
        ),
    }


# =============================================================================
# Tool-class bases (same shape as tau2_retail_tools._Tau2InvokeMixin et al)
# =============================================================================


class _Tau2InvokeMixin:
    """Provides sync ``invoke(**kwargs)`` + async ``_execute(params)`` routing."""

    PARAMS_MODEL: ClassVar[Optional[Type[BaseModel]]] = None

    def invoke(self, **kwargs) -> Any:
        try:
            p = self.PARAMS_MODEL.model_validate(kwargs) if self.PARAMS_MODEL else None
        except ValidationError as exc:
            return validation_error_response(exc)
        return self._do_work(p)

    async def _execute(self, params: FunctionCallParams) -> None:
        result = self.invoke(**(params.arguments or {}))
        await params.result_callback(result)

    def _do_work(self, p) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError(f"{type(self).__name__} must implement _do_work(p)")


class _Tau2TelecomReadTool(_Tau2InvokeMixin, StandardSchemaTool):
    """Base for telecom user-side read-only tools."""

    DESCRIPTION: ClassVar[str] = ""

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or self.DESCRIPTION)
        self.state = shared_state if shared_state is not None else {}


class _Tau2TelecomWriteTool(_Tau2InvokeMixin, WriteScenarioTool):
    """Base for telecom user-side write tools."""

    ACTION_TYPES = TAU2_TELECOM_USER_ACTION_TYPES
    DESCRIPTION: ClassVar[str] = ""

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or self.DESCRIPTION)
        self.state = shared_state if shared_state is not None else {}


# =============================================================================
# Write tools — phone state mutations
# =============================================================================


@register_schema_tool_for_eval(domain="tau2_telecom")
class ToggleDataTool(_Tau2TelecomWriteTool):
    name = "toggle_data"
    DESCRIPTION = (
        "Toggles your phone's mobile data connection ON or OFF. Controls whether "
        "your phone can use cellular data for internet access when Wi-Fi is "
        "unavailable. Returns the new data connection status."
    )
    PARAMS_MODEL = ToggleDataParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        new_state = _toggle_data(db)
        self._record_action(
            {
                "action_type": "toggle_data",
                "name": "toggle_data",
                "arguments": {},
                "result": {"data_enabled": new_state},
            }
        )
        status_bar = _check_status_bar(db)
        on_off = "ON" if new_state else "OFF"
        return f"Mobile Data is now {on_off}.\nStatus Bar: {status_bar}"


@register_schema_tool_for_eval(domain="tau2_telecom")
class ToggleAirplaneModeTool(_Tau2TelecomWriteTool):
    name = "toggle_airplane_mode"
    DESCRIPTION = (
        "Toggles Airplane Mode ON or OFF. When ON, it disconnects all wireless "
        "communications including cellular, Wi-Fi, and Bluetooth. Returns the "
        "new state of airplane_mode."
    )
    PARAMS_MODEL = ToggleAirplaneModeParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        new_state = _toggle_airplane_mode(db)
        self._record_action(
            {
                "action_type": "toggle_airplane_mode",
                "name": "toggle_airplane_mode",
                "arguments": {},
                "result": {"airplane_mode": new_state},
            }
        )
        status_bar = _check_status_bar(db)
        on_off = "ON" if new_state else "OFF"
        return f"Airplane Mode is now {on_off}.\nStatus Bar: {status_bar}"


# =============================================================================
# Read tools — phone state inspection
# =============================================================================


@register_schema_tool_for_eval(domain="tau2_telecom")
class CheckStatusBarTool(_Tau2TelecomReadTool):
    name = "check_status_bar"
    DESCRIPTION = (
        "Shows what icons are currently visible in your phone's status bar (the "
        "area at the top of the screen). Displays network signal strength, "
        "mobile data status (enabled, disabled, data saver), Wi-Fi status, VPN "
        "status, and battery level."
    )
    PARAMS_MODEL = CheckStatusBarParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        return f"Status Bar: {_check_status_bar(db)}"


@register_schema_tool_for_eval(domain="tau2_telecom")
class RunSpeedTestTool(_Tau2TelecomReadTool):
    name = "run_speed_test"
    DESCRIPTION = (
        "Measures your current internet connection speed (download speed). "
        "Provides information about connection quality and what activities it "
        "can support."
    )
    PARAMS_MODEL = RunSpeedTestParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        speed_mbps, description = _run_speed_test(db)
        if speed_mbps is None:
            return f"Speed test failed: {description or 'Could not determine speed'}."

        advice_map = {
            "Very Poor": "Connection is very slow. Basic web browsing might be difficult.",
            "Poor": "Connection is slow. Web browsing may be sluggish, streaming difficult.",
            "Fair": "Connection is okay for web browsing and some standard definition streaming.",
            "Good": "Connection is good for most activities, including HD streaming.",
            "Excellent": "Connection is very fast.",
        }
        advice = advice_map.get(description or "", "")
        return f"Speed Test Result: {speed_mbps:.2f} Mbps ({description}). {advice}"


# =============================================================================
# Helpers shared across multiple new tools
# =============================================================================


def _get_mobile_data_working(db: dict) -> bool:
    """Mirror of upstream's ``_get_mobile_data_working``.

    True only when every prerequisite for mobile data is satisfied:
    airplane mode off, signal not NONE, connection status not
    NO_SERVICE, roaming-allowed when abroad, data enabled, usage not
    exceeded.
    """
    device = db["device"]
    surroundings = db["surroundings"]
    if device["airplane_mode"] or device["network_signal_strength"] == SignalStrength.NONE.value:
        return False
    if device["network_connection_status"] == NetworkStatus.NO_SERVICE.value:
        return False
    if surroundings["is_abroad"]:
        if not device["roaming_enabled"] or not surroundings.get("roaming_allowed", True):
            return False
    if not device["data_enabled"]:
        return False
    if surroundings.get("mobile_data_usage_exceeded", False):
        return False
    return True


def _can_send_mms(db: dict) -> bool:
    """Mirror of upstream's ``TelecomUserTools._can_send_mms``."""
    device = db["device"]
    if not _get_mobile_data_working(db):
        return False
    if device["network_technology_connected"] == NetworkTechnology.TWO_G.value:
        return False
    if device.get("wifi_calling_enabled") and device.get("wifi_calling_mms_over_wifi"):
        return False
    if device["active_apn_settings"].get("mmsc_url") is None:
        return False
    msg_app = device["app_statuses"].get("messaging")
    if msg_app is None:
        return False
    perms = msg_app["permissions"]
    return bool(perms.get("storage")) and bool(perms.get("sms"))


def _set_network_mode_preference(db: dict, mode: str) -> Optional[str]:
    """Set preferred network mode; trigger search. Returns canonical value
    or ``None`` on invalid mode (upstream returns None instead of raising)."""
    try:
        normalized = NetworkModePreference(mode).value
    except ValueError:
        return None
    db["device"]["network_mode_preference"] = normalized
    _simulate_network_search(db)
    return normalized


def _toggle_roaming(db: dict) -> bool:
    new_state = not db["device"]["roaming_enabled"]
    db["device"]["roaming_enabled"] = new_state
    _simulate_network_search(db)
    return new_state


def _toggle_data_saver_mode(db: dict) -> bool:
    new_state = not db["device"]["data_saver_mode"]
    db["device"]["data_saver_mode"] = new_state
    return new_state


def _toggle_wifi(db: dict) -> Optional[bool]:
    """Toggle Wi-Fi radio; returns None when airplane mode is on (can't change)."""
    device = db["device"]
    if device["airplane_mode"]:
        return None
    new_state = not device["wifi_enabled"]
    device["wifi_enabled"] = new_state
    if not new_state:
        device["wifi_connected"] = False
        device["wifi_ssid"] = None
        device["wifi_signal_strength"] = SignalStrength.NONE.value
    return new_state


def _toggle_wifi_calling(db: dict) -> bool:
    new_state = not db["device"]["wifi_calling_enabled"]
    db["device"]["wifi_calling_enabled"] = new_state
    return new_state


def _reseat_sim_card(db: dict) -> str:
    """Re-insert the SIM card, then re-derive network state."""
    db["device"]["sim_card_missing"] = False
    _simulate_network_search(db)
    return "SIM card re-seated successfully."


def _set_apn_settings(db: dict, apn_settings: dict) -> str:
    """Set the device's active APN settings to the supplied dict + trigger search."""
    # Validate via Pydantic (raises if invalid) but store as dict in state.
    validated = APNSettings.model_validate(apn_settings).model_dump(mode="json")
    db["device"]["active_apn_settings"] = validated
    _simulate_network_search(db)
    return f"APN settings set to: {validated.get('apn_name')}"


def _reset_apn_settings(db: dict) -> str:
    """Defer APN reset to the next reboot — mirror of upstream's behavior."""
    db["device"]["active_apn_settings"]["reset_at_reboot"] = True
    _simulate_network_search(db)
    return "APN settings will reset at reboot."


def _connect_vpn_tool(db: dict) -> Optional[bool]:
    """Connect VPN with default details. Returns None if already connected,
    True on connect, False if upstream considered it a no-op (matches the
    upstream tristate semantics)."""
    device = db["device"]
    if device["vpn_connected"]:
        return None
    device["vpn_connected"] = True
    device["vpn_details"] = dict(_DEFAULT_VPN_DETAILS)
    return True


def _disconnect_vpn_tool(db: dict) -> bool:
    """Disconnect VPN. Returns True on disconnect, False if no active VPN."""
    device = db["device"]
    if not device["vpn_connected"]:
        return False
    device["vpn_connected"] = False
    device["vpn_details"] = None
    return True


def _reboot_device(db: dict) -> str:
    """Reboot: apply any pending APN reset, then re-derive network state."""
    device = db["device"]
    lines = []
    if device["active_apn_settings"].get("reset_at_reboot"):
        lines.append("Resetting APN settings...")
        # Reset to default APNSettings shape (validated, dumped to dict)
        device["active_apn_settings"] = APNSettings().model_dump(mode="json")
    lines.append("Restarting network services...")
    device["network_connection_status"] = NetworkStatus.SEARCHING.value
    _simulate_network_search(db)
    return "\n".join(lines)


def _grant_app_permission(db: dict, app_name: str, permission: str) -> "tuple[bool, str]":
    """Grant ``permission`` to ``app_name``. Returns (success, message)."""
    apps = db["device"]["app_statuses"]
    app = apps.get(app_name)
    if app is None:
        return False, f"App '{app_name}' not found. Cannot grant permission."
    permission = permission.lower()
    perms = app["permissions"]
    if permission not in perms:
        return (
            False,
            f"Permission '{permission}' not tracked for app '{app_name}', "
            f"available permissions: {list(perms.keys())}",
        )
    perms[permission] = True
    return True, f"Permission '{permission}' granted to app '{app_name}'."


# =============================================================================
# User-side read tools (no DB-mutation; no action records)
# =============================================================================


@register_schema_tool_for_eval(domain="tau2_telecom")
class CheckNetworkStatusTool(_Tau2TelecomReadTool):
    name = "check_network_status"
    DESCRIPTION = (
        "Checks your phone's connection status to cellular networks and Wi-Fi. "
        "Shows airplane mode status, signal strength, network type, whether "
        "mobile data is enabled, and whether data roaming is enabled."
    )
    PARAMS_MODEL = CheckNetworkStatusParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        device = db["device"]
        surroundings = db["surroundings"]
        lines = [
            f"Airplane Mode: {'ON' if device['airplane_mode'] else 'OFF'}",
            # SIM status mirrors upstream: missing flag overrides sim_card_status
            f"SIM Card Status: {(SimStatus.MISSING.value if device['sim_card_missing'] else device['sim_card_status'])}",
            f"Cellular Connection: {device['network_connection_status']}",
            f"Cellular Signal: {device['network_signal_strength']}",
            f"Cellular Network Type: {device['network_technology_connected']}",
            f"Mobile Data Enabled: {'Yes' if device['data_enabled'] else 'No'}",
            f"Data Roaming Enabled: {'Yes' if device['roaming_enabled'] else 'No'}",
            f"Wi-Fi Radio: {'ON' if device['wifi_enabled'] else 'OFF'}",
            f"Wi-Fi Connected: {'Yes' if device['wifi_connected'] else 'No'}",
        ]
        if device["wifi_connected"]:
            lines.append(f"Connected Wi-Fi Network: {device.get('wifi_ssid')}")
        return "\n".join(lines)


@register_schema_tool_for_eval(domain="tau2_telecom")
class CheckNetworkModePreferenceTool(_Tau2TelecomReadTool):
    name = "check_network_mode_preference"
    DESCRIPTION = "Shows the current network mode preference (e.g., 5G, 4G, 3G, 2G)."
    PARAMS_MODEL = CheckNetworkModePreferenceParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        return f"Network Mode Preference: {db['device']['network_mode_preference']}"


@register_schema_tool_for_eval(domain="tau2_telecom")
class CheckSimStatusTool(_Tau2TelecomReadTool):
    name = "check_sim_status"
    DESCRIPTION = (
        "Checks if your SIM card is working correctly and displays its current "
        "status. Shows if the SIM is active, missing, or locked with a PIN or "
        "PUK code."
    )
    PARAMS_MODEL = CheckSimStatusParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        device = db["device"]
        status = SimStatus.MISSING.value if device["sim_card_missing"] else device["sim_card_status"]
        status_map = {
            SimStatus.ACTIVE.value: "Your SIM card is active and working.",
            SimStatus.MISSING.value: "No SIM card detected in the phone.",
            SimStatus.LOCKED_PIN.value: "The SIM card is locked with a PIN code.",
            SimStatus.LOCKED_PUK.value: "The SIM card is locked with a PUK code.",
        }
        return status_map.get(status, f"Unknown SIM status: {status}")


@register_schema_tool_for_eval(domain="tau2_telecom")
class CheckDataRestrictionStatusTool(_Tau2TelecomReadTool):
    name = "check_data_restriction_status"
    DESCRIPTION = (
        "Checks if your phone has any data-limiting features active. Shows if "
        "Data Saver mode is on."
    )
    PARAMS_MODEL = CheckDataRestrictionStatusParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        if db["device"]["data_saver_mode"]:
            return "Data Saver mode is ON (limits data usage)."
        return "Data Saver mode is OFF."


@register_schema_tool_for_eval(domain="tau2_telecom")
class CheckApnSettingsTool(_Tau2TelecomReadTool):
    name = "check_apn_settings"
    DESCRIPTION = (
        "Checks the technical APN settings your phone uses to connect to your "
        "carrier's mobile data network. Shows current APN name and MMSC URL "
        "for picture messaging."
    )
    PARAMS_MODEL = CheckApnSettingsParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        apn = db["device"]["active_apn_settings"]
        apn_name = apn.get("apn_name") or "Not Set"
        mmsc_url = apn.get("mmsc_url") or "Not Set"
        return (
            f"Current APN Name: {apn_name}\n"
            f"MMSC URL (for picture messages): {mmsc_url}\n"
            "(These are technical settings, usually best left unchanged.)"
        )


@register_schema_tool_for_eval(domain="tau2_telecom")
class CheckWifiStatusTool(_Tau2TelecomReadTool):
    name = "check_wifi_status"
    DESCRIPTION = (
        "Checks your Wi-Fi connection status. Shows if Wi-Fi is turned on, "
        "which network you're connected to (if any), and the signal strength."
    )
    PARAMS_MODEL = CheckWifiStatusParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        device = db["device"]
        if not device["wifi_enabled"]:
            return "Wi-Fi is turned OFF."
        if device["wifi_connected"]:
            return (
                f"Wi-Fi is ON and connected to '{device.get('wifi_ssid')}'. "
                f"Signal strength: {device['wifi_signal_strength']}."
            )
        return "Wi-Fi is ON but not connected to any network."


@register_schema_tool_for_eval(domain="tau2_telecom")
class CheckWifiCallingStatusTool(_Tau2TelecomReadTool):
    name = "check_wifi_calling_status"
    DESCRIPTION = (
        "Checks if Wi-Fi Calling is enabled on your device. This feature "
        "allows you to make and receive calls over a Wi-Fi network instead of "
        "using the cellular network."
    )
    PARAMS_MODEL = CheckWifiCallingStatusParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        return (
            f"Wi-Fi Calling is currently turned "
            f"{'ON' if db['device']['wifi_calling_enabled'] else 'OFF'}."
        )


@register_schema_tool_for_eval(domain="tau2_telecom")
class CheckVpnStatusTool(_Tau2TelecomReadTool):
    name = "check_vpn_status"
    DESCRIPTION = (
        "Checks if you're using a VPN (Virtual Private Network) connection. "
        "Shows if a VPN is active, connected, and displays any available "
        "connection details."
    )
    PARAMS_MODEL = CheckVpnStatusParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        device = db["device"]
        if device["vpn_connected"]:
            details = device.get("vpn_details")
            if details:
                return f"VPN is ON and connected. Details: {details}"
            return "VPN is ON and connected (no specific details available)."
        if device.get("vpn_enabled_setting"):
            return "VPN is turned ON in settings, but currently not connected."
        return "VPN is turned OFF."


@register_schema_tool_for_eval(domain="tau2_telecom")
class CheckInstalledAppsTool(_Tau2TelecomReadTool):
    name = "check_installed_apps"
    DESCRIPTION = "Returns the name of all installed apps on the phone."
    PARAMS_MODEL = CheckInstalledAppsParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        app_names = ", ".join(db["device"]["app_statuses"].keys())
        return f"The following apps are installed on the phone: {app_names}"


@register_schema_tool_for_eval(domain="tau2_telecom")
class CheckAppStatusTool(_Tau2TelecomReadTool):
    name = "check_app_status"
    DESCRIPTION = (
        "Checks detailed information about a specific app. Shows its "
        "permissions and background data usage settings."
    )
    PARAMS_MODEL = CheckAppStatusParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "app_name": {
                "type": "string",
                "description": "The name of the app to inspect (e.g., 'messaging').",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["app_name"]

    def _do_work(self, p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        app = db["device"]["app_statuses"].get(p.app_name)
        if app is None:
            return f"App '{p.app_name}' not found on this phone."
        lines = [f"Status for App: {p.app_name}"]
        allowed = [perm.replace("_", " ").lower() for perm, granted in app["permissions"].items() if granted]
        if not allowed:
            lines.append(" - Permissions: None granted.")
        else:
            lines.append(" - Permissions Granted:")
            for perm in allowed:
                lines.append(f"   - {perm}")
        return "\n".join(lines)


@register_schema_tool_for_eval(domain="tau2_telecom")
class CheckAppPermissionsTool(_Tau2TelecomReadTool):
    name = "check_app_permissions"
    DESCRIPTION = (
        "Checks what permissions a specific app currently has. Shows if the "
        "app has access to features like storage, camera, location, etc."
    )
    PARAMS_MODEL = CheckAppPermissionsParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "app_name": {
                "type": "string",
                "description": "The name of the app to inspect.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["app_name"]

    def _do_work(self, p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        app = db["device"]["app_statuses"].get(p.app_name)
        if app is None:
            return f"App '{p.app_name}' not found on this phone."
        allowed = [perm.replace("_", " ").lower() for perm, granted in app["permissions"].items() if granted]
        if not allowed:
            return f"App '{p.app_name}' currently has no permissions granted."
        return f"App '{p.app_name}' has permission for: {', '.join(allowed)}."


@register_schema_tool_for_eval(domain="tau2_telecom")
class CanSendMmsTool(_Tau2TelecomReadTool):
    name = "can_send_mms"
    DESCRIPTION = "Checks if the default messaging app can send MMS messages."
    PARAMS_MODEL = CanSendMmsParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        if _can_send_mms(db):
            return "Your messaging app can send MMS messages."
        return "Your messaging app cannot send MMS messages."


@register_schema_tool_for_eval(domain="tau2_telecom")
class CheckPaymentRequestTool(_Tau2TelecomReadTool):
    name = "check_payment_request"
    DESCRIPTION = "Checks if the agent has sent you a payment request."
    PARAMS_MODEL = CheckPaymentRequestParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        pr = db["surroundings"].get("payment_request")
        if pr is None:
            return "No payment request has been made."
        return (
            f"You have a payment request for bill {pr['bill_id']} "
            f"of {pr['amount_due']} USD."
        )


# =============================================================================
# User-side write tools (mutate DB; record actions)
# =============================================================================


@register_schema_tool_for_eval(domain="tau2_telecom")
class SetNetworkModePreferenceTool(_Tau2TelecomWriteTool):
    name = "set_network_mode_preference"
    DESCRIPTION = (
        "Changes the type of cellular network your phone prefers to connect to "
        "(e.g., 5G, LTE/4G, 3G). Higher-speed networks (LTE/5G) provide faster "
        "data but may use more battery."
    )
    PARAMS_MODEL = SetNetworkModePreferenceParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "mode": {
                "type": "string",
                "description": (
                    "One of '4g_5g_preferred', '4g_only', '3g_only', '2g_only'."
                ),
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["mode"]

    def _do_work(self, p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        valid = _set_network_mode_preference(db, p.mode)
        if valid is None:
            return (
                f"Failed to set network mode: '{p.mode}' is not a valid option. "
                f"Please use one of: {', '.join(m.value for m in NetworkModePreference)}\n"
                f"Status Bar: {_check_status_bar(db)}"
            )
        self._record_action(
            {
                "action_type": "set_network_mode_preference",
                "name": "set_network_mode_preference",
                "arguments": {"mode": valid},
                "result": {"network_mode_preference": valid},
            }
        )
        return f"Preferred Network Mode set to: {valid}\nStatus Bar: {_check_status_bar(db)}"


@register_schema_tool_for_eval(domain="tau2_telecom")
class ReseatSimCardTool(_Tau2TelecomWriteTool):
    name = "reseat_sim_card"
    DESCRIPTION = (
        "Simulates removing and reinserting your SIM card. This can help "
        "resolve recognition issues."
    )
    PARAMS_MODEL = ReseatSimCardParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        msg = _reseat_sim_card(db)
        self._record_action(
            {
                "action_type": "reseat_sim_card",
                "name": "reseat_sim_card",
                "arguments": {},
                "result": {"sim_card_missing": False},
            }
        )
        return f"{msg}\nStatus Bar: {_check_status_bar(db)}"


@register_schema_tool_for_eval(domain="tau2_telecom")
class ToggleRoamingTool(_Tau2TelecomWriteTool):
    name = "toggle_roaming"
    DESCRIPTION = (
        "Toggles Data Roaming ON or OFF. When ON, your phone can use data "
        "networks in areas outside your carrier's coverage. Returns the new "
        "data roaming status."
    )
    PARAMS_MODEL = ToggleRoamingParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        new_state = _toggle_roaming(db)
        self._record_action(
            {
                "action_type": "toggle_roaming",
                "name": "toggle_roaming",
                "arguments": {},
                "result": {"roaming_enabled": new_state},
            }
        )
        return (
            f"Data Roaming is now {'ON' if new_state else 'OFF'}.\n"
            f"Status Bar: {_check_status_bar(db)}"
        )


@register_schema_tool_for_eval(domain="tau2_telecom")
class ToggleDataSaverModeTool(_Tau2TelecomWriteTool):
    name = "toggle_data_saver_mode"
    DESCRIPTION = (
        "Toggles Data Saver mode ON or OFF. When ON, it reduces data usage, "
        "which may affect data speed. Returns the new data saver mode status."
    )
    PARAMS_MODEL = ToggleDataSaverModeParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        new_state = _toggle_data_saver_mode(db)
        self._record_action(
            {
                "action_type": "toggle_data_saver_mode",
                "name": "toggle_data_saver_mode",
                "arguments": {},
                "result": {"data_saver_mode": new_state},
            }
        )
        return (
            f"Data Saver Mode is now {'ON' if new_state else 'OFF'}.\n"
            f"Status Bar: {_check_status_bar(db)}"
        )


@register_schema_tool_for_eval(domain="tau2_telecom")
class SetApnSettingsTool(_Tau2TelecomWriteTool):
    name = "set_apn_settings"
    DESCRIPTION = "Sets the APN settings for the phone."
    PARAMS_MODEL = SetApnSettingsParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "apn_settings": {
                "type": "object",
                "description": (
                    "APN settings dict. Keys may include ``apn_name``, "
                    "``mmsc_url`` (string URL or null), and ``reset_at_reboot`` "
                    "(boolean)."
                ),
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["apn_settings"]

    def _do_work(self, p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        try:
            msg = _set_apn_settings(db, p.apn_settings)
        except ValidationError as exc:
            return validation_error_response(exc)
        self._record_action(
            {
                "action_type": "set_apn_settings",
                "name": "set_apn_settings",
                "arguments": {"apn_settings": p.apn_settings},
                "result": db["device"]["active_apn_settings"],
            }
        )
        return f"{msg}\nStatus Bar: {_check_status_bar(db)}"


@register_schema_tool_for_eval(domain="tau2_telecom")
class ResetApnSettingsTool(_Tau2TelecomWriteTool):
    name = "reset_apn_settings"
    DESCRIPTION = "Resets your APN settings to the default settings."
    PARAMS_MODEL = ResetApnSettingsParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        msg = _reset_apn_settings(db)
        self._record_action(
            {
                "action_type": "reset_apn_settings",
                "name": "reset_apn_settings",
                "arguments": {},
                "result": {"reset_at_reboot": True},
            }
        )
        return f"{msg}\nStatus Bar: {_check_status_bar(db)}"


@register_schema_tool_for_eval(domain="tau2_telecom")
class ToggleWifiTool(_Tau2TelecomWriteTool):
    name = "toggle_wifi"
    DESCRIPTION = (
        "Toggles your phone's Wi-Fi radio ON or OFF. Controls whether your "
        "phone can discover and connect to wireless networks for internet "
        "access. Returns the new Wi-Fi status."
    )
    PARAMS_MODEL = ToggleWifiParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        new_state = _toggle_wifi(db)
        if new_state is None:
            return (
                "Cannot change Wi-Fi settings while Airplane Mode is ON.\n"
                f"Status Bar: {_check_status_bar(db)}"
            )
        self._record_action(
            {
                "action_type": "toggle_wifi",
                "name": "toggle_wifi",
                "arguments": {},
                "result": {"wifi_enabled": new_state},
            }
        )
        return (
            f"Wi-Fi is now {'ON' if new_state else 'OFF'}.\n"
            f"Status Bar: {_check_status_bar(db)}"
        )


@register_schema_tool_for_eval(domain="tau2_telecom")
class ToggleWifiCallingTool(_Tau2TelecomWriteTool):
    name = "toggle_wifi_calling"
    DESCRIPTION = (
        "Toggles Wi-Fi Calling ON or OFF. This feature allows you to make and "
        "receive calls over Wi-Fi instead of the cellular network, which can "
        "help in areas with weak cellular signal. Returns the new Wi-Fi Calling "
        "status."
    )
    PARAMS_MODEL = ToggleWifiCallingParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        new_state = _toggle_wifi_calling(db)
        self._record_action(
            {
                "action_type": "toggle_wifi_calling",
                "name": "toggle_wifi_calling",
                "arguments": {},
                "result": {"wifi_calling_enabled": new_state},
            }
        )
        return (
            f"Wi-Fi Calling is now {'ON' if new_state else 'OFF'}.\n"
            f"Status Bar: {_check_status_bar(db)}"
        )


@register_schema_tool_for_eval(domain="tau2_telecom")
class ConnectVpnTool(_Tau2TelecomWriteTool):
    name = "connect_vpn"
    DESCRIPTION = "Connects to your VPN (Virtual Private Network)."
    PARAMS_MODEL = ConnectVpnParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        connected = _connect_vpn_tool(db)
        if connected is None:
            return "VPN already connected."
        msg = "VPN connected successfully." if connected else "No VPN connection to connect."
        if connected:
            self._record_action(
                {
                    "action_type": "connect_vpn",
                    "name": "connect_vpn",
                    "arguments": {},
                    "result": {"vpn_connected": True},
                }
            )
        return f"{msg}\nStatus Bar: {_check_status_bar(db)}"


@register_schema_tool_for_eval(domain="tau2_telecom")
class DisconnectVpnTool(_Tau2TelecomWriteTool):
    name = "disconnect_vpn"
    DESCRIPTION = (
        "Disconnects any active VPN (Virtual Private Network) connection. "
        "Stops routing your internet traffic through a VPN server, which might "
        "affect connection speed or access to content."
    )
    PARAMS_MODEL = DisconnectVpnParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        disconnected = _disconnect_vpn_tool(db)
        msg = (
            "VPN disconnected successfully."
            if disconnected
            else "No active VPN connection to disconnect."
        )
        if disconnected:
            self._record_action(
                {
                    "action_type": "disconnect_vpn",
                    "name": "disconnect_vpn",
                    "arguments": {},
                    "result": {"vpn_connected": False},
                }
            )
        return f"{msg}\nStatus Bar: {_check_status_bar(db)}"


@register_schema_tool_for_eval(domain="tau2_telecom")
class GrantAppPermissionTool(_Tau2TelecomWriteTool):
    name = "grant_app_permission"
    DESCRIPTION = (
        "Gives a specific permission to an app (like access to storage, camera, "
        "or location). Required for some app functions to work properly."
    )
    PARAMS_MODEL = GrantAppPermissionParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "app_name": {
                "type": "string",
                "description": "Name of the app (e.g., 'messaging').",
            },
            "permission": {
                "type": "string",
                "description": (
                    "Lowercase permission key: 'storage', 'camera', 'microphone', "
                    "'location', 'contacts', 'sms', 'phone'."
                ),
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["app_name", "permission"]

    def _do_work(self, p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        success, message = _grant_app_permission(db, p.app_name, p.permission)
        if success:
            self._record_action(
                {
                    "action_type": "grant_app_permission",
                    "name": "grant_app_permission",
                    "arguments": {"app_name": p.app_name, "permission": p.permission},
                    "result": {"granted": True},
                }
            )
        prefix = "Success. " if success else "Error. "
        return f"{prefix}{message}\nStatus Bar: {_check_status_bar(db)}"


@register_schema_tool_for_eval(domain="tau2_telecom")
class RebootDeviceTool(_Tau2TelecomWriteTool):
    name = "reboot_device"
    DESCRIPTION = (
        "Restarts your phone completely. This can help resolve many temporary "
        "software glitches by refreshing all running services and connections."
    )
    PARAMS_MODEL = RebootDeviceParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        msg = _reboot_device(db)
        self._record_action(
            {
                "action_type": "reboot_device",
                "name": "reboot_device",
                "arguments": {},
                "result": {"rebooted": True},
            }
        )
        return f"{msg}\nStatus Bar: {_check_status_bar(db)}"


@register_schema_tool_for_eval(domain="tau2_telecom")
class MakePaymentTool(_Tau2TelecomWriteTool):
    name = "make_payment"
    DESCRIPTION = "Makes a payment for the bill that the agent has sent you."
    PARAMS_MODEL = MakePaymentParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, _p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        pr = db["surroundings"].get("payment_request")
        if pr is None:
            return "You do not have a payment request."
        pr["paid"] = True
        self._record_action(
            {
                "action_type": "make_payment",
                "name": "make_payment",
                "arguments": {},
                "result": {"bill_id": pr["bill_id"], "amount": pr["amount_due"]},
            }
        )
        return f"Payment of {pr['amount_due']} USD has been made for bill {pr['bill_id']}."


# =============================================================================
# Name → class map (consumed by Tau2TelecomBaseScenario._build_tool_map)
# =============================================================================


TAU2_TELECOM_USER_TOOL_NAME_TO_CLASS: Dict[str, Type[StandardSchemaTool]] = {
    cls.name: cls
    for cls in (
        # Reads (14)
        CheckStatusBarTool,
        CheckNetworkStatusTool,
        CheckNetworkModePreferenceTool,
        CheckSimStatusTool,
        CheckDataRestrictionStatusTool,
        CheckApnSettingsTool,
        CheckWifiStatusTool,
        CheckWifiCallingStatusTool,
        CheckVpnStatusTool,
        CheckInstalledAppsTool,
        CheckAppStatusTool,
        CheckAppPermissionsTool,
        CanSendMmsTool,
        CheckPaymentRequestTool,
        RunSpeedTestTool,
        # Writes (16)
        ToggleAirplaneModeTool,
        ToggleDataTool,
        ToggleRoamingTool,
        ToggleDataSaverModeTool,
        ToggleWifiTool,
        ToggleWifiCallingTool,
        SetNetworkModePreferenceTool,
        ReseatSimCardTool,
        SetApnSettingsTool,
        ResetApnSettingsTool,
        ConnectVpnTool,
        DisconnectVpnTool,
        GrantAppPermissionTool,
        RebootDeviceTool,
        MakePaymentTool,
    )
}
# Single source of truth: ``cls.name`` is the snake_case identifier
# used as registry key (read by ``_do_register_tool``), LLM-visible
# function name (used by ``StandardSchemaTool.schema``), and gold-replay
# dispatch key (used by ``_build_tool_map``). The dict keys above
# drive both ``user_resources.tools`` (registry lookup) and
# ``_build_tool_map`` (gold-replay dispatch) — they're identical now,
# no more parallel naming maps.

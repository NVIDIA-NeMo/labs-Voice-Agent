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
#   src/tau2/domains/telecom/data_model.py       — agent-facing DB models
#       (TelecomDB / Customer / Bill / Line / Device / Plan / etc.)
#   src/tau2/domains/telecom/user_data_model.py  — user-facing DB models
#       (TelecomUserDB / MockPhoneAttributes / UserSurroundings / status enums)
#
# Differences vs upstream:
#   - No ``DB`` base class — upstream's ``DB`` adds ``load``/``dump``/``get_hash``
#     convenience methods that we don't need here; our framework hashes via
#     ``nemo_voice_agent.evaluation.db_hash.get_dict_hash`` (which works on
#     plain dicts, sidestepping Pydantic). The two DB-level models below
#     inherit from a local ``BaseModelNoExtra`` only.
#   - No ``update_device`` / ``update_pydantic_model_with_dict`` helpers —
#     our init functions operate on plain dicts via the M4
#     ``apply_initialization_actions`` path, not through Pydantic mutation.
#   - The ``StatusBar`` model is dropped from the live runtime — upstream
#     only uses it as a derived view in ``check_status_bar`` (a user-side
#     LLM tool), and our predicate/init paths read raw device fields
#     directly. The model is still defined here for completeness in case a
#     future ported tool needs it.

"""Pydantic parameter models for tau2_telecom DB shapes.

Used primarily by ``scripts/prepare_tau2_data/prepare_telecom.py`` to
round-trip the upstream TOML files through Pydantic — this materializes
default field values (e.g. ``UserSurroundings.signal_strength`` per-network
defaults, ``MockPhoneAttributes`` field defaults) that aren't present in
the raw TOML. After round-trip the converted JSON is shape-complete and
downstream predicate / init function code can read fields via dict access
without defensive ``.get(..., default)`` everywhere.

Per-tool ``*Params`` argument-schema classes will be added incrementally
in M5e+ as each tool is ported (same pattern as ``tau2_airline_params.py`` /
``tau2_retail_params.py``).
"""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class BaseModelNoExtra(BaseModel):
    """Mirror of upstream ``tau2.utils.pydantic_utils.BaseModelNoExtra``.

    Forbids extra fields on input — catches typos in upstream data and in
    init/predicate function arguments at validation time.
    """

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Agent-side data models (from data_model.py)
# =============================================================================


# Upstream's ``DEFAULT_START_DATE = datetime.date(2025, 1, 1)`` is used as the
# default for ``Customer.created_at`` (typed as datetime.datetime upstream).
# We keep the same value; Pydantic will coerce date → datetime at validation
# time.
DEFAULT_START_DATE = datetime.date(2025, 1, 1)


class Address(BaseModelNoExtra):
    street: str = Field(description="Street address including house/apartment number")
    city: str = Field(description="City name")
    state: str = Field(description="State or province code (e.g., CA, NY)")
    zip_code: str = Field(description="Postal/ZIP code")


class Plan(BaseModelNoExtra):
    plan_id: str = Field(description="Unique identifier for the plan")
    name: str = Field(description="Display name of the plan")
    data_limit_gb: float = Field(description="Monthly data allowance in gigabytes (GB)")
    price_per_month: float = Field(description="Monthly price of the plan in USD")
    data_refueling_price_per_gb: float = Field(description="Price per gigabyte for data refueling")


class DeviceType(str, Enum):
    PHONE = "phone"
    ROUTER = "router"
    TABLET = "tablet"
    WATCH = "watch"
    OTHER = "other"


class Device(BaseModelNoExtra):
    device_id: str = Field(description="Unique identifier for the device")
    device_type: DeviceType = Field(description="Type/category of the device")
    model: str = Field(description="Model name/number of the device")
    imei: Optional[str] = Field(None, description="International Mobile Equipment Identity number")
    is_esim_capable: bool = Field(description="Whether the device supports eSIM technology")
    activated: bool = Field(False, description="Whether the device has been activated on the network")
    activation_date: Optional[datetime.datetime] = Field(
        None,
        description="Date and time when the device was activated (YYYY-MM-DDTHH:MM:SS, EST)",
    )
    last_esim_transfer_date: Optional[datetime.datetime] = Field(
        None,
        description="Last date an eSIM profile was transferred to this device (YYYY-MM-DDTHH:MM:SS, EST)",
    )


class LineStatus(str, Enum):
    ACTIVE = "Active"
    SUSPENDED = "Suspended"
    PENDING_ACTIVATION = "Pending Activation"
    CLOSED = "Closed"


class Line(BaseModelNoExtra):
    line_id: str = Field(description="Unique identifier for the line")
    phone_number: str = Field(description="Phone number associated with the line")
    status: LineStatus = Field(LineStatus.PENDING_ACTIVATION, description="Current status of the line")
    plan_id: str = Field(description="Plan associated with this line")
    device_id: Optional[str] = Field(None, description="Device associated with this line")
    data_used_gb: float = Field(0.0, description="Data used in the current billing cycle (GB)")
    data_refueling_gb: float = Field(0.0, description="Data refueled in the current billing cycle (GB)")
    roaming_enabled: bool = Field(False, description="International roaming enabled for this line")
    contract_end_date: Optional[datetime.date] = Field(None, description="End date of the current contract (YYYY-MM-DD, EST)")
    last_plan_change_date: Optional[datetime.date] = Field(None, description="Date of the most recent plan change (YYYY-MM-DD, EST)")
    last_sim_replacement_date: Optional[datetime.date] = Field(None, description="Date of the most recent SIM card replacement (YYYY-MM-DD, EST)")
    suspension_start_date: Optional[datetime.date] = Field(None, description="Start date of the current suspension period (YYYY-MM-DD, EST)")


class LineItem(BaseModelNoExtra):
    description: str = Field(description="Descriptive text for the line item")
    amount: float = Field(description="USD amount (positive=charge, negative=credit)")
    date: datetime.date = Field(description="Date the line item was applied (YYYY-MM-DD, EST)")
    item_type: str = Field(description="Category (Plan Charge, Overage, Fee, Credit, Payment, ...)")


class BillStatus(str, Enum):
    DRAFT = "Draft"
    ISSUED = "Issued"
    AWAITING_PAYMENT = "Awaiting Payment"
    PAID = "Paid"
    OVERDUE = "Overdue"
    DISPUTED = "Disputed"


class Bill(BaseModelNoExtra):
    bill_id: str = Field(description="Unique identifier for the bill")
    customer_id: str = Field(description="ID of the customer this bill belongs to")
    period_start: datetime.date = Field(description="Start of the billing period (YYYY-MM-DD, EST)")
    period_end: datetime.date = Field(description="End of the billing period (YYYY-MM-DD, EST)")
    issue_date: datetime.date = Field(description="Date the bill was issued (YYYY-MM-DD, EST)")
    total_due: float = Field(description="Total amount due in USD")
    due_date: datetime.date = Field(description="Date by which payment is due (YYYY-MM-DD, EST)")
    line_items: List[LineItem] = Field(default_factory=list, description="Individual charges, credits, and payments")
    status: BillStatus = Field(BillStatus.DRAFT, description="Current status of the bill")


class AccountStatus(str, Enum):
    ACTIVE = "Active"
    SUSPENDED = "Suspended"
    PENDING_VERIFICATION = "Pending Verification"
    CLOSED = "Closed"


class PaymentMethodType(str, Enum):
    CREDIT_CARD = "Credit Card"
    DEBIT_CARD = "Debit Card"
    PAYPAL = "PayPal"


class PaymentMethod(BaseModelNoExtra):
    method_type: PaymentMethodType = Field(description="Type of payment method")
    account_number_last_4: str = Field(description="Last 4 digits of the account number")
    expiration_date: str = Field(description="Expiration date in MM/YYYY format")


class Customer(BaseModelNoExtra):
    customer_id: str = Field(description="Unique identifier for the customer")
    full_name: str = Field(description="Customer's full name")
    date_of_birth: str = Field(description="Date of birth for identity verification (YYYY-MM-DD)")
    email: str = Field(description="Customer's email address")
    phone_number: str = Field(description="Customer's primary contact phone number")
    address: Address = Field(description="Customer's billing address")
    account_status: AccountStatus = Field(
        AccountStatus.PENDING_VERIFICATION, description="Current status of the customer account"
    )
    payment_methods: List[PaymentMethod] = Field(default_factory=list, description="Stored payment methods")
    line_ids: List[str] = Field(default_factory=list, description="Phone/data lines owned by this customer")
    bill_ids: List[str] = Field(default_factory=list, description="Bills associated with this customer")
    created_at: datetime.datetime = Field(
        DEFAULT_START_DATE,
        description="Account-creation timestamp (YYYY-MM-DDTHH:MM:SS, EST)",
    )
    last_extension_date: Optional[datetime.date] = Field(
        None,
        description="Date of the most recent payment extension (for quarterly limit check) (YYYY-MM-DD, EST)",
    )
    goodwill_credit_used_this_year: float = Field(
        0.0, description="Goodwill credit used in the current calendar year"
    )


class TelecomDB(BaseModelNoExtra):
    """Database interface for the telecom domain (agent-facing).

    Round-trip target for ``scripts/prepare_tau2_data/prepare_telecom.py``'s
    ``db.toml`` → ``db.json`` conversion. Materializes default fields on
    nested types (e.g. ``Bill.line_items=[]``).
    """

    plans: List[Plan] = Field(default_factory=list, description="Available service plans")
    customers: List[Customer] = Field(default_factory=list, description="All customers in the system")
    lines: List[Line] = Field(default_factory=list, description="All lines in the system")
    bills: List[Bill] = Field(default_factory=list, description="All bills in the system")
    devices: List[Device] = Field(default_factory=list, description="All devices in the system")


# =============================================================================
# User-side data models (from user_data_model.py)
# =============================================================================


class SimStatus(str, Enum):
    ACTIVE = "active"
    MISSING = "missing"
    LOCKED_PIN = "locked_pin"
    LOCKED_PUK = "locked_puk"


class NetworkTechnology(str, Enum):
    NONE = "none"
    TWO_G = "2G"
    THREE_G = "3G"
    FOUR_G = "4G"
    FIVE_G = "5G"


class NetworkModePreference(str, Enum):
    FOUR_G_5G_PREFERRED = "4g_5g_preferred"
    FOUR_G_ONLY = "4g_only"
    THREE_G_ONLY = "3g_only"
    TWO_G_ONLY = "2g_only"


class SignalStrength(str, Enum):
    NONE = "none"  # No signal bars
    POOR = "poor"  # 1 bar
    FAIR = "fair"  # 2 bars
    GOOD = "good"  # 3 bars
    EXCELLENT = "excellent"  # 4+ bars


class PerformanceLevel(str, Enum):
    UNKNOWN = "unknown"
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


class NetworkStatus(str, Enum):
    CONNECTED = "connected"
    SEARCHING = "searching"
    NO_SERVICE = "no_service"
    EMERGENCY_ONLY = "emergency_only"


class APNNames(str, Enum):
    INTERNET = "internet"
    BROKEN = "broken"


class APNSettings(BaseModelNoExtra):
    """Access Point Name configuration."""

    apn_name: APNNames = Field(APNNames.INTERNET, description="APN identifier")
    reset_at_reboot: bool = Field(False, description="Whether APN settings reset at next reboot")
    mms_apn: Optional[str] = Field("mms", description="APN name used for MMS traffic, if different")
    mmsc_url: Optional[str] = Field(
        "http://mms.carrier.com/mms/wapenc",
        description="MMSC URL (crucial for MMS)",
    )
    mms_proxy: Optional[str] = Field(None, description="MMS proxy server address")
    mms_port: Optional[int] = Field(None, description="MMS proxy server port")


class VpnDetails(BaseModelNoExtra):
    """Details about the VPN connection if active."""

    server_address: Optional[str] = Field(None, description="VPN server address")
    protocol: Optional[str] = Field(None, description="VPN protocol (e.g., WireGuard, OpenVPN)")
    server_performance: PerformanceLevel = Field(
        default=PerformanceLevel.UNKNOWN,
        validate_default=True,
        description="Estimated performance/latency of the VPN connection",
    )


class AppPermissions(BaseModelNoExtra):
    """Permissions relevant to an application."""

    sms: bool = Field(False, description="Permission to send/read SMS/MMS")
    storage: bool = Field(False, description="Permission to access device storage")
    phone: bool = Field(False, description="Permission to make/manage phone calls")
    network: bool = Field(False, description="Permission to access network/internet")


class AppStatus(BaseModelNoExtra):
    """Status of a specific application relevant to issues."""

    app_name: str
    permissions: AppPermissions = Field(default_factory=AppPermissions, description="Application permissions")


class StatusBar(BaseModelNoExtra):
    """Information displayed in the phone's status bar.

    Upstream uses this as a derived view in ``check_status_bar``; we don't
    consume it from the runtime DB (predicate code reads device fields
    directly). Kept for parity in case a future ported tool needs it.
    """

    signal_strength: SignalStrength = Field(
        default=SignalStrength.NONE,
        validate_default=True,
        description="Cellular signal strength shown in the status bar",
    )
    network_type: NetworkTechnology = Field(
        default=NetworkTechnology.NONE,
        validate_default=True,
        description="Network technology (2G/3G/4G/...) shown in the status bar",
    )
    wifi_connected: bool = Field(False, description="WiFi connected and shown in status bar")
    airplane_mode: bool = Field(False, description="Airplane mode on and shown in status bar")
    vpn_active: bool = Field(False, description="VPN active and shown in status bar")
    data_saver_active: bool = Field(False, description="Data saver mode active and shown in status bar")
    battery_level: int = Field(100, description="Battery level (0-100) shown in the status bar")


class MockPhoneAttributes(BaseModelNoExtra):
    """State attributes of the mock phone device on the user side."""

    # --- SIM and Basic Network ---
    sim_card_status: SimStatus = Field(
        default=SimStatus.ACTIVE,
        validate_default=True,
        description="Current status of the physical or eSIM card",
    )
    sim_card_missing: bool = Field(False, description="Whether the SIM card is missing")
    airplane_mode: bool = Field(False, description="Airplane Mode enabled (disables all radios)")
    network_signal_strength: SignalStrength = Field(
        default=SignalStrength.GOOD,
        validate_default=True,
        description="Current cellular signal strength",
    )
    network_technology_connected: NetworkTechnology = Field(
        default=NetworkTechnology.FIVE_G,
        validate_default=True,
        description="Currently connected cellular network technology (5G/4G/...)",
    )
    network_connection_status: NetworkStatus = Field(
        default=NetworkStatus.CONNECTED,
        validate_default=True,
        description="High-level network status",
    )

    # --- Battery ---
    battery_level: int = Field(80, description="Battery level (0-100)")

    # --- Mobile Data ---
    data_enabled: bool = Field(True, description="Master switch for Mobile/Cellular Data")
    roaming_enabled: bool = Field(False, description="User setting allowing data usage while roaming")
    network_mode_preference: NetworkModePreference = Field(
        default=NetworkModePreference.FOUR_G_5G_PREFERRED,
        validate_default=True,
        description="User's preferred network type",
    )
    active_apn_settings: APNSettings = Field(
        default_factory=APNSettings,
        description="Currently active Access Point Name configuration",
    )

    # --- Wi-Fi ---
    wifi_enabled: bool = Field(False, description="Whether the Wi-Fi radio is enabled")
    wifi_connected: bool = Field(False, description="Currently connected to a Wi-Fi network")
    wifi_ssid: Optional[str] = Field(None, description="SSID of the connected Wi-Fi network")
    wifi_signal_strength: SignalStrength = Field(
        default=SignalStrength.NONE,
        validate_default=True,
        description="Wi-Fi signal strength",
    )

    # --- Calling Features ---
    wifi_calling_enabled: bool = Field(False, description="Whether Wi-Fi Calling is enabled")
    wifi_calling_mms_over_wifi: bool = Field(
        False,
        description="Capability/preference to send/receive MMS over Wi-Fi",
    )

    # --- System-Wide Settings ---
    data_saver_mode: bool = Field(False, description="Data Saver mode enabled")

    # --- VPN ---
    vpn_enabled_setting: bool = Field(False, description="VPN profile configured/enabled in settings")
    vpn_connected: bool = Field(False, description="Active VPN connection tunnel")
    vpn_details: Optional[VpnDetails] = Field(None, description="Active VPN connection details")

    # --- Application State ---
    app_statuses: Dict[str, AppStatus] = Field(
        default_factory=lambda: {
            "messaging": AppStatus(
                app_name="messaging",
                permissions=AppPermissions(sms=True, storage=True, phone=True),
            ),
            "browser": AppStatus(
                app_name="browser",
                permissions=AppPermissions(network=True, storage=True),
            ),
        },
        description="Per-app status (e.g., messaging, browser)",
    )


class PaymentRequest(BaseModelNoExtra):
    """Payment request initiated by the agent that the user can pay."""

    bill_id: str = Field(description="The ID of the bill")
    amount_due: float = Field(description="Amount of the payment in USD")
    paid: bool = Field(default=False, description="Whether the payment has been made")


class UserSurroundings(BaseModelNoExtra):
    """Physical surroundings of the user (per-task; populated by init actions).

    Per-task user state — name, phone number, location, signal-strength
    table at the user's current location, etc. Upstream's user_db.toml
    only carries the ``device`` defaults; ``surroundings`` is populated
    per-task by ``initialization_actions`` like ``set_user_info`` and
    ``set_user_location``. The Pydantic round-trip in
    ``prepare_telecom.py`` materializes the default surroundings block
    (with name=None, phone_number=None, is_abroad=False, etc.) so init
    functions can mutate fields without needing defensive .setdefault().
    """

    name: Optional[str] = Field(None, description="The name of the user")
    phone_number: Optional[str] = Field(None, description="The phone number of the user")
    is_abroad: bool = Field(False, description="Whether the user is currently abroad")
    roaming_allowed: bool = Field(False, description="Whether the user is allowed to roam")
    signal_strength: Dict[NetworkTechnology, SignalStrength] = Field(
        default_factory=lambda: {
            NetworkTechnology.TWO_G: SignalStrength.POOR,
            NetworkTechnology.THREE_G: SignalStrength.FAIR,
            NetworkTechnology.FOUR_G: SignalStrength.GOOD,
            NetworkTechnology.FIVE_G: SignalStrength.EXCELLENT,
        },
        description="Signal strength for each network technology at the user's location",
    )
    mobile_data_usage_exceeded: bool = Field(False, description="User has exceeded data usage limit")
    line_active: bool = Field(True, description="User has an active line")
    payment_request: Optional[PaymentRequest] = Field(
        None, description="Payment request the agent has initiated"
    )


class TelecomUserDB(BaseModelNoExtra):
    """Database interface for the telecom domain (user-facing).

    Round-trip target for ``scripts/prepare_tau2_data/prepare_telecom.py``'s
    ``user_db.toml`` → ``user_db.json`` conversion. Critical to materialize
    the default ``surroundings`` block so predicate / init functions can
    read its fields without dict-key-missing errors.
    """

    device: MockPhoneAttributes = Field(default_factory=MockPhoneAttributes, description="Mock phone device")
    surroundings: UserSurroundings = Field(
        default_factory=UserSurroundings, description="User's physical surroundings"
    )


# =============================================================================
# Convenience exports
# =============================================================================

__all__ = [
    # base helper
    "BaseModelNoExtra",
    # agent-side enums
    "DeviceType", "LineStatus", "BillStatus", "AccountStatus", "PaymentMethodType",
    # agent-side models
    "Address", "Plan", "Device", "Line", "LineItem", "Bill",
    "PaymentMethod", "Customer", "TelecomDB",
    # user-side enums
    "SimStatus", "NetworkTechnology", "NetworkModePreference", "SignalStrength",
    "PerformanceLevel", "NetworkStatus", "APNNames",
    # user-side models
    "APNSettings", "VpnDetails", "AppPermissions", "AppStatus", "StatusBar",
    "MockPhoneAttributes", "PaymentRequest", "UserSurroundings", "TelecomUserDB",
    # constants
    "DEFAULT_START_DATE",
]

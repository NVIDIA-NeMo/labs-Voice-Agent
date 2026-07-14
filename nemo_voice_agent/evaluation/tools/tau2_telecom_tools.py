# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from https://github.com/sierra-research/tau2-bench/tree/voice-user-sim-v1.0
#   src/tau2/domains/telecom/tools.py — agent-side LLM-callable tools
#   (customer lookup, line/bill management, billing, plan changes, etc.).

"""Tau2-telecom agent-side LLM tool ports.

Mirrors upstream ``TelecomTools``. Tools operate on the agent-side
``TelecomDB`` dict stored in ``shared_state["db"]``. Reads return data;
writes mutate ``state["db"]`` and append an entry to
``state["actions"]`` via ``WriteScenarioTool._record_action``.

Conventions match ``tau2_airline_tools`` / ``tau2_retail_tools``:
``_Tau2InvokeMixin`` handles sync ``invoke`` + async ``_execute``,
``_Tau2TelecomAgentReadTool`` / ``_Tau2TelecomAgentWriteTool`` are
the base classes, and every tool declares a snake_case ``name``
class attribute that drives the registry key + LLM-visible function
name + gold-replay dispatch key (matching how upstream policy.md
references tools).
"""

# pylint: disable=line-too-long
# flake8: noqa: E501

from datetime import date, datetime, timedelta
from typing import Any, ClassVar, Dict, List, Optional, Type

from pipecat.services.llm_service import FunctionCallParams
from pydantic import BaseModel, ValidationError

from nemo_voice_agent.evaluation.tools import normalize_tool_result, register_schema_tool_for_eval
from nemo_voice_agent.evaluation.tools._write_tool_base import WriteScenarioTool
from nemo_voice_agent.evaluation.tools.tau2_telecom_params import (
    BillStatus,
    DisableRoamingParams,
    EnableRoamingParams,
    GetBillsForCustomerParams,
    GetCustomerByIdParams,
    GetCustomerByNameParams,
    GetCustomerByPhoneParams,
    GetDataUsageParams,
    GetDetailsByIdParams,
    LineStatus,
    RefuelDataParams,
    ResumeLineParams,
    SendPaymentRequestParams,
    SuspendLineParams,
    TransferToHumanAgentsParams,
    validation_error_response,
)
from nemo_voice_agent.utils.tool_calling import StandardSchemaTool


# =============================================================================
# Action-type vocabulary (agent-side)
# =============================================================================
#
# Matches upstream's @is_tool method names 1:1. The bridge stamps
# ``side="agent"`` on agent-bot mutations at pull time; combined with these
# names, the pulled action list aligns with upstream tasks'
# ``evaluation_criteria.actions`` references.


TAU2_TELECOM_AGENT_ACTION_TYPES: List[str] = [
    # Line management
    "suspend_line",
    "resume_line",
    "enable_roaming",
    "disable_roaming",
    # Billing
    "send_payment_request",
    "refuel_data",
    # Escalation
    "transfer_to_human_agents",
    # Read-only tools (``get_*``) don't produce action records.
]


# =============================================================================
# Helpers: today's date + DB lookups
# =============================================================================


def _get_today(scenario_today: Optional[str] = None) -> date:
    """Return today as a ``date`` object.

    ``scenario_today`` (when supplied) is the scenario's authored anchor
    date — used for tasks whose timing assumptions depend on a fixed
    "now" (e.g., bill due dates relative to a current month). Falls
    back to today when not supplied.
    """
    if scenario_today:
        return datetime.fromisoformat(scenario_today).date()
    return date.today()


def _find_customer(db: dict, customer_id: str) -> Optional[dict]:
    for c in db["customers"]:
        if c["customer_id"] == customer_id:
            return c
    return None


def _find_line(db: dict, line_id: str) -> Optional[dict]:
    for line in db["lines"]:
        if line["line_id"] == line_id:
            return line
    return None


def _find_line_by_phone(db: dict, phone_number: str) -> Optional[dict]:
    for line in db["lines"]:
        if line["phone_number"] == phone_number:
            return line
    return None


def _find_plan(db: dict, plan_id: str) -> Optional[dict]:
    for p in db["plans"]:
        if p["plan_id"] == plan_id:
            return p
    return None


def _find_device(db: dict, device_id: str) -> Optional[dict]:
    for d in db["devices"]:
        if d["device_id"] == device_id:
            return d
    return None


def _find_bill(db: dict, bill_id: str) -> Optional[dict]:
    for b in db["bills"]:
        if b["bill_id"] == bill_id:
            return b
    return None


def _get_target_line(db: dict, customer_id: str, line_id: str) -> dict:
    """Resolve a line + verify it belongs to the customer. Raises ValueError on miss."""
    customer = _find_customer(db, customer_id)
    if customer is None:
        raise ValueError(f"Customer with ID {customer_id} not found")
    if line_id not in customer["line_ids"]:
        raise ValueError(f"Line {line_id} not found for customer {customer_id}")
    line = _find_line(db, line_id)
    if line is None:
        raise ValueError(f"Line with ID {line_id} not found")
    return line


def _customer_to_user_view(customer: dict) -> dict:
    """Return a JSON-safe shallow copy of a customer dict for tool output.

    Same shape as upstream's Customer schema (already dict, just defensive
    copy so the caller can't accidentally mutate ``state["db"]`` through
    the result).
    """
    return dict(customer)


# =============================================================================
# Tool-class bases
# =============================================================================


class _Tau2InvokeMixin:
    """Provides sync ``invoke(**kwargs)`` + async ``_execute(params)`` routing.

    Same shape as ``_Tau2InvokeMixin`` in airline / retail / user-side
    telecom files. Subclasses implement ``_do_work(p)``.
    """

    PARAMS_MODEL: ClassVar[Optional[Type[BaseModel]]] = None

    def invoke(self, **kwargs) -> Any:
        try:
            p = self.PARAMS_MODEL.model_validate(kwargs) if self.PARAMS_MODEL else None
        except ValidationError as exc:
            return validation_error_response(exc)
        try:
            return self._do_work(p)
        except ValueError as exc:
            # Upstream raises ValueError for not-found / state-violation
            # paths. Convert to a structured error so the LLM gets a
            # usable signal instead of a 500-shaped exception.
            return {"status": "error", "error_type": "tool_error", "message": str(exc)}

    async def _execute(self, params: FunctionCallParams) -> None:
        result = self.invoke(**(params.arguments or {}))
        # Guard against pipecat masking a falsy result (e.g. an empty match
        # list) as the literal "COMPLETED"; the LLM would read that as success.
        await params.result_callback(normalize_tool_result(result))

    def _do_work(self, p) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError(f"{type(self).__name__} must implement _do_work(p)")


class _Tau2TelecomAgentReadTool(_Tau2InvokeMixin, StandardSchemaTool):
    """Base for telecom agent-side read-only tools."""

    DESCRIPTION: ClassVar[str] = ""

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or self.DESCRIPTION)
        self.state = shared_state if shared_state is not None else {}


class _Tau2TelecomAgentWriteTool(_Tau2InvokeMixin, WriteScenarioTool):
    """Base for telecom agent-side write tools."""

    ACTION_TYPES = TAU2_TELECOM_AGENT_ACTION_TYPES
    DESCRIPTION: ClassVar[str] = ""

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or self.DESCRIPTION)
        self.state = shared_state if shared_state is not None else {}


def _db_not_initialized() -> dict:
    return {
        "status": "error",
        "error_type": "db_not_initialized",
        "message": "Scenario agent-side database not loaded. This indicates a fixture-loading bug.",
    }


# =============================================================================
# Customer lookup (read)
# =============================================================================


@register_schema_tool_for_eval(domain="tau2_telecom")
class GetCustomerByPhoneTool(_Tau2TelecomAgentReadTool):
    name = "get_customer_by_phone"
    DESCRIPTION = (
        "Finds a customer by their primary contact or line phone number. "
        "Returns the customer record if found, or an error if not."
    )
    PARAMS_MODEL = GetCustomerByPhoneParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "phone_number": {"type": "string", "description": "The phone number to search for."},
        }

    @property
    def required_properties(self) -> List[str]:
        return ["phone_number"]

    def _do_work(self, p) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        # Match primary contact OR any line phone number on the customer's lines.
        for c in db["customers"]:
            if c["phone_number"] == p.phone_number:
                return _customer_to_user_view(c)
            for lid in c["line_ids"]:
                line = _find_line(db, lid)
                if line and line["phone_number"] == p.phone_number:
                    return _customer_to_user_view(c)
        raise ValueError(f"Customer with phone number {p.phone_number} not found")


@register_schema_tool_for_eval(domain="tau2_telecom")
class GetCustomerByIdTool(_Tau2TelecomAgentReadTool):
    name = "get_customer_by_id"
    DESCRIPTION = "Retrieves a customer directly by their unique customer ID."
    PARAMS_MODEL = GetCustomerByIdParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "customer_id": {"type": "string", "description": "Customer ID (e.g., 'C1001')."},
        }

    @property
    def required_properties(self) -> List[str]:
        return ["customer_id"]

    def _do_work(self, p) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        c = _find_customer(db, p.customer_id)
        if c is None:
            raise ValueError(f"Customer with ID {p.customer_id} not found")
        return _customer_to_user_view(c)


@register_schema_tool_for_eval(domain="tau2_telecom")
class GetCustomerByNameTool(_Tau2TelecomAgentReadTool):
    name = "get_customer_by_name"
    DESCRIPTION = (
        "Searches for customers by full name + date of birth. DOB is required "
        "to disambiguate customers sharing a name. Returns a list of matching "
        "customer records (usually 0 or 1)."
    )
    PARAMS_MODEL = GetCustomerByNameParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "full_name": {"type": "string", "description": "Customer's full name."},
            "dob": {"type": "string", "description": "Date of birth in YYYY-MM-DD format."},
        }

    @property
    def required_properties(self) -> List[str]:
        return ["full_name", "dob"]

    def _do_work(self, p) -> List[dict]:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        full_name_lc = p.full_name.lower()
        matches = [
            _customer_to_user_view(c)
            for c in db["customers"]
            if c["full_name"].lower() == full_name_lc and c["date_of_birth"] == p.dob
        ]
        return matches


@register_schema_tool_for_eval(domain="tau2_telecom")
class GetDetailsByIdTool(_Tau2TelecomAgentReadTool):
    name = "get_details_by_id"
    DESCRIPTION = (
        "Retrieves the details for a given ID. The ID must be a valid ID for "
        "a Customer (C-prefix), Line (L-prefix), Device (D-prefix), Bill "
        "(B-prefix), or Plan (P-prefix)."
    )
    PARAMS_MODEL = GetDetailsByIdParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "id": {
                "type": "string",
                "description": "ID with prefix C/L/D/B/P (e.g., 'C1001', 'L1002', 'B1003').",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["id"]

    def _do_work(self, p) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        obj_id = p.id
        if obj_id.startswith("L"):
            obj = _find_line(db, obj_id)
        elif obj_id.startswith("D"):
            obj = _find_device(db, obj_id)
        elif obj_id.startswith("B"):
            obj = _find_bill(db, obj_id)
        elif obj_id.startswith("C"):
            obj = _find_customer(db, obj_id)
        elif obj_id.startswith("P"):
            obj = _find_plan(db, obj_id)
        else:
            raise ValueError(f"Unknown ID format or type: {obj_id}")
        if obj is None:
            raise ValueError(f"Object with ID {obj_id} not found")
        return dict(obj)


@register_schema_tool_for_eval(domain="tau2_telecom")
class GetBillsForCustomerTool(_Tau2TelecomAgentReadTool):
    name = "get_bills_for_customer"
    DESCRIPTION = (
        "Retrieves a list of the customer's bills, most-recent-first. Optionally limits the number of bills returned."
    )
    PARAMS_MODEL = GetBillsForCustomerParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "customer_id": {"type": "string", "description": "Customer ID."},
            "limit": {
                "type": "integer",
                "description": "Maximum number of bills to return (default 12).",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["customer_id"]

    def _do_work(self, p) -> List[dict]:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        customer = _find_customer(db, p.customer_id)
        if customer is None:
            raise ValueError(f"Customer with ID {p.customer_id} not found")
        bills = [_find_bill(db, bid) for bid in customer["bill_ids"]]
        bills = [b for b in bills if b is not None]
        bills.sort(key=lambda b: b["issue_date"], reverse=True)
        return [dict(b) for b in bills[: p.limit]]


@register_schema_tool_for_eval(domain="tau2_telecom")
class GetDataUsageTool(_Tau2TelecomAgentReadTool):
    name = "get_data_usage"
    DESCRIPTION = (
        "Retrieves current billing cycle data usage for a line, including data "
        "refueling amount, data limit, and cycle end date."
    )
    PARAMS_MODEL = GetDataUsageParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "customer_id": {"type": "string", "description": "Customer ID."},
            "line_id": {"type": "string", "description": "Line ID."},
        }

    @property
    def required_properties(self) -> List[str]:
        return ["customer_id", "line_id"]

    def _do_work(self, p) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        line = _get_target_line(db, p.customer_id, p.line_id)
        plan = _find_plan(db, line["plan_id"])
        if plan is None:
            raise ValueError(f"Plan {line['plan_id']} not found for line {p.line_id}")
        today = _get_today()
        # End of current month (last day) — mirrors upstream cycle_end_date.
        if today.month < 12:
            next_month = date(today.year, today.month + 1, 1)
        else:
            next_month = date(today.year + 1, 1, 1)
        cycle_end_date = next_month - timedelta(days=1)
        return {
            "line_id": p.line_id,
            "data_used_gb": line["data_used_gb"],
            "data_limit_gb": plan["data_limit_gb"],
            "data_refueling_gb": line["data_refueling_gb"],
            "cycle_end_date": cycle_end_date.isoformat(),
        }


# =============================================================================
# Line management (write)
# =============================================================================


@register_schema_tool_for_eval(domain="tau2_telecom")
class SuspendLineTool(_Tau2TelecomAgentWriteTool):
    name = "suspend_line"
    DESCRIPTION = (
        "Suspends a specific Active line (max 6 months). Sets line status to "
        "Suspended and records suspension_start_date. Fails if the line is "
        "not currently Active."
    )
    PARAMS_MODEL = SuspendLineParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "customer_id": {"type": "string", "description": "Customer ID who owns the line."},
            "line_id": {"type": "string", "description": "Line ID to suspend."},
            "reason": {"type": "string", "description": "Reason for the suspension."},
        }

    @property
    def required_properties(self) -> List[str]:
        return ["customer_id", "line_id", "reason"]

    def _do_work(self, p) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        line = _get_target_line(db, p.customer_id, p.line_id)
        if line["status"] != LineStatus.ACTIVE.value:
            raise ValueError("Line must be active to suspend")
        line["status"] = LineStatus.SUSPENDED.value
        line["suspension_start_date"] = _get_today().isoformat()
        self._record_action(
            {
                "action_type": "suspend_line",
                "name": "suspend_line",
                "arguments": {
                    "customer_id": p.customer_id,
                    "line_id": p.line_id,
                    "reason": p.reason,
                },
                "result": {"line_id": p.line_id, "status": LineStatus.SUSPENDED.value},
            }
        )
        return {
            "message": "Line suspended successfully. $5/month holding fee will apply.",
            "line": dict(line),
        }


@register_schema_tool_for_eval(domain="tau2_telecom")
class ResumeLineTool(_Tau2TelecomAgentWriteTool):
    name = "resume_line"
    DESCRIPTION = (
        "Resumes a Suspended or Pending Activation line. Sets line status to Active and clears suspension_start_date."
    )
    PARAMS_MODEL = ResumeLineParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "customer_id": {"type": "string", "description": "Customer ID who owns the line."},
            "line_id": {"type": "string", "description": "Line ID to resume."},
        }

    @property
    def required_properties(self) -> List[str]:
        return ["customer_id", "line_id"]

    def _do_work(self, p) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        line = _get_target_line(db, p.customer_id, p.line_id)
        if line["status"] not in (LineStatus.SUSPENDED.value, LineStatus.PENDING_ACTIVATION.value):
            raise ValueError("Line must be suspended to resume")
        line["status"] = LineStatus.ACTIVE.value
        line["suspension_start_date"] = None
        self._record_action(
            {
                "action_type": "resume_line",
                "name": "resume_line",
                "arguments": {"customer_id": p.customer_id, "line_id": p.line_id},
                "result": {"line_id": p.line_id, "status": LineStatus.ACTIVE.value},
            }
        )
        return {"message": "Line resumed successfully", "line": dict(line)}


@register_schema_tool_for_eval(domain="tau2_telecom")
class EnableRoamingTool(_Tau2TelecomAgentWriteTool):
    name = "enable_roaming"
    DESCRIPTION = "Enables international roaming on a line."
    PARAMS_MODEL = EnableRoamingParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "customer_id": {"type": "string", "description": "Customer ID."},
            "line_id": {"type": "string", "description": "Line ID."},
        }

    @property
    def required_properties(self) -> List[str]:
        return ["customer_id", "line_id"]

    def _do_work(self, p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        line = _get_target_line(db, p.customer_id, p.line_id)
        if line["roaming_enabled"]:
            return "Roaming was already enabled"
        line["roaming_enabled"] = True
        self._record_action(
            {
                "action_type": "enable_roaming",
                "name": "enable_roaming",
                "arguments": {"customer_id": p.customer_id, "line_id": p.line_id},
                "result": {"roaming_enabled": True},
            }
        )
        return "Roaming enabled successfully"


@register_schema_tool_for_eval(domain="tau2_telecom")
class DisableRoamingTool(_Tau2TelecomAgentWriteTool):
    name = "disable_roaming"
    DESCRIPTION = "Disables international roaming on a line."
    PARAMS_MODEL = DisableRoamingParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "customer_id": {"type": "string", "description": "Customer ID."},
            "line_id": {"type": "string", "description": "Line ID."},
        }

    @property
    def required_properties(self) -> List[str]:
        return ["customer_id", "line_id"]

    def _do_work(self, p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        line = _get_target_line(db, p.customer_id, p.line_id)
        if not line["roaming_enabled"]:
            return "Roaming was already disabled"
        line["roaming_enabled"] = False
        self._record_action(
            {
                "action_type": "disable_roaming",
                "name": "disable_roaming",
                "arguments": {"customer_id": p.customer_id, "line_id": p.line_id},
                "result": {"roaming_enabled": False},
            }
        )
        return "Roaming disabled successfully"


# =============================================================================
# Billing (write)
# =============================================================================


def _bills_awaiting_payment(db: dict, customer: dict) -> List[dict]:
    return [
        b
        for bid in customer["bill_ids"]
        for b in (_find_bill(db, bid),)
        if b is not None and b["status"] == BillStatus.AWAITING_PAYMENT.value
    ]


@register_schema_tool_for_eval(domain="tau2_telecom")
class SendPaymentRequestTool(_Tau2TelecomAgentWriteTool):
    name = "send_payment_request"
    DESCRIPTION = (
        "Sends a payment request to the customer for a specific bill. Sets the "
        "bill to AWAITING_PAYMENT. Fails if another bill is already awaiting "
        "payment for this customer (one open request at a time)."
    )
    PARAMS_MODEL = SendPaymentRequestParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "customer_id": {"type": "string", "description": "Customer ID."},
            "bill_id": {"type": "string", "description": "Bill ID."},
        }

    @property
    def required_properties(self) -> List[str]:
        return ["customer_id", "bill_id"]

    def _do_work(self, p) -> str:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        customer = _find_customer(db, p.customer_id)
        if customer is None:
            raise ValueError(f"Customer {p.customer_id} not found")
        if _bills_awaiting_payment(db, customer):
            raise ValueError("A bill is already awaiting payment for this customer")
        if p.bill_id not in customer["bill_ids"]:
            raise ValueError(f"Bill {p.bill_id} not found for customer {p.customer_id}")
        bill = _find_bill(db, p.bill_id)
        bill["status"] = BillStatus.AWAITING_PAYMENT.value
        # Also surface the payment request to the user-side via the user
        # bot's `check_payment_request` tool. The user-side flow reads
        # ``surroundings["payment_request"]`` (set by upstream's
        # ``send_payment_request`` indirectly through the env coupling).
        # We don't have direct access to the user bot's DB here, so the
        # bridge handles cross-side propagation via the runner's
        # initialization_actions / db_state_assertions plumbing. For
        # now, mark the bill and trust the per-scenario user_db state
        # to carry the payment_request when the task requires it.
        self._record_action(
            {
                "action_type": "send_payment_request",
                "name": "send_payment_request",
                "arguments": {"customer_id": p.customer_id, "bill_id": p.bill_id},
                "result": {"bill_id": p.bill_id, "status": BillStatus.AWAITING_PAYMENT.value},
            }
        )
        return f"Payment request sent to the customer for bill {p.bill_id}"


@register_schema_tool_for_eval(domain="tau2_telecom")
class RefuelDataTool(_Tau2TelecomAgentWriteTool):
    name = "refuel_data"
    DESCRIPTION = (
        "Refuels data for a specific line with a positive GB amount. Adds a "
        "one-time charge to the customer's next bill at the plan's per-GB "
        "refueling rate."
    )
    PARAMS_MODEL = RefuelDataParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "customer_id": {"type": "string", "description": "Customer ID."},
            "line_id": {"type": "string", "description": "Line ID."},
            "gb_amount": {"type": "number", "description": "Amount of data to add, in GB (positive)."},
        }

    @property
    def required_properties(self) -> List[str]:
        return ["customer_id", "line_id", "gb_amount"]

    def _do_work(self, p) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        if p.gb_amount <= 0:
            raise ValueError("Refuel amount must be positive")
        line = _get_target_line(db, p.customer_id, p.line_id)
        plan = _find_plan(db, line["plan_id"])
        if plan is None:
            raise ValueError("Plan not found for this line")
        charge_amount = p.gb_amount * plan["data_refueling_price_per_gb"]
        line["data_refueling_gb"] = line.get("data_refueling_gb", 0.0) + p.gb_amount
        # Apply charge — upstream creates a draft bill if none exists.
        # We mirror the addition to the customer's draft bill total_due
        # without minting new bill IDs (live-runtime DB is what the
        # gold-replay compares against via hashing; adding a synthetic
        # bill_id would diverge from upstream's UUID-generated value).
        self._apply_one_time_charge(
            db,
            p.customer_id,
            charge_amount,
            f"Data refueling: {p.gb_amount} GB at ${plan['data_refueling_price_per_gb']}/GB",
        )
        self._record_action(
            {
                "action_type": "refuel_data",
                "name": "refuel_data",
                "arguments": {
                    "customer_id": p.customer_id,
                    "line_id": p.line_id,
                    "gb_amount": p.gb_amount,
                },
                "result": {
                    "new_data_refueling_gb": line["data_refueling_gb"],
                    "charge": charge_amount,
                },
            }
        )
        return {
            "message": (f"Successfully added {p.gb_amount} GB of data for line {p.line_id} for ${charge_amount:.2f}"),
            "new_data_refueling_gb": line["data_refueling_gb"],
            "charge": charge_amount,
        }

    @staticmethod
    def _apply_one_time_charge(db: dict, customer_id: str, amount: float, description: str) -> None:
        """Mirrors upstream ``TelecomTools._apply_one_time_charge``.

        Finds an existing DRAFT bill for the customer, or creates one.
        Appends a charge line item and updates ``total_due``. Does NOT
        mint a stable bill_id when creating — upstream uses ``uuid4().hex[:8]``,
        which is non-deterministic; we use the same shape so the hash
        comparison sees an extra-bill structural change but not a
        deterministic ID match. Scenarios that exercise refuel_data
        should not depend on the new bill_id text.
        """
        import uuid

        customer = _find_customer(db, customer_id)
        if customer is None:
            raise ValueError(f"Customer {customer_id} not found")
        draft_bill = None
        for bid in customer["bill_ids"]:
            b = _find_bill(db, bid)
            if b and b["status"] == BillStatus.DRAFT.value:
                draft_bill = b
                break
        if draft_bill is None:
            today = _get_today()
            if today.month < 12:
                next_month = date(today.year, today.month + 1, 1)
                period_end_month_first = (
                    date(today.year, today.month + 2, 1) if today.month < 11 else date(today.year + 1, 1, 1)
                )
            else:
                next_month = date(today.year + 1, 1, 1)
                period_end_month_first = date(today.year + 1, 2, 1)
            new_bill_id = f"B{uuid.uuid4().hex[:8]}"
            draft_bill = {
                "bill_id": new_bill_id,
                "customer_id": customer_id,
                "period_start": next_month.isoformat(),
                "period_end": (period_end_month_first - timedelta(days=1)).isoformat(),
                "issue_date": next_month.isoformat(),
                "total_due": 0,
                "due_date": (next_month + timedelta(days=14)).isoformat(),
                "line_items": [],
                "status": BillStatus.DRAFT.value,
            }
            db["bills"].append(draft_bill)
            customer["bill_ids"].append(new_bill_id)
        draft_bill["line_items"].append(
            {
                "description": description,
                "amount": amount,
                "date": _get_today().isoformat(),
                "item_type": "Credit" if amount < 0 else "Charge",
            }
        )
        draft_bill["total_due"] = float(draft_bill.get("total_due", 0)) + amount


# =============================================================================
# Escalation (generic)
# =============================================================================


@register_schema_tool_for_eval(domain="tau2_telecom")
class TransferToHumanAgentsTool(_Tau2TelecomAgentWriteTool):
    name = "transfer_to_human_agents"
    DESCRIPTION = (
        "Transfer the user to a human agent, with a summary of the user's "
        "issue. Only transfer if the user explicitly asks for a human agent, "
        "or if (given the policy and the available tools) you cannot solve "
        "the user's issue."
    )
    PARAMS_MODEL = TransferToHumanAgentsParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "summary": {"type": "string", "description": "A summary of the user's issue."},
        }

    @property
    def required_properties(self) -> List[str]:
        return ["summary"]

    def _do_work(self, p) -> str:
        # Record-only — no DB mutation, but we still want the action
        # to appear in the pulled action list so the runner can match
        # against ``transfer_to_human_agents`` reference actions.
        self._record_action(
            {
                "action_type": "transfer_to_human_agents",
                "name": "transfer_to_human_agents",
                "arguments": {"summary": p.summary},
                "result": {"transferred": True},
            }
        )
        return "Transfer successful"

    async def _execute(self, params: FunctionCallParams) -> None:
        await super()._execute(params)
        await self._send_exit_message()


# =============================================================================
# Name → class map (consumed by Tau2TelecomBaseScenario._build_tool_map for
# agent-side gold-replay dispatch + agent_resources.tools listing)
# =============================================================================


TAU2_TELECOM_AGENT_TOOL_NAME_TO_CLASS: Dict[str, Type[StandardSchemaTool]] = {
    cls.name: cls
    for cls in (
        # Reads (6)
        GetCustomerByPhoneTool,
        GetCustomerByIdTool,
        GetCustomerByNameTool,
        GetDetailsByIdTool,
        GetBillsForCustomerTool,
        GetDataUsageTool,
        # Writes (6)
        SuspendLineTool,
        ResumeLineTool,
        EnableRoamingTool,
        DisableRoamingTool,
        SendPaymentRequestTool,
        RefuelDataTool,
        # Generic (1)
        TransferToHumanAgentsTool,
    )
}

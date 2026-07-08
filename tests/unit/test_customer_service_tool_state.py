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

"""Unit tests for customer-service evaluation tools over shared-state dictionaries."""

import asyncio
import json

from nemo_voice_agent.evaluation.tools.customer_service_tools import (
    CancelSubscriptionTool,
    ChangePlanTool,
    CheckOrderStatusTool,
    LookupAccountTool,
    ProcessRefundTool,
    ResolveTicketTool,
    StartItemReturnTool,
    UnlockAccountTool,
    _format_money,
    _parse_money,
)


class _Params:
    """Duck-typed FunctionCallParams replacement for customer-service tool tests."""

    def __init__(self, arguments=None):
        """Create params with arguments and callback result capture."""
        self.arguments = arguments or {}
        self.results = []

    async def result_callback(self, result):
        """Capture one tool result."""
        self.results.append(result)


class _CapturingRTVI:
    """Fake RTVI processor that records scenario-summary transport messages."""

    def __init__(self):
        """Initialize an empty message capture."""
        self.messages = []

    async def push_transport_message(self, message, exclude_none=True):
        """Record the pushed message object and exclude_none flag."""
        self.messages.append((message, exclude_none))


def _state():
    """Return a small customer account state fixture."""
    return {
        "accounts": {
            "acct-1": {
                "name": "Ada",
                "balance": "$100.00",
                "plan": "Basic",
                "monthly_rate": "$10.00",
                "account_status": "Locked",
                "failed_login_attempts": "3",
                "recent_charges": [{"description": "Monthly service", "amount": "$10.00"}],
                "orders": {
                    "order-1": {"status": "Shipped", "items": ["router"]},
                },
            }
        }
    }


def test_money_helpers_parse_and_format_signed_currency_values():
    """Customer-service money helpers normalize commas, dollar signs, and signs."""
    assert _parse_money("$1,234.56") == 1234.56
    assert _parse_money("-$49.99") == -49.99
    assert _format_money(1234.5) == "$1,234.50"
    assert _format_money(-49.99) == "-$49.99"


def test_lookup_account_tool_returns_account_or_not_found_error():
    """LookupAccountTool reads configured accounts and reports missing IDs."""
    state = _state()
    tool = LookupAccountTool(shared_state=state)
    ok_params = _Params({"account_id": "acct-1"})
    missing_params = _Params({"account_id": "missing"})

    asyncio.run(tool._execute(ok_params))
    asyncio.run(tool._execute(missing_params))

    assert ok_params.results == [state["accounts"]["acct-1"]]
    assert missing_params.results == [{"error": "Account 'missing' not found."}]


def test_check_order_status_tool_handles_account_order_and_missing_paths():
    """CheckOrderStatusTool returns nested orders and distinguishes missing account/order errors."""
    state = _state()
    tool = CheckOrderStatusTool(shared_state=state)
    ok_params = _Params({"account_id": "acct-1", "order_id": "order-1"})
    missing_order_params = _Params({"account_id": "acct-1", "order_id": "missing"})
    missing_account_params = _Params({"account_id": "missing", "order_id": "order-1"})

    asyncio.run(tool._execute(ok_params))
    asyncio.run(tool._execute(missing_order_params))
    asyncio.run(tool._execute(missing_account_params))

    assert ok_params.results == [{"status": "Shipped", "items": ["router"]}]
    assert missing_order_params.results == [{"error": "Order 'missing' not found for account 'acct-1'."}]
    assert missing_account_params.results == [{"error": "Account 'missing' not found."}]


def test_process_refund_tool_appends_charge_and_updates_balance():
    """ProcessRefundTool records a negative charge entry and decrements the account balance."""
    state = _state()
    tool = ProcessRefundTool(shared_state=state)
    params = _Params(
        {
            "account_id": "acct-1",
            "amount": "25.50",
            "description": "late fee",
            "date": "2026-07-08",
        }
    )

    asyncio.run(tool._execute(params))

    account = state["accounts"]["acct-1"]
    assert account["balance"] == "$74.50"
    assert account["recent_charges"][-1] == {
        "description": "Refund - late fee",
        "amount": "-$25.50",
        "date": "2026-07-08",
    }
    assert params.results[0]["new_balance"] == "$74.50"


def test_start_item_return_tool_updates_order_or_reports_missing_paths():
    """StartItemReturnTool mutates order status and handles missing account/order branches."""
    state = _state()
    tool = StartItemReturnTool(shared_state=state)
    ok_params = _Params({"account_id": "acct-1", "order_id": "order-1", "reason": "defective"})
    missing_account_params = _Params({"account_id": "missing", "order_id": "order-1", "reason": "defective"})
    missing_order_params = _Params({"account_id": "acct-1", "order_id": "missing", "reason": "defective"})

    asyncio.run(tool._execute(ok_params))
    asyncio.run(tool._execute(missing_account_params))
    asyncio.run(tool._execute(missing_order_params))

    assert state["accounts"]["acct-1"]["orders"]["order-1"]["status"] == "Return Started"
    assert state["accounts"]["acct-1"]["orders"]["order-1"]["return_reason"] == "defective"
    assert ok_params.results[0]["success"] is True
    assert missing_account_params.results == [{"error": "Account 'missing' not found."}]
    assert missing_order_params.results == [{"error": "Order 'missing' not found for account 'acct-1'."}]


def test_change_plan_tool_updates_plan_and_reports_unavailable_plan():
    """ChangePlanTool updates plan/rate from configured plans and reports invalid choices."""
    state = _state()
    tool = ChangePlanTool(shared_state=state, plans={"Premium": "$30.00"})
    ok_params = _Params({"account_id": "acct-1", "new_plan": "Premium"})
    bad_plan_params = _Params({"account_id": "acct-1", "new_plan": "Enterprise"})

    asyncio.run(tool._execute(ok_params))
    asyncio.run(tool._execute(bad_plan_params))

    assert state["accounts"]["acct-1"]["plan"] == "Premium"
    assert state["accounts"]["acct-1"]["monthly_rate"] == "$30.00"
    assert ok_params.results[0]["new_monthly_rate"] == "$30.00"
    assert bad_plan_params.results == [{"error": "Plan 'Enterprise' is not available. Available plans: ['Premium']."}]


def test_unlock_and_cancel_subscription_tools_mutate_account_status():
    """UnlockAccountTool and CancelSubscriptionTool update simple account status fields."""
    state = _state()
    unlock_params = _Params({"account_id": "acct-1"})
    cancel_params = _Params({"account_id": "acct-1"})

    asyncio.run(UnlockAccountTool(shared_state=state)._execute(unlock_params))
    asyncio.run(CancelSubscriptionTool(shared_state=state)._execute(cancel_params))

    account = state["accounts"]["acct-1"]
    assert account["account_status"] == "Active"
    assert account["failed_login_attempts"] == "0"
    assert account["plan"] == "Canceled"
    assert account["monthly_rate"] == "$0.00"
    assert unlock_params.results[0]["success"] is True
    assert cancel_params.results[0]["success"] is True


def test_resolve_ticket_tool_sends_summary_with_account_snapshot():
    """ResolveTicketTool sends a final-response summary containing the latest account snapshot."""
    state = _state()
    rtvi = _CapturingRTVI()
    tool = ResolveTicketTool(rtvi=rtvi, shared_state=state, resolution_types=["refund"])
    params = _Params(
        {
            "account_id": "acct-1",
            "issue_summary": "Billing issue",
            "resolution_type": "refund",
            "resolution_details": "Issued a refund.",
        }
    )

    asyncio.run(tool._execute(params))

    message = rtvi.messages[0][0].data.text
    summary = json.loads(message.removeprefix("<final_response>").removesuffix("</final_response>"))
    assert summary["account_id"] == "acct-1"
    assert summary["account"]["name"] == "Ada"
    assert params.results == [{"success": True, "message": "Ticket resolved successfully."}]

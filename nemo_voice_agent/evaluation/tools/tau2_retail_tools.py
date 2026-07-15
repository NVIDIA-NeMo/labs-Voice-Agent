# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
#   src/tau2/domains/retail/tools.py — all 16 @is_tool methods of RetailTools
#       ported as StandardSchemaTool / WriteScenarioTool subclasses. Internal
#       helpers (``_get_order``, ``_get_user``, ``_get_payment_method``,
#       ``_is_pending_order``) are reproduced as module-level functions
#       operating on the dict-form DB.

"""Tau2-retail tool ports.

Same architecture as ``tau2_airline_tools.py``: dict-form DB in ``self.state["db"]``,
``_Tau2InvokeMixin`` for sync/async dual-routing, ``WriteScenarioTool`` subclass
records each mutation into ``self.state["actions"]``.
"""

# pylint: disable=line-too-long
# flake8: noqa: E501

import json
from typing import Any, ClassVar, Dict, List, Optional, Type

from pipecat.services.llm_service import FunctionCallParams
from pydantic import BaseModel, ValidationError

from nemo_voice_agent.evaluation.tools import normalize_tool_result, register_schema_tool_for_eval
from nemo_voice_agent.evaluation.tools._write_tool_base import WriteScenarioTool
from nemo_voice_agent.evaluation.tools.tau2_retail_params import (
    CalculateParams,
    CancelPendingOrderParams,
    ExchangeDeliveredOrderItemsParams,
    FindUserIdByEmailParams,
    FindUserIdByNameZipParams,
    GetItemDetailsParams,
    GetOrderDetailsParams,
    GetProductDetailsParams,
    GetUserDetailsParams,
    ListAllProductTypesParams,
    ModifyPendingOrderAddressParams,
    ModifyPendingOrderItemsParams,
    ModifyPendingOrderPaymentParams,
    ModifyUserAddressParams,
    ReturnDeliveredOrderItemsParams,
    TransferToHumanAgentsParams,
    validation_error_response,
)
from nemo_voice_agent.utils.tool_calling import StandardSchemaTool


# =============================================================================
# Action-type vocabulary
# Names match tau2's @is_tool method names 1:1 so the gold-replay's records
# align with what we'd later wire into action-list comparison.
# =============================================================================


TAU2_RETAIL_ACTION_TYPES: List[str] = [
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
    "transfer_to_human_agents",
]


# =============================================================================
# DB-dict helpers
# =============================================================================


def _db_not_initialized() -> dict:
    return {
        "status": "error",
        "error_type": "db_not_initialized",
        "message": "Scenario database not loaded. This indicates a fixture-loading bug.",
    }


def _get_user_dict(db: dict, user_id: str) -> Optional[dict]:
    # tau2 retail user_ids are lowercase in db.json (e.g. "ivan_hernandez_6923");
    # voice ASR can emit mixed case. Normalize on lookup.
    return db.get("users", {}).get(user_id.lower() if user_id else user_id)


def _get_order_dict(db: dict, order_id: str) -> Optional[dict]:
    # tau2 retail order_ids are uppercase in db.json (e.g. "#W0000000").
    # The leading '#' is always present in tau2's schema; users sometimes drop
    # it when speaking. Normalize: uppercase + prepend '#' if missing.
    if not order_id:
        return None
    normalized = order_id.upper()
    if not normalized.startswith("#"):
        normalized = "#" + normalized
    return db.get("orders", {}).get(normalized)


def _get_product_dict(db: dict, product_id: str) -> Optional[dict]:
    return db.get("products", {}).get(product_id)


def _get_item_dict(db: dict, item_id: str) -> Optional[dict]:
    """Find a variant by item_id across all products (mirrors tau2's _get_item)."""
    if not item_id:
        return None
    for product in (db.get("products") or {}).values():
        variants = product.get("variants") or {}
        if item_id in variants:
            return variants[item_id]
    return None


def _get_variant_dict(db: dict, product_id: str, variant_id: str) -> Optional[dict]:
    product = _get_product_dict(db, product_id)
    if product is None:
        return None
    return (product.get("variants") or {}).get(variant_id)


def _get_payment_method_dict(user: dict, payment_method_id: str) -> Optional[dict]:
    return (user.get("payment_methods") or {}).get(payment_method_id)


def _is_pending_order(order: dict) -> bool:
    """Mirror tau2: any status containing 'pending' counts as pending for some checks."""
    return "pending" in (order.get("status") or "")


def _round2(value: float) -> float:
    return round(float(value), 2)


# =============================================================================
# Base classes
# =============================================================================


class _Tau2InvokeMixin:
    """Provides sync ``invoke(**kwargs)`` + async ``_execute(params)`` routing.

    Subclasses set ``PARAMS_MODEL = MyParams`` and implement ``_do_work(p)``.
    """

    PARAMS_MODEL: ClassVar[Optional[Type[BaseModel]]] = None

    def invoke(self, **kwargs) -> dict:
        """Synchronous entry — used by ``Tau2BaseScenario._gold_replay``."""
        try:
            p = self.PARAMS_MODEL.model_validate(kwargs) if self.PARAMS_MODEL else None
        except ValidationError as exc:
            return validation_error_response(exc)
        return self._do_work(p)

    async def _execute(self, params: FunctionCallParams) -> None:
        result = self.invoke(**(params.arguments or {}))
        # Guard against pipecat masking a falsy result (e.g. an empty match
        # list) as the literal "COMPLETED"; the LLM would read that as success.
        await params.result_callback(normalize_tool_result(result))

    def _do_work(self, p) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError(f"{type(self).__name__} must implement _do_work(p)")


class _Tau2ReadTool(_Tau2InvokeMixin, StandardSchemaTool):
    """Base for retail read-only tools."""

    DESCRIPTION: ClassVar[str] = ""

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or self.DESCRIPTION)
        self.state = shared_state if shared_state is not None else {}


class _Tau2WriteTool(_Tau2InvokeMixin, WriteScenarioTool):
    """Base for retail write tools (includes generic action-recording tools)."""

    ACTION_TYPES = TAU2_RETAIL_ACTION_TYPES
    DESCRIPTION: ClassVar[str] = ""

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or self.DESCRIPTION)
        self.state = shared_state if shared_state is not None else {}


# =============================================================================
# Read tools
# =============================================================================


@register_schema_tool_for_eval(domain="tau2_retail")
class FindUserIdByNameZipTool(_Tau2ReadTool):
    DESCRIPTION = (
        "Find user id by first name, last name, and zip code. If the user is not "
        "found, the function will return an error message. By default, find user "
        "id by email, and only call this function if the user is not found by "
        "email or cannot remember email."
    )
    PARAMS_MODEL = FindUserIdByNameZipParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "first_name": {"type": "string", "description": "The first name of the customer, such as 'John'."},
            "last_name": {"type": "string", "description": "The last name of the customer, such as 'Doe'."},
            "zip": {"type": "string", "description": "The zip code of the customer, such as '12345'."},
        }

    @property
    def required_properties(self) -> List[str]:
        return ["first_name", "last_name", "zip"]

    def _do_work(self, p: FindUserIdByNameZipParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        for user_id, user in (db.get("users") or {}).items():
            name = user.get("name") or {}
            address = user.get("address") or {}
            if (
                (name.get("first_name") or "").lower() == p.first_name.lower()
                and (name.get("last_name") or "").lower() == p.last_name.lower()
                and (address.get("zip") or "") == p.zip
            ):
                return {"status": "success", "user_id": user_id}
        return {"status": "error", "error_type": "not_found", "message": "User not found"}


@register_schema_tool_for_eval(domain="tau2_retail")
class FindUserIdByEmailTool(_Tau2ReadTool):
    DESCRIPTION = "Find user id by email. If the user is not found, the function will return an error message."
    PARAMS_MODEL = FindUserIdByEmailParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {"email": {"type": "string", "description": "The email of the user, such as 'something@example.com'."}}

    @property
    def required_properties(self) -> List[str]:
        return ["email"]

    def _do_work(self, p: FindUserIdByEmailParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        for user_id, user in (db.get("users") or {}).items():
            if (user.get("email") or "").lower() == p.email.lower():
                return {"status": "success", "user_id": user_id}
        return {"status": "error", "error_type": "not_found", "message": "User not found"}


@register_schema_tool_for_eval(domain="tau2_retail")
class GetOrderDetailsTool(_Tau2ReadTool):
    DESCRIPTION = "Get the status and details of an order."
    PARAMS_MODEL = GetOrderDetailsParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "order_id": {
                "type": "string",
                "description": (
                    "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id."
                ),
            }
        }

    @property
    def required_properties(self) -> List[str]:
        return ["order_id"]

    def _do_work(self, p: GetOrderDetailsParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        order = _get_order_dict(db, p.order_id)
        if order is None:
            return {"status": "error", "error_type": "not_found", "message": f"Order {p.order_id} not found"}
        return order


@register_schema_tool_for_eval(domain="tau2_retail")
class GetProductDetailsTool(_Tau2ReadTool):
    DESCRIPTION = "Get the inventory details of a product."
    PARAMS_MODEL = GetProductDetailsParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "product_id": {
                "type": "string",
                "description": (
                    "The product id, such as '6086499569'. Be careful the product id is different from the item id."
                ),
            }
        }

    @property
    def required_properties(self) -> List[str]:
        return ["product_id"]

    def _do_work(self, p: GetProductDetailsParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        product = _get_product_dict(db, p.product_id)
        if product is None:
            return {"status": "error", "error_type": "not_found", "message": f"Product {p.product_id} not found"}
        return product


@register_schema_tool_for_eval(domain="tau2_retail")
class GetItemDetailsTool(_Tau2ReadTool):
    DESCRIPTION = "Get the inventory details of an item."
    PARAMS_MODEL = GetItemDetailsParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "item_id": {
                "type": "string",
                "description": (
                    "The item id, such as '6086499569'. Be careful the item id is different from the product id."
                ),
            }
        }

    @property
    def required_properties(self) -> List[str]:
        return ["item_id"]

    def _do_work(self, p: GetItemDetailsParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        item = _get_item_dict(db, p.item_id)
        if item is None:
            return {"status": "error", "error_type": "not_found", "message": f"Item {p.item_id} not found"}
        return item


@register_schema_tool_for_eval(domain="tau2_retail")
class GetUserDetailsTool(_Tau2ReadTool):
    DESCRIPTION = "Get the details of a user, including their orders."
    PARAMS_MODEL = GetUserDetailsParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {"user_id": {"type": "string", "description": "The user id, such as 'sara_doe_496'."}}

    @property
    def required_properties(self) -> List[str]:
        return ["user_id"]

    def _do_work(self, p: GetUserDetailsParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        user = _get_user_dict(db, p.user_id)
        if user is None:
            return {"status": "error", "error_type": "not_found", "message": f"User {p.user_id} not found"}
        return user


@register_schema_tool_for_eval(domain="tau2_retail")
class ListAllProductTypesTool(_Tau2ReadTool):
    DESCRIPTION = (
        "List the name and product id of all product types. Each product type has a "
        "variety of different items with unique item ids and options. There are only "
        "50 product types in the store."
    )
    PARAMS_MODEL = ListAllProductTypesParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, p: ListAllProductTypesParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        product_dict = {
            product.get("name"): product.get("product_id") for product in (db.get("products") or {}).values()
        }
        return {"status": "success", "product_types": json.dumps(product_dict, sort_keys=True)}


# =============================================================================
# Write tools
# =============================================================================


@register_schema_tool_for_eval(domain="tau2_retail")
class CancelPendingOrderTool(_Tau2WriteTool):
    DESCRIPTION = (
        "Cancel a pending order. If the order is already processed or delivered, it "
        "cannot be cancelled. The agent needs to explain the cancellation detail and "
        "ask for explicit user confirmation (yes/no) to proceed. If the user confirms, "
        "the order status will be changed to 'cancelled' and the payment will be refunded. "
        "The refund will be added to the user's gift card balance immediately if the payment "
        "was made using a gift card, otherwise the refund would take 5-7 business days to process."
    )
    PARAMS_MODEL = CancelPendingOrderParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "order_id": {"type": "string", "description": "The order id, such as '#W0000000'."},
            "reason": {
                "type": "string",
                "enum": ["no longer needed", "ordered by mistake"],
                "description": "The reason for cancellation.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["order_id", "reason"]

    def _do_work(self, p: CancelPendingOrderParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        order = _get_order_dict(db, p.order_id)
        if order is None:
            return {"status": "error", "error_type": "not_found", "message": f"Order {p.order_id} not found"}
        if order.get("status") != "pending":
            return {"status": "error", "error_type": "value_error", "message": "Non-pending order cannot be cancelled"}
        if p.reason not in {"no longer needed", "ordered by mistake"}:
            return {"status": "error", "error_type": "value_error", "message": "Invalid reason"}

        user = _get_user_dict(db, order.get("user_id"))
        if user is None:
            return {"status": "error", "error_type": "not_found", "message": "User not found"}

        # Build refund entries and apply gift-card balance updates
        refunds: List[dict] = []
        for payment in order.get("payment_history") or []:
            payment_id = payment.get("payment_method_id")
            amount = payment.get("amount", 0)
            refunds.append({"transaction_type": "refund", "amount": amount, "payment_method_id": payment_id})
            method = _get_payment_method_dict(user, payment_id)
            if method is not None and method.get("source") == "gift_card":
                method["balance"] = _round2((method.get("balance") or 0) + amount)

        order["status"] = "cancelled"
        order["cancel_reason"] = p.reason
        order.setdefault("payment_history", []).extend(refunds)

        self._record_action(
            {
                "action_type": "cancel_pending_order",
                "name": "cancel_pending_order",
                "arguments": p.model_dump(),
                "result": {"order_id": p.order_id, "status": "cancelled"},
            }
        )
        return order


@register_schema_tool_for_eval(domain="tau2_retail")
class ExchangeDeliveredOrderItemsTool(_Tau2WriteTool):
    DESCRIPTION = (
        "Exchange items in a delivered order to new items of the same product type. "
        "For a delivered order, return or exchange can be only done once by the agent. "
        "The agent needs to explain the exchange detail and ask for explicit user "
        "confirmation (yes/no) to proceed."
    )
    PARAMS_MODEL = ExchangeDeliveredOrderItemsParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "order_id": {"type": "string", "description": "The order id, such as '#W0000000'."},
            "item_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The item ids to be exchanged. There could be duplicate items in the list.",
            },
            "new_item_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "The item ids to be exchanged for. Each new item id should match the item id in the same "
                    "position and be of the same product."
                ),
            },
            "payment_method_id": {
                "type": "string",
                "description": "The payment method id, such as 'gift_card_0000000' or 'credit_card_0000000'.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["order_id", "item_ids", "new_item_ids", "payment_method_id"]

    def _do_work(self, p: ExchangeDeliveredOrderItemsParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        order = _get_order_dict(db, p.order_id)
        if order is None:
            return {"status": "error", "error_type": "not_found", "message": f"Order {p.order_id} not found"}
        if order.get("status") != "delivered":
            return {
                "status": "error",
                "error_type": "value_error",
                "message": "Non-delivered order cannot be exchanged",
            }

        # Validate counts of items-to-exchange
        all_item_ids = [item.get("item_id") for item in order.get("items") or []]
        for item_id in p.item_ids:
            if p.item_ids.count(item_id) > all_item_ids.count(item_id):
                return {"status": "error", "error_type": "value_error", "message": f"Number of {item_id} not found."}
        if len(p.item_ids) != len(p.new_item_ids):
            return {
                "status": "error",
                "error_type": "value_error",
                "message": "The number of items to be exchanged should match.",
            }

        # Validate new items exist within the same product and are available; compute price diff
        diff_price = 0.0
        for item_id, new_item_id in zip(p.item_ids, p.new_item_ids):
            order_item = next((item for item in order.get("items") or [] if item.get("item_id") == item_id), None)
            if order_item is None:
                return {"status": "error", "error_type": "value_error", "message": f"Item {item_id} not found"}
            variant = _get_variant_dict(db, order_item.get("product_id"), new_item_id)
            if variant is None or not variant.get("available", False):
                return {
                    "status": "error",
                    "error_type": "value_error",
                    "message": f"New item {new_item_id} not found or available",
                }
            diff_price += float(variant.get("price", 0)) - float(order_item.get("price", 0))
        diff_price = _round2(diff_price)

        # Validate payment method
        user = _get_user_dict(db, order.get("user_id"))
        if user is None:
            return {"status": "error", "error_type": "not_found", "message": "User not found"}
        payment_method = _get_payment_method_dict(user, p.payment_method_id)
        if payment_method is None:
            return {"status": "error", "error_type": "not_found", "message": "Payment method not found"}
        if payment_method.get("source") == "gift_card" and (payment_method.get("balance") or 0) < diff_price:
            return {
                "status": "error",
                "error_type": "value_error",
                "message": "Insufficient gift card balance to pay for the price difference",
            }

        order["status"] = "exchange requested"
        order["exchange_items"] = sorted(p.item_ids)
        order["exchange_new_items"] = sorted(p.new_item_ids)
        order["exchange_payment_method_id"] = p.payment_method_id
        order["exchange_price_difference"] = diff_price

        self._record_action(
            {
                "action_type": "exchange_delivered_order_items",
                "name": "exchange_delivered_order_items",
                "arguments": p.model_dump(),
                "result": {"order_id": p.order_id, "status": "exchange requested", "price_difference": diff_price},
            }
        )
        return order


@register_schema_tool_for_eval(domain="tau2_retail")
class ModifyPendingOrderAddressTool(_Tau2WriteTool):
    DESCRIPTION = (
        "Modify the shipping address of a pending order. The agent needs to explain "
        "the modification detail and ask for explicit user confirmation (yes/no) to proceed."
    )
    PARAMS_MODEL = ModifyPendingOrderAddressParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "order_id": {"type": "string", "description": "The order id, such as '#W0000000'."},
            "address1": {"type": "string", "description": "The first line of the address, such as '123 Main St'."},
            "address2": {"type": "string", "description": "The second line of the address, such as 'Apt 1' or ''."},
            "city": {"type": "string", "description": "The city, such as 'San Francisco'."},
            "state": {"type": "string", "description": "The state, such as 'CA'."},
            "country": {"type": "string", "description": "The country, such as 'USA'."},
            "zip": {"type": "string", "description": "The zip code, such as '12345'."},
        }

    @property
    def required_properties(self) -> List[str]:
        return ["order_id", "address1", "address2", "city", "state", "country", "zip"]

    def _do_work(self, p: ModifyPendingOrderAddressParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        order = _get_order_dict(db, p.order_id)
        if order is None:
            return {"status": "error", "error_type": "not_found", "message": f"Order {p.order_id} not found"}
        if not _is_pending_order(order):
            return {"status": "error", "error_type": "value_error", "message": "Non-pending order cannot be modified"}

        order["address"] = {
            "address1": p.address1,
            "address2": p.address2,
            "city": p.city,
            "state": p.state,
            "country": p.country,
            "zip": p.zip,
        }
        self._record_action(
            {
                "action_type": "modify_pending_order_address",
                "name": "modify_pending_order_address",
                "arguments": p.model_dump(),
                "result": {"order_id": p.order_id},
            }
        )
        return order


@register_schema_tool_for_eval(domain="tau2_retail")
class ModifyPendingOrderItemsTool(_Tau2WriteTool):
    DESCRIPTION = (
        "Modify items in a pending order to new items of the same product type. For a "
        "pending order, this function can only be called once. The agent needs to explain "
        "the exchange detail and ask for explicit user confirmation (yes/no) to proceed."
    )
    PARAMS_MODEL = ModifyPendingOrderItemsParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "order_id": {"type": "string", "description": "The order id, such as '#W0000000'."},
            "item_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The item ids to be modified. There could be duplicate items in the list.",
            },
            "new_item_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "The item ids to be modified for. Each new item id should match the item id in the same "
                    "position and be of the same product."
                ),
            },
            "payment_method_id": {
                "type": "string",
                "description": "The payment method id, such as 'gift_card_0000000' or 'credit_card_0000000'.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["order_id", "item_ids", "new_item_ids", "payment_method_id"]

    def _do_work(self, p: ModifyPendingOrderItemsParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        order = _get_order_dict(db, p.order_id)
        if order is None:
            return {"status": "error", "error_type": "not_found", "message": f"Order {p.order_id} not found"}
        if order.get("status") != "pending":
            return {"status": "error", "error_type": "value_error", "message": "Non-pending order cannot be modified"}

        # Validate counts
        all_item_ids = [item.get("item_id") for item in order.get("items") or []]
        for item_id in p.item_ids:
            if p.item_ids.count(item_id) > all_item_ids.count(item_id):
                return {"status": "error", "error_type": "value_error", "message": f"{item_id} not found"}
        if len(p.item_ids) != len(p.new_item_ids):
            return {
                "status": "error",
                "error_type": "value_error",
                "message": "The number of items to be exchanged should match",
            }

        # Validate new items + compute diff
        diff_price = 0.0
        # We need the final variant for each new_item_id to apply price/options below.
        variants_to_apply: List[dict] = []
        for item_id, new_item_id in zip(p.item_ids, p.new_item_ids):
            if item_id == new_item_id:
                return {
                    "status": "error",
                    "error_type": "value_error",
                    "message": "The new item id should be different from the old item id",
                }
            order_item = next((item for item in order.get("items") or [] if item.get("item_id") == item_id), None)
            if order_item is None:
                return {"status": "error", "error_type": "value_error", "message": f"Item {item_id} not found"}
            variant = _get_variant_dict(db, order_item.get("product_id"), new_item_id)
            if variant is None or not variant.get("available", False):
                return {
                    "status": "error",
                    "error_type": "value_error",
                    "message": f"New item {new_item_id} not found or available",
                }
            variants_to_apply.append(variant)
            diff_price += float(variant.get("price", 0)) - float(order_item.get("price", 0))
        diff_price = _round2(diff_price)

        # Validate payment method
        user = _get_user_dict(db, order.get("user_id"))
        if user is None:
            return {"status": "error", "error_type": "not_found", "message": "User not found"}
        payment_method = _get_payment_method_dict(user, p.payment_method_id)
        if payment_method is None:
            return {"status": "error", "error_type": "not_found", "message": "Payment method not found"}
        if payment_method.get("source") == "gift_card" and (payment_method.get("balance") or 0) < diff_price:
            return {
                "status": "error",
                "error_type": "value_error",
                "message": "Insufficient gift card balance to pay for the new item",
            }

        # Apply payment/refund + gift-card balance update
        order.setdefault("payment_history", []).append(
            {
                "transaction_type": "payment" if diff_price > 0 else "refund",
                "amount": abs(diff_price),
                "payment_method_id": p.payment_method_id,
            }
        )
        if payment_method.get("source") == "gift_card":
            payment_method["balance"] = _round2((payment_method.get("balance") or 0) - diff_price)

        # Apply item changes
        for (item_id, new_item_id), variant in zip(zip(p.item_ids, p.new_item_ids), variants_to_apply):
            for item in order.get("items") or []:
                if item.get("item_id") == item_id:
                    item["item_id"] = new_item_id
                    item["price"] = variant.get("price")
                    item["options"] = variant.get("options")
                    break
        order["status"] = "pending (item modified)"

        self._record_action(
            {
                "action_type": "modify_pending_order_items",
                "name": "modify_pending_order_items",
                "arguments": p.model_dump(),
                "result": {"order_id": p.order_id, "price_difference": diff_price},
            }
        )
        return order


@register_schema_tool_for_eval(domain="tau2_retail")
class ModifyPendingOrderPaymentTool(_Tau2WriteTool):
    DESCRIPTION = (
        "Modify the payment method of a pending order. The agent needs to explain the "
        "modification detail and ask for explicit user confirmation (yes/no) to proceed."
    )
    PARAMS_MODEL = ModifyPendingOrderPaymentParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "order_id": {"type": "string", "description": "The order id, such as '#W0000000'."},
            "payment_method_id": {
                "type": "string",
                "description": "The payment method id, such as 'gift_card_0000000' or 'credit_card_0000000'.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["order_id", "payment_method_id"]

    def _do_work(self, p: ModifyPendingOrderPaymentParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        order = _get_order_dict(db, p.order_id)
        if order is None:
            return {"status": "error", "error_type": "not_found", "message": f"Order {p.order_id} not found"}
        if not _is_pending_order(order):
            return {"status": "error", "error_type": "value_error", "message": "Non-pending order cannot be modified"}

        user = _get_user_dict(db, order.get("user_id"))
        if user is None:
            return {"status": "error", "error_type": "not_found", "message": "User not found"}
        new_payment_method = _get_payment_method_dict(user, p.payment_method_id)
        if new_payment_method is None:
            return {"status": "error", "error_type": "not_found", "message": "Payment method not found"}

        history = order.get("payment_history") or []
        if len(history) != 1 or history[0].get("transaction_type") != "payment":
            return {
                "status": "error",
                "error_type": "value_error",
                "message": "There should be exactly one payment for a pending order",
            }

        old_payment_method_id = history[0].get("payment_method_id")
        if old_payment_method_id == p.payment_method_id:
            return {
                "status": "error",
                "error_type": "value_error",
                "message": "The new payment method should be different from the current one",
            }

        amount = history[0].get("amount", 0)

        if new_payment_method.get("source") == "gift_card" and (new_payment_method.get("balance") or 0) < amount:
            return {
                "status": "error",
                "error_type": "value_error",
                "message": "Insufficient gift card balance to pay for the order",
            }

        order["payment_history"] = history + [
            {"transaction_type": "payment", "amount": amount, "payment_method_id": p.payment_method_id},
            {"transaction_type": "refund", "amount": amount, "payment_method_id": old_payment_method_id},
        ]
        if new_payment_method.get("source") == "gift_card":
            new_payment_method["balance"] = _round2((new_payment_method.get("balance") or 0) - amount)
        old_payment_method = _get_payment_method_dict(user, old_payment_method_id)
        if old_payment_method is not None and old_payment_method.get("source") == "gift_card":
            old_payment_method["balance"] = _round2((old_payment_method.get("balance") or 0) + amount)

        self._record_action(
            {
                "action_type": "modify_pending_order_payment",
                "name": "modify_pending_order_payment",
                "arguments": p.model_dump(),
                "result": {"order_id": p.order_id},
            }
        )
        return order


@register_schema_tool_for_eval(domain="tau2_retail")
class ModifyUserAddressTool(_Tau2WriteTool):
    DESCRIPTION = (
        "Modify the default address of a user. The agent needs to explain the modification "
        "detail and ask for explicit user confirmation (yes/no) to proceed."
    )
    PARAMS_MODEL = ModifyUserAddressParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "user_id": {"type": "string", "description": "The user id, such as 'sara_doe_496'."},
            "address1": {"type": "string", "description": "The first line of the address, such as '123 Main St'."},
            "address2": {"type": "string", "description": "The second line of the address, such as 'Apt 1' or ''."},
            "city": {"type": "string", "description": "The city, such as 'San Francisco'."},
            "state": {"type": "string", "description": "The state, such as 'CA'."},
            "country": {"type": "string", "description": "The country, such as 'USA'."},
            "zip": {"type": "string", "description": "The zip code, such as '12345'."},
        }

    @property
    def required_properties(self) -> List[str]:
        return ["user_id", "address1", "address2", "city", "state", "country", "zip"]

    def _do_work(self, p: ModifyUserAddressParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        user = _get_user_dict(db, p.user_id)
        if user is None:
            return {"status": "error", "error_type": "not_found", "message": f"User {p.user_id} not found"}
        user["address"] = {
            "address1": p.address1,
            "address2": p.address2,
            "city": p.city,
            "state": p.state,
            "country": p.country,
            "zip": p.zip,
        }
        self._record_action(
            {
                "action_type": "modify_user_address",
                "name": "modify_user_address",
                "arguments": p.model_dump(),
                "result": {"user_id": p.user_id},
            }
        )
        return user


@register_schema_tool_for_eval(domain="tau2_retail")
class ReturnDeliveredOrderItemsTool(_Tau2WriteTool):
    DESCRIPTION = (
        "Return some items of a delivered order. The order status will be changed to "
        "'return requested'. The agent needs to explain the return detail and ask for "
        "explicit user confirmation (yes/no) to proceed. The user will receive follow-up "
        "email for how and where to return the item."
    )
    PARAMS_MODEL = ReturnDeliveredOrderItemsParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "order_id": {"type": "string", "description": "The order id, such as '#W0000000'."},
            "item_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The item ids to be returned. There could be duplicate items in the list.",
            },
            "payment_method_id": {
                "type": "string",
                "description": "The payment method id, such as 'gift_card_0000000' or 'credit_card_0000000'.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["order_id", "item_ids", "payment_method_id"]

    def _do_work(self, p: ReturnDeliveredOrderItemsParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        order = _get_order_dict(db, p.order_id)
        if order is None:
            return {"status": "error", "error_type": "not_found", "message": f"Order {p.order_id} not found"}
        if order.get("status") != "delivered":
            return {
                "status": "error",
                "error_type": "value_error",
                "message": "Non-delivered order cannot be returned",
            }

        user = _get_user_dict(db, order.get("user_id"))
        if user is None:
            return {"status": "error", "error_type": "not_found", "message": "User not found"}
        payment_method = _get_payment_method_dict(user, p.payment_method_id)
        if payment_method is None:
            return {"status": "error", "error_type": "not_found", "message": "Payment method not found"}

        # Either original payment method OR a gift card (mirrors tau2)
        original_payment_method_id = ((order.get("payment_history") or [{}])[0]).get("payment_method_id")
        if payment_method.get("source") != "gift_card" and p.payment_method_id != original_payment_method_id:
            return {
                "status": "error",
                "error_type": "value_error",
                "message": "Payment method should be the original payment method",
            }

        # Validate counts
        all_item_ids = [item.get("item_id") for item in order.get("items") or []]
        for item_id in p.item_ids:
            if p.item_ids.count(item_id) > all_item_ids.count(item_id):
                return {"status": "error", "error_type": "value_error", "message": "Some item not found"}

        order["status"] = "return requested"
        order["return_items"] = sorted(p.item_ids)
        order["return_payment_method_id"] = p.payment_method_id

        self._record_action(
            {
                "action_type": "return_delivered_order_items",
                "name": "return_delivered_order_items",
                "arguments": p.model_dump(),
                "result": {"order_id": p.order_id, "status": "return requested"},
            }
        )
        return order


# =============================================================================
# Generic tools
# =============================================================================


@register_schema_tool_for_eval(domain="tau2_retail")
class CalculateTool(_Tau2ReadTool):
    DESCRIPTION = (
        "Calculate the result of a mathematical expression. The expression can contain "
        "numbers, operators (+, -, *, /), parentheses, and spaces."
    )
    PARAMS_MODEL = CalculateParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "expression": {
                "type": "string",
                "description": (
                    "The mathematical expression to calculate, such as '2 + 2'. The expression can contain "
                    "numbers, operators (+, -, *, /), parentheses, and spaces."
                ),
            }
        }

    @property
    def required_properties(self) -> List[str]:
        return ["expression"]

    def _do_work(self, p: CalculateParams) -> dict:
        if not all(char in "0123456789+-*/(). " for char in p.expression):
            return {"status": "error", "error_type": "value_error", "message": "Invalid characters in expression"}
        try:
            result = round(float(eval(p.expression, {"__builtins__": None}, {})), 2)
        except Exception as e:
            return {"status": "error", "error_type": "value_error", "message": f"Invalid expression: {e}"}
        return {"status": "success", "result": str(result)}


@register_schema_tool_for_eval(domain="tau2_retail")
class TransferToHumanAgentsTool(_Tau2WriteTool):
    DESCRIPTION = (
        "Transfer the user to a human agent, with a summary of the user's issue. Only "
        "transfer if the user explicitly asks for a human agent, or given the policy and "
        "the available tools, you cannot solve the user's issue."
    )
    PARAMS_MODEL = TransferToHumanAgentsParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {"summary": {"type": "string", "description": "A summary of the user's issue."}}

    @property
    def required_properties(self) -> List[str]:
        return ["summary"]

    def _do_work(self, p: TransferToHumanAgentsParams) -> dict:
        # No DB mutation, but record so action-list comparison can match by name.
        self._record_action(
            {
                "action_type": "transfer_to_human_agents",
                "name": "transfer_to_human_agents",
                "arguments": p.model_dump(),
                "result": {"message": "Transfer successful"},
            }
        )
        return {"status": "success", "message": "Transfer successful"}

    async def _execute(self, params: FunctionCallParams) -> None:
        await super()._execute(params)
        await self._send_exit_message()


# =============================================================================
# Convenience: ordered tool list + name → class map
# =============================================================================


TAU2_RETAIL_TOOL_CLASSES: List[Type[StandardSchemaTool]] = [
    # reads
    FindUserIdByNameZipTool,
    FindUserIdByEmailTool,
    GetOrderDetailsTool,
    GetProductDetailsTool,
    GetItemDetailsTool,
    GetUserDetailsTool,
    ListAllProductTypesTool,
    # writes
    CancelPendingOrderTool,
    ExchangeDeliveredOrderItemsTool,
    ModifyPendingOrderAddressTool,
    ModifyPendingOrderItemsTool,
    ModifyPendingOrderPaymentTool,
    ModifyUserAddressTool,
    ReturnDeliveredOrderItemsTool,
    # generics
    CalculateTool,
    TransferToHumanAgentsTool,
]


# Map from tau2 @is_tool method name → Tool class. Used by ``_build_tool_map``
# in ``Tau2BaseScenario._gold_replay``.
TAU2_RETAIL_TOOL_NAME_TO_CLASS: Dict[str, Type[StandardSchemaTool]] = {
    "find_user_id_by_name_zip": FindUserIdByNameZipTool,
    "find_user_id_by_email": FindUserIdByEmailTool,
    "get_order_details": GetOrderDetailsTool,
    "get_product_details": GetProductDetailsTool,
    "get_item_details": GetItemDetailsTool,
    "get_user_details": GetUserDetailsTool,
    "list_all_product_types": ListAllProductTypesTool,
    "cancel_pending_order": CancelPendingOrderTool,
    "exchange_delivered_order_items": ExchangeDeliveredOrderItemsTool,
    "modify_pending_order_address": ModifyPendingOrderAddressTool,
    "modify_pending_order_items": ModifyPendingOrderItemsTool,
    "modify_pending_order_payment": ModifyPendingOrderPaymentTool,
    "modify_user_address": ModifyUserAddressTool,
    "return_delivered_order_items": ReturnDeliveredOrderItemsTool,
    "calculate": CalculateTool,
    "transfer_to_human_agents": TransferToHumanAgentsTool,
}

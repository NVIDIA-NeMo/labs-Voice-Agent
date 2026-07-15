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
#
# Adapted from https://github.com/sierra-research/tau2-bench/tree/voice-user-sim-v1.0
#   src/tau2/domains/retail/data_model.py — data models ported as Pydantic v2
#       (RetailDB / User / Order / OrderItem / Product / Variant / payment subtypes).
#   src/tau2/domains/retail/tools.py     — per-tool argument schemas derived from
#       the @is_tool method signatures + Args: docstring blocks.

"""Pydantic parameter models for tau2_retail tool functions.

Same pattern as ``tau2_airline_params.py``: two classes of models, one mirroring
``data_model.py`` for DB round-tripping and one ``*Params`` per LLM-exposed
tool for strict (``extra="forbid"``) argument validation.
"""

from enum import StrEnum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError


# =============================================================================
# Domain enums / type aliases
# =============================================================================


class OrderStatus(StrEnum):
    processed = "processed"
    pending = "pending"
    pending_item_modified = "pending (item modified)"
    delivered = "delivered"
    cancelled = "cancelled"
    exchange_requested = "exchange requested"
    return_requested = "return requested"


class CancelReason(StrEnum):
    no_longer_needed = "no longer needed"
    ordered_by_mistake = "ordered by mistake"


OrderPaymentType = Literal["payment", "refund"]


# =============================================================================
# Common domain models
# =============================================================================


class Variant(BaseModel):
    """Represents a specific variant of a product with its options, availability and price."""

    item_id: str = Field(description="Unique identifier for the variant")
    options: Dict[str, str] = Field(
        description="Dictionary of option names to values (e.g. {'color': 'blue', 'size': 'large'})"
    )
    available: bool = Field(description="Whether this variant is currently in stock")
    price: float = Field(description="Price of this variant")


class Product(BaseModel):
    """Represents a product with its variants."""

    name: str = Field(description="Name of the product")
    product_id: str = Field(description="Unique identifier for the product")
    variants: Dict[str, Variant] = Field(description="Dictionary of variants indexed by variant ID")


class UserName(BaseModel):
    first_name: str = Field(description="User's first name")
    last_name: str = Field(description="User's last name")


class UserAddress(BaseModel):
    address1: str = Field(description="Primary address line")
    address2: str = Field(description="Secondary address line")
    city: str = Field(description="City name")
    country: str = Field(description="Country name")
    state: str = Field(description="State or province name")
    zip: str = Field(description="Postal code")


# -----------------------------------------------------------------------------
# Payment-method variants
# -----------------------------------------------------------------------------


class PaymentMethodBase(BaseModel):
    source: str = Field(description="Type of payment method")
    id: str = Field(description="Unique identifier for the payment method")


class CreditCard(PaymentMethodBase):
    source: Literal["credit_card"] = Field(description="Indicates this is a credit card payment method")
    brand: str = Field(description="Credit card brand (e.g., visa, mastercard)")
    last_four: str = Field(description="Last four digits of the credit card")


class Paypal(PaymentMethodBase):
    source: Literal["paypal"] = Field(description="Indicates this is a paypal payment method")


class GiftCard(PaymentMethodBase):
    source: Literal["gift_card"] = Field(description="Indicates this is a gift card payment method")
    balance: float = Field(description="Gift card value amount")
    id: str = Field(description="Unique identifier for the gift card")


PaymentMethod = Union[CreditCard, GiftCard, Paypal]


# -----------------------------------------------------------------------------
# User
# -----------------------------------------------------------------------------


class User(BaseModel):
    """A user with their personal info, payment methods and order history."""

    user_id: str = Field(description="Unique identifier for the user")
    name: UserName = Field(description="User's full name")
    address: UserAddress = Field(description="User's primary address")
    email: str = Field(description="User's email address")
    payment_methods: Dict[str, PaymentMethod] = Field(
        description="Dictionary of payment methods indexed by payment method ID"
    )
    orders: List[str] = Field(description="List of order IDs associated with this user")


# -----------------------------------------------------------------------------
# Order
# -----------------------------------------------------------------------------


class OrderFullfilment(BaseModel):
    """Represents the fulfillment details for items in an order."""

    tracking_id: List[str] = Field(description="List of tracking IDs for shipments")
    item_ids: List[str] = Field(description="List of item IDs included in this fulfillment")


class OrderItem(BaseModel):
    name: str = Field(description="Name of the product")
    product_id: str = Field(description="ID of the product")
    item_id: str = Field(description="ID of the specific variant")
    price: float = Field(description="Price of the item at time of purchase")
    options: Dict[str, str] = Field(description="Options selected for this item")


class OrderPayment(BaseModel):
    transaction_type: OrderPaymentType = Field(description="Type of transaction (payment or refund)")
    amount: float = Field(description="Amount of the transaction")
    payment_method_id: str = Field(description="ID of the payment method used")


class Order(BaseModel):
    """An order with its items, status, fulfillment and payment details."""

    order_id: str = Field(description="Unique identifier for the order")
    user_id: str = Field(description="Unique identifier for the user")
    address: UserAddress = Field(description="Address of the user")
    items: List[OrderItem] = Field(description="Items in the order")
    status: OrderStatus = Field(description="Status of the order")
    fulfillments: List[OrderFullfilment] = Field(description="Fulfillments of the order")
    payment_history: List[OrderPayment] = Field(description="Payments of the order")
    cancel_reason: Optional[CancelReason] = Field(default=None, description="Reason for cancelling the order")
    exchange_items: Optional[List[str]] = Field(default=None, description="Items to be exchanged")
    exchange_new_items: Optional[List[str]] = Field(default=None, description="Items exchanged for")
    exchange_payment_method_id: Optional[str] = Field(default=None, description="Payment method ID for the exchange")
    exchange_price_difference: Optional[float] = Field(default=None, description="Price difference for the exchange")
    return_items: Optional[List[str]] = Field(default=None, description="Items to be returned")
    return_payment_method_id: Optional[str] = Field(default=None, description="Payment method ID for the return")


# -----------------------------------------------------------------------------
# Top-level DB shape
# -----------------------------------------------------------------------------


class RetailDB(BaseModel):
    """Top-level shape of ``nemo_voice_agent/evaluation/data/tau2_retail/db.json``."""

    products: Dict[str, Product] = Field(description="Dictionary of all products indexed by product ID")
    users: Dict[str, User] = Field(description="Dictionary of all users indexed by user ID")
    orders: Dict[str, Order] = Field(description="Dictionary of all orders indexed by order ID")


# =============================================================================
# Tool argument schemas
# =============================================================================


class _ToolParamsBase(BaseModel):
    """Strict base: reject unknown args."""

    model_config = ConfigDict(extra="forbid")


# ---- reads ----


class FindUserIdByNameZipParams(_ToolParamsBase):
    first_name: str = Field(description="The first name of the customer, such as 'John'.")
    last_name: str = Field(description="The last name of the customer, such as 'Doe'.")
    zip: str = Field(description="The zip code of the customer, such as '12345'.")


class FindUserIdByEmailParams(_ToolParamsBase):
    email: str = Field(description="The email of the user, such as 'something@example.com'.")


class GetOrderDetailsParams(_ToolParamsBase):
    order_id: str = Field(
        description=(
            "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id."
        )
    )


class GetProductDetailsParams(_ToolParamsBase):
    product_id: str = Field(
        description=("The product id, such as '6086499569'. Be careful the product id is different from the item id.")
    )


class GetItemDetailsParams(_ToolParamsBase):
    item_id: str = Field(
        description=("The item id, such as '6086499569'. Be careful the item id is different from the product id.")
    )


class GetUserDetailsParams(_ToolParamsBase):
    user_id: str = Field(description="The user id, such as 'sara_doe_496'.")


class ListAllProductTypesParams(_ToolParamsBase):
    """No arguments — LLM still passes an empty dict ``{}``."""

    pass


# ---- writes ----


class CancelPendingOrderParams(_ToolParamsBase):
    order_id: str = Field(
        description=(
            "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id."
        )
    )
    reason: str = Field(
        description=("The reason for cancellation, which should be either 'no longer needed' or 'ordered by mistake'.")
    )


class ExchangeDeliveredOrderItemsParams(_ToolParamsBase):
    order_id: str = Field(description="The order id, such as '#W0000000'.")
    item_ids: List[str] = Field(
        description=(
            "The item ids to be exchanged, each such as '1008292230'. There could be duplicate items in the list."
        )
    )
    new_item_ids: List[str] = Field(
        description=(
            "The item ids to be exchanged for, each such as '1008292230'. There "
            "could be duplicate items in the list. Each new item id should match "
            "the item id in the same position and be of the same product."
        )
    )
    payment_method_id: str = Field(
        description=(
            "The payment method id to pay or receive refund for the item price "
            "difference, such as 'gift_card_0000000' or 'credit_card_0000000'."
        )
    )


class ModifyPendingOrderAddressParams(_ToolParamsBase):
    order_id: str = Field(description="The order id, such as '#W0000000'.")
    address1: str = Field(description="The first line of the address, such as '123 Main St'.")
    address2: str = Field(description="The second line of the address, such as 'Apt 1' or ''.")
    city: str = Field(description="The city, such as 'San Francisco'.")
    state: str = Field(description="The state, such as 'CA'.")
    country: str = Field(description="The country, such as 'USA'.")
    zip: str = Field(description="The zip code, such as '12345'.")


class ModifyPendingOrderItemsParams(_ToolParamsBase):
    order_id: str = Field(description="The order id, such as '#W0000000'.")
    item_ids: List[str] = Field(
        description=(
            "The item ids to be modified, each such as '1008292230'. There could be duplicate items in the list."
        )
    )
    new_item_ids: List[str] = Field(
        description=(
            "The item ids to be modified for, each such as '1008292230'. There "
            "could be duplicate items in the list. Each new item id should match "
            "the item id in the same position and be of the same product."
        )
    )
    payment_method_id: str = Field(
        description=(
            "The payment method id to pay or receive refund for the item price "
            "difference, such as 'gift_card_0000000' or 'credit_card_0000000'."
        )
    )


class ModifyPendingOrderPaymentParams(_ToolParamsBase):
    order_id: str = Field(description="The order id, such as '#W0000000'.")
    payment_method_id: str = Field(
        description=("The payment method id, such as 'gift_card_0000000' or 'credit_card_0000000'.")
    )


class ModifyUserAddressParams(_ToolParamsBase):
    user_id: str = Field(description="The user id, such as 'sara_doe_496'.")
    address1: str = Field(description="The first line of the address, such as '123 Main St'.")
    address2: str = Field(description="The second line of the address, such as 'Apt 1' or ''.")
    city: str = Field(description="The city, such as 'San Francisco'.")
    state: str = Field(description="The state, such as 'CA'.")
    country: str = Field(description="The country, such as 'USA'.")
    zip: str = Field(description="The zip code, such as '12345'.")


class ReturnDeliveredOrderItemsParams(_ToolParamsBase):
    order_id: str = Field(description="The order id, such as '#W0000000'.")
    item_ids: List[str] = Field(
        description=(
            "The item ids to be returned, each such as '1008292230'. There could be duplicate items in the list."
        )
    )
    payment_method_id: str = Field(
        description=(
            "The payment method id to pay or receive refund for the item price "
            "difference, such as 'gift_card_0000000' or 'credit_card_0000000'."
        )
    )


# ---- generic ----


class CalculateParams(_ToolParamsBase):
    expression: str = Field(
        description=(
            "The mathematical expression to calculate, such as '2 + 2'. The expression "
            "can contain numbers, operators (+, -, *, /), parentheses, and spaces."
        )
    )


class TransferToHumanAgentsParams(_ToolParamsBase):
    summary: str = Field(description="A summary of the user's issue.")


# =============================================================================
# Shared helpers
# =============================================================================


def validation_error_response(error: ValidationError) -> Dict[str, Any]:
    """Translate a pydantic ValidationError into a standard tool-error dict.

    Mirrors ``tau2_airline_params.validation_error_response`` so the runner's
    downstream handling is uniform across domains.
    """
    return {
        "status": "error",
        "error_type": "validation_error",
        "message": "Tool arguments failed validation.",
        "details": [
            {
                "field": ".".join(str(p) for p in e.get("loc", ())),
                "error": e.get("msg", ""),
                "type": e.get("type", ""),
            }
            for e in error.errors()
        ],
    }

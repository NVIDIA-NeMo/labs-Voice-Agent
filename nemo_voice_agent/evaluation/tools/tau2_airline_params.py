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
#   src/tau2/domains/airline/data_model.py — data models ported as Pydantic v2
#       (FlightDB / User / Reservation / Payment / Flight / Passenger / ...).
#   src/tau2/domains/airline/tools.py     — per-tool argument schemas derived from
#       the @is_tool method signatures + Args: docstring blocks.

"""Pydantic parameter models and enums for tau2_airline tool functions.

This file holds **two** classes of models, both consumed by
``nemo_voice_agent.evaluation.tools.tau2_airline_tools``:

1. **Domain data models** — ``FlightDB`` / ``User`` / ``Reservation`` / ``Flight`` /
   ``Passenger`` / payment-method variants / etc. Mirror tau2's ``data_model.py``
   1:1 so dicts loaded from ``nemo_voice_agent/evaluation/data/tau2_airline/db.json`` round-trip
   through ``Model.model_validate(...)`` cleanly. Used by tools that need to
   manipulate strongly-typed records (e.g., creating a new Reservation).

2. **Tool argument schemas** — one ``*Params`` model per LLM-exposed tool. Each
   tool's body calls ``Model.model_validate(params)`` at the top and catches
   ``ValidationError`` to produce a standard ``{"status": "error", ...}`` response
   for bad LLM-supplied inputs. Field descriptions come verbatim from tau2's
   tool docstring ``Args:`` blocks so the LLM sees identical instructions.

Convention:

- Enums use ``StrEnum`` so values JSON-serialize as plain strings and compare
  equal to their literal counterparts.
- All param models use ``extra="forbid"`` — required by plan §5.1 step 3 to
  match tau2's strict validation semantics.
- Data models do NOT use ``extra="forbid"`` because ``db.json`` may carry
  upstream-added fields we don't model yet; dropping them silently is safer
  than crashing on load.
"""

from enum import StrEnum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

# =============================================================================
# Domain enums / type aliases (mirror data_model.py top-of-file)
# =============================================================================


class FlightType(StrEnum):
    round_trip = "round_trip"
    one_way = "one_way"


class CabinClass(StrEnum):
    business = "business"
    economy = "economy"
    basic_economy = "basic_economy"


class Insurance(StrEnum):
    yes = "yes"
    no = "no"


class MembershipLevel(StrEnum):
    gold = "gold"
    silver = "silver"
    regular = "regular"


# =============================================================================
# Common domain models
# =============================================================================


class AirportCode(BaseModel):
    iata: str = Field(description="IATA code")
    city: str = Field(description="City name")


AirportInfo = Annotated[List[AirportCode], Field(description="Airport information")]


class Name(BaseModel):
    first_name: str = Field(description="The person's first name")
    last_name: str = Field(description="The person's last name")


class Address(BaseModel):
    address1: str = Field(description="Primary address line")
    address2: Optional[str] = Field(None, description="Secondary address line (optional)")
    city: str = Field(description="City name")
    country: str = Field(description="Country name")
    state: str = Field(description="State or province name")
    zip: str = Field(description="Postal code")


class Passenger(BaseModel):
    first_name: str = Field(description="Passenger's first name")
    last_name: str = Field(description="Passenger's last name")
    dob: str = Field(description="Date of birth in YYYY-MM-DD format")


# -----------------------------------------------------------------------------
# Payment / payment-method models
# -----------------------------------------------------------------------------


class Payment(BaseModel):
    """A payment record (line item) attached to a reservation."""

    payment_id: str = Field(description="Unique identifier for the payment")
    amount: int = Field(description="Payment amount in dollars")


class PaymentMethodBase(BaseModel):
    source: str = Field(description="Type of payment method")
    id: str = Field(description="Unique identifier for the payment method")


class CreditCard(PaymentMethodBase):
    source: Literal["credit_card"] = Field(description="Indicates this is a credit card payment method")
    brand: str = Field(description="Credit card brand (e.g., visa, mastercard)")
    last_four: str = Field(description="Last four digits of the credit card")


class GiftCard(PaymentMethodBase):
    source: Literal["gift_card"] = Field(description="Indicates this is a gift card payment method")
    amount: float = Field(description="Gift card value amount")
    id: str = Field(description="Unique identifier for the gift card")


class Certificate(PaymentMethodBase):
    source: Literal["certificate"] = Field(description="Indicates this is a certificate payment method")
    amount: float = Field(description="Certificate value amount")


PaymentMethod = Union[CreditCard, GiftCard, Certificate]


# -----------------------------------------------------------------------------
# Flight / flight-date-status models
# -----------------------------------------------------------------------------


SeatPrices = Annotated[Dict[CabinClass, int], Field(description="Prices for different cabin classes")]
AvailableSeats = Annotated[
    Dict[CabinClass, int],
    Field(description="Available seats for different cabin classes"),
]


class FlightDateStatusAvailable(BaseModel):
    status: Literal["available"] = Field(description="Indicates flight is available for booking")
    available_seats: AvailableSeats = Field(description="Available seats by class")
    prices: SeatPrices = Field(description="Current prices by class")


class FlightDataStatusOnTime(BaseModel):
    status: Literal["on time"] = Field(description="Indicates flight is on time")
    estimated_departure_time_est: str = Field(
        description="Estimated departure time in EST in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-15T06:04:00"
    )
    estimated_arrival_time_est: str = Field(
        description="Estimated arrival time in EST in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-15T07:30:00"
    )


class FlightDataStatusFlying(BaseModel):
    status: Literal["flying"] = Field(description="Indicates flight is in flight")
    actual_departure_time_est: str = Field(
        description="Actual departure time in EST in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-15T06:04:00"
    )
    estimated_arrival_time_est: str = Field(
        description="Estimated arrival time in EST in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-15T07:30:00"
    )


class FlightDateStatusLanded(BaseModel):
    status: Literal["landed"] = Field(description="Indicates flight has landed")
    actual_departure_time_est: str = Field(
        description="Actual departure time in EST in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-15T06:04:00"
    )
    actual_arrival_time_est: str = Field(
        description="Actual arrival time in EST in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-15T07:30:00"
    )


class FlightDateStatusCancelled(BaseModel):
    status: Literal["cancelled"] = Field(description="Indicates flight was cancelled")


class FlightDateStatusDelayed(BaseModel):
    status: Literal["delayed"] = Field(description="Indicates flight was delayed")
    estimated_departure_time_est: str = Field(
        description="Estimated departure time in EST in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-15T06:04:00"
    )
    estimated_arrival_time_est: str = Field(
        description="Estimated arrival time in EST in the format YYYY-MM-DDTHH:MM:SS, e.g 2024-05-15T07:30:00"
    )


FlightDateStatus = Union[
    FlightDateStatusAvailable,
    FlightDateStatusLanded,
    FlightDateStatusCancelled,
    FlightDateStatusDelayed,
    FlightDataStatusFlying,
    FlightDataStatusOnTime,
]


class FlightBase(BaseModel):
    flight_number: str = Field(description="Unique flight identifier")
    origin: str = Field(description="IATA code for origin airport")
    destination: str = Field(description="IATA code for destination airport")


class Flight(FlightBase):
    scheduled_departure_time_est: str = Field(
        description="Scheduled departure time in EST in the format HH:MM:SS, e.g 06:00:00"
    )
    scheduled_arrival_time_est: str = Field(
        description="Scheduled arrival time in EST in the format HH:MM:SS, e.g 07:00:00"
    )
    dates: Dict[str, FlightDateStatus] = Field(description="Flight status by date (YYYY-MM-DD)")


class DirectFlight(FlightBase):
    status: Literal["available"] = Field(description="Indicates flight is available for booking")
    scheduled_departure_time_est: str = Field(
        description="Scheduled departure time in EST in the format HH:MM:SS, e.g 06:00:00"
    )
    scheduled_arrival_time_est: str = Field(
        description="Scheduled arrival time in EST in the format HH:MM:SS, e.g 07:00:00"
    )
    date: Optional[str] = Field(None, description="Flight date in YYYY-MM-DD format")
    available_seats: AvailableSeats = Field(description="Available seats by class")
    prices: SeatPrices = Field(description="Current prices by class")


class ReservationFlight(FlightBase):
    date: str = Field(description="Flight date in YYYY-MM-DD format")
    price: int = Field(description="Flight price in dollars.")


class FlightInfo(BaseModel):
    """One element of the ``flights`` argument to book_reservation /
    update_reservation_flights — just a flight-number + date pair."""

    flight_number: str = Field(description="Flight number, such as 'HAT001'.")
    date: str = Field(description="The date for the flight in the format 'YYYY-MM-DD', such as '2024-05-01'.")


# -----------------------------------------------------------------------------
# User + Reservation
# -----------------------------------------------------------------------------


class User(BaseModel):
    user_id: str = Field(description="Unique identifier for the user")
    name: Name = Field(description="User's full name")
    address: Address = Field(description="User's address information")
    email: str = Field(description="User's email address")
    dob: str = Field(description="User's date of birth in the format YYYY-MM-DD, e.g 1990-04-05")
    payment_methods: Dict[str, PaymentMethod] = Field(description="User's saved payment methods")
    saved_passengers: List[Passenger] = Field(description="User's saved passenger information")
    membership: MembershipLevel = Field(description="User's membership level")
    reservations: List[str] = Field(description="List of user's reservation IDs")


class Reservation(BaseModel):
    reservation_id: str = Field(description="Unique identifier for the reservation")
    user_id: str = Field(description="ID of the user who made the reservation")
    origin: str = Field(description="IATA code for trip origin")
    destination: str = Field(description="IATA code for trip destination")
    flight_type: FlightType = Field(description="Type of trip")
    cabin: CabinClass = Field(description="Selected cabin class")
    flights: List[ReservationFlight] = Field(description="List of flights in the reservation")
    passengers: List[Passenger] = Field(description="List of passengers on the reservation")
    payment_history: List[Payment] = Field(description="History of payments for this reservation")
    created_at: str = Field(description="Timestamp when reservation was created in the format YYYY-MM-DDTHH:MM:SS")
    total_baggages: int = Field(description="Total number of bags in reservation")
    nonfree_baggages: int = Field(description="Number of paid bags in reservation")
    insurance: Insurance = Field(description="Whether travel insurance was purchased")
    status: Optional[Literal["cancelled"]] = Field(None, description="Status of the reservation")


class FlightDB(BaseModel):
    """Top-level shape of ``nemo_voice_agent/evaluation/data/tau2_airline/db.json``."""

    flights: Dict[str, Flight] = Field(description="Dictionary of all flights indexed by flight number")
    users: Dict[str, User] = Field(description="Dictionary of all users indexed by user ID")
    reservations: Dict[str, Reservation] = Field(
        description="Dictionary of all reservations indexed by reservation ID"
    )


# =============================================================================
# Tool argument schemas
# Each *Params model matches one LLM-exposed tool. ``extra="forbid"`` rejects
# unknown args so a hallucinated key fails fast instead of silently being dropped.
# Field descriptions are copied verbatim from tau2's tool docstrings so the
# LLM sees the same prompt surface.
# =============================================================================


class _ToolParamsBase(BaseModel):
    """Strict base: reject unknown args, freeze types after validation."""

    model_config = ConfigDict(extra="forbid")


# ---- reads ----


class GetUserDetailsParams(_ToolParamsBase):
    user_id: str = Field(description="The user ID, such as 'sara_doe_496'.")


class GetReservationDetailsParams(_ToolParamsBase):
    reservation_id: str = Field(description="The reservation ID, such as '8JX2WO'.")


class ListAllAirportsParams(_ToolParamsBase):
    """No arguments — LLM still passes an empty dict ``{}``."""

    pass


class SearchDirectFlightParams(_ToolParamsBase):
    origin: str = Field(description="The origin city airport in three letters, such as 'JFK'.")
    destination: str = Field(description="The destination city airport in three letters, such as 'LAX'.")
    date: str = Field(description="The date of the flight in the format 'YYYY-MM-DD', such as '2024-01-01'.")


class SearchOnestopFlightParams(_ToolParamsBase):
    origin: str = Field(description="The origin city airport in three letters, such as 'JFK'.")
    destination: str = Field(description="The destination city airport in three letters, such as 'LAX'.")
    date: str = Field(description="The date of the flight in the format 'YYYY-MM-DD', such as '2024-05-01'.")


class GetFlightStatusParams(_ToolParamsBase):
    flight_number: str = Field(description="The flight number.")
    date: str = Field(description="The date of the flight.")


# ---- writes ----


class BookReservationParams(_ToolParamsBase):
    user_id: str = Field(description="The ID of the user to book the reservation such as 'sara_doe_496'.")
    origin: str = Field(description="The IATA code for the origin city such as 'SFO'.")
    destination: str = Field(description="The IATA code for the destination city such as 'JFK'.")
    flight_type: FlightType = Field(description="The type of flight such as 'one_way' or 'round_trip'.")
    cabin: CabinClass = Field(description="The cabin class such as 'basic_economy', 'economy', or 'business'.")
    flights: List[FlightInfo] = Field(description="An array of objects containing details about each piece of flight.")
    passengers: List[Passenger] = Field(description="An array of objects containing details about each passenger.")
    payment_methods: List[Payment] = Field(
        description="An array of objects containing details about each payment method."
    )
    total_baggages: int = Field(description="The total number of baggage items to book the reservation.")
    nonfree_baggages: int = Field(description="The number of non-free baggage items to book the reservation.")
    insurance: Insurance = Field(description="Whether the reservation has insurance.")


class CancelReservationParams(_ToolParamsBase):
    reservation_id: str = Field(description="The reservation ID, such as 'ZFA04Y'.")


class SendCertificateParams(_ToolParamsBase):
    user_id: str = Field(description="The ID of the user to book the reservation, such as 'sara_doe_496'.")
    amount: int = Field(description="The amount of the certificate to send.")


class UpdateReservationBaggagesParams(_ToolParamsBase):
    reservation_id: str = Field(description="The reservation ID, such as 'ZFA04Y'")
    total_baggages: int = Field(description="The updated total number of baggage items included in the reservation.")
    nonfree_baggages: int = Field(
        description="The updated number of non-free baggage items included in the reservation."
    )
    payment_id: str = Field(
        description=(
            "The payment id stored in user profile, such as 'credit_card_7815826', "
            "'gift_card_7815826', 'certificate_7815826'."
        )
    )


class UpdateReservationFlightsParams(_ToolParamsBase):
    reservation_id: str = Field(description="The reservation ID, such as 'ZFA04Y'.")
    cabin: CabinClass = Field(description="The cabin class of the reservation")
    flights: List[FlightInfo] = Field(
        description=(
            "An array of objects containing details about each piece of flight in the ENTIRE new reservation. "
            "Even if the a flight segment is not changed, it should still be included in the array."
        )
    )
    payment_id: str = Field(
        description=(
            "The payment id stored in user profile, such as 'credit_card_7815826', "
            "'gift_card_7815826', 'certificate_7815826'."
        )
    )


class UpdateReservationPassengersParams(_ToolParamsBase):
    reservation_id: str = Field(description="The reservation ID, such as 'ZFA04Y'.")
    passengers: List[Passenger] = Field(description="An array of objects containing details about each passenger.")


# ---- generics (no DB I/O) ----


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
# Validation-error helper (mirrors eva_airline_params.validation_error_response)
# =============================================================================


def validation_error_response(error: ValidationError) -> Dict[str, Any]:
    """Translate a pydantic ``ValidationError`` into a standard tool-error dict.

    Tools should call this in their ``except ValidationError`` block so the LLM
    sees a structured error rather than a raw Python traceback. Format mirrors
    eva_airline tools' error shape so the runner's downstream handling doesn't
    need a separate code path per domain.
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

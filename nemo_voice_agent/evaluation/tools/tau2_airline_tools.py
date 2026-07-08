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
#   src/tau2/domains/airline/tools.py — all 14 @is_tool methods of AirlineTools
#       ported as StandardSchemaTool / WriteScenarioTool subclasses. Helpers like
#       ``_get_new_reservation_id``, ``_payment_for_update`` and the search
#       loops are reproduced here as module-level functions operating on the
#       dict-form DB (rather than the Pydantic FlightDB) to avoid a 7 MB
#       validate-and-dump round-trip per tool call.

"""Tau2-airline tool ports.

Each LLM-exposed tool from tau2's ``AirlineTools`` class becomes a
``StandardSchemaTool`` or ``WriteScenarioTool`` subclass. The DB lives in
``self.state["db"]`` as a raw dict (loaded from ``nemo_voice_agent/evaluation/data/tau2_airline/db.json``);
tools mutate it in place. Write-tools call ``self._record_action({...})`` with an
``action_type`` matching one of ``TAU2_AIRLINE_ACTION_TYPES``.

Two parallel call paths exist, both routing through ``_do_work(p)``:

1. **Live agent** — pipecat invokes ``await tool(params)``, which routes through
   ``StandardSchemaTool.__call__`` → ``_execute(params)`` → ``_do_work(p)`` →
   ``params.result_callback(result)``.
2. **Gold replay** — ``Tau2BaseScenario._gold_replay`` invokes
   ``tool.invoke(**arguments)`` → ``_do_work(p)`` synchronously, returning the
   result directly.

The ``_Tau2InvokeMixin`` handles both routings so each individual tool only
implements ``_do_work``.
"""

# pylint: disable=line-too-long
# flake8: noqa: E501

import copy
from typing import Any, ClassVar, Dict, List, Optional, Type

from pipecat.services.llm_service import FunctionCallParams
from pydantic import BaseModel, ValidationError

from nemo_voice_agent.evaluation.tools import normalize_tool_result, register_schema_tool_for_eval
from nemo_voice_agent.evaluation.tools._write_tool_base import WriteScenarioTool
from nemo_voice_agent.evaluation.tools.tau2_airline_params import (
    BookReservationParams,
    CalculateParams,
    CancelReservationParams,
    GetFlightStatusParams,
    GetReservationDetailsParams,
    GetUserDetailsParams,
    ListAllAirportsParams,
    Reservation,
    ReservationFlight,
    SearchDirectFlightParams,
    SearchOnestopFlightParams,
    SendCertificateParams,
    TransferToHumanAgentsParams,
    UpdateReservationBaggagesParams,
    UpdateReservationFlightsParams,
    UpdateReservationPassengersParams,
    validation_error_response,
)
from nemo_voice_agent.utils.tool_calling import StandardSchemaTool


# =============================================================================
# Action-type vocabulary
# Recorded by every _Tau2WriteTool subclass via ``_record_action({"action_type": ...})``.
# Names match tau2's @is_tool method names 1:1, so the gold-replay's records
# line up with what we'd later wire into action-list comparison.
# =============================================================================


TAU2_AIRLINE_ACTION_TYPES: List[str] = [
    "book_reservation",
    "cancel_reservation",
    "send_certificate",
    "update_reservation_baggages",
    "update_reservation_flights",
    "update_reservation_passengers",
    "transfer_to_human_agents",
]


# =============================================================================
# DB-dict helpers — operate on the JSON-loaded ``state["db"]`` dict directly
# rather than via the Pydantic FlightDB (7 MB validate-and-dump per tool call
# would be wasteful). Mirror tau2's ``_get_user`` / ``_get_reservation`` / etc.
# =============================================================================


def _db_not_initialized() -> dict:
    return {
        "status": "error",
        "error_type": "db_not_initialized",
        "message": "Scenario database not loaded. This indicates a fixture-loading bug.",
    }


def _get_user_dict(db: dict, user_id: str) -> Optional[dict]:
    # tau2 user_ids are lowercase in db.json ("daiki_muller_1116"). Voice ASR
    # can emit mixed/upper case ("Daiki_Muller_1116"); normalize so the lookup
    # succeeds without wasting tool-call retries.
    return db.get("users", {}).get(user_id.lower() if user_id else user_id)


def _get_reservation_dict(db: dict, reservation_id: str) -> Optional[dict]:
    # tau2 reservation_ids are uppercase in db.json ("XEHM4B"). Voice ASR
    # often emits lowercase ("xehm4b") after letter-by-letter spelling;
    # normalize to avoid spurious "not found" responses.
    return db.get("reservations", {}).get(reservation_id.upper() if reservation_id else reservation_id)


def _get_flight_dict(db: dict, flight_number: str) -> Optional[dict]:
    # Flight numbers are uppercase in db.json ("HAT001"). Same ASR-casing
    # concern as reservation_id.
    return db.get("flights", {}).get(flight_number.upper() if flight_number else flight_number)


def _get_flight_instance_dict(db: dict, flight_number: str, date: str) -> Optional[dict]:
    flight = _get_flight_dict(db, flight_number)
    if flight is None:
        return None
    return (flight.get("dates") or {}).get(date)


def _get_new_reservation_id(db: dict) -> Optional[str]:
    """Mirror tau2: try HATHAT/HATHAU/HATHAV in order; first unused wins."""
    for rid in ("HATHAT", "HATHAU", "HATHAV"):
        if rid not in db.get("reservations", {}):
            return rid
    return None


# Tau2 hard-codes three payment ids per task.
_NEW_PAYMENT_IDS = (3221322, 3221323, 3221324)


def _current_datetime() -> str:
    """Mirror tau2's ``_get_datetime``."""
    return "2024-05-15T15:00:00"


def _payment_for_update(user: dict, payment_id: str, total_price: int) -> dict:
    """Validate a payment method and (if applicable) deduct from a gift card.

    Returns a dict with either ``{"status": "ok", "payment": {...} | None}`` or
    a structured error. Mirrors tau2's ``_payment_for_update`` semantics:
      - certificate cannot be used for updates (always error)
      - gift_card requires sufficient balance
      - credit_card has no balance check
    Mutates the user dict in place when deducting from a gift card.
    """
    payment_methods = user.get("payment_methods") or {}
    if payment_id not in payment_methods:
        return {"status": "error", "error_type": "value_error", "message": "Payment method not found"}
    method = payment_methods[payment_id]
    source = method.get("source")
    if source == "certificate":
        return {
            "status": "error",
            "error_type": "value_error",
            "message": "Certificate cannot be used to update reservation",
        }
    if source == "gift_card" and method.get("amount", 0) < total_price:
        return {"status": "error", "error_type": "value_error", "message": "Gift card balance is not enough"}
    if source == "gift_card":
        method["amount"] -= total_price

    payment = None
    if total_price != 0:
        payment = {"payment_id": payment_id, "amount": total_price}
    return {"status": "ok", "payment": payment}


def _search_direct_flights(
    db: dict,
    date: str,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    leave_after: Optional[str] = None,
) -> List[dict]:
    """Mirror tau2's ``_search_direct_flight`` helper.

    Returns DirectFlight-shaped dicts: ``{flight_number, origin, destination,
    status="available", scheduled_departure_time_est, scheduled_arrival_time_est,
    available_seats, prices}``. Date is **not** stamped here — callers (e.g.
    ``search_onestop_flight``) attach it.
    """
    results: List[dict] = []
    for flight in (db.get("flights") or {}).values():
        flight_origin = flight.get("origin")
        flight_dest = flight.get("destination")
        dates = flight.get("dates") or {}
        if origin is not None and flight_origin != origin:
            continue
        if destination is not None and flight_dest != destination:
            continue
        if date not in dates:
            continue
        flight_date = dates[date]
        if flight_date.get("status") != "available":
            continue
        if leave_after is not None and flight.get("scheduled_departure_time_est", "") < leave_after:
            continue
        results.append(
            {
                "flight_number": flight.get("flight_number"),
                "origin": flight_origin,
                "destination": flight_dest,
                "status": "available",
                "scheduled_departure_time_est": flight.get("scheduled_departure_time_est"),
                "scheduled_arrival_time_est": flight.get("scheduled_arrival_time_est"),
                "available_seats": flight_date.get("available_seats"),
                "prices": flight_date.get("prices"),
            }
        )
    return results


# Tau2 ships ``list_all_airports`` as a fixed return — keep the list verbatim.
_ALL_AIRPORTS: List[Dict[str, str]] = [
    {"iata": "SFO", "city": "San Francisco"},
    {"iata": "JFK", "city": "New York"},
    {"iata": "LAX", "city": "Los Angeles"},
    {"iata": "ORD", "city": "Chicago"},
    {"iata": "DFW", "city": "Dallas"},
    {"iata": "DEN", "city": "Denver"},
    {"iata": "SEA", "city": "Seattle"},
    {"iata": "ATL", "city": "Atlanta"},
    {"iata": "MIA", "city": "Miami"},
    {"iata": "BOS", "city": "Boston"},
    {"iata": "PHX", "city": "Phoenix"},
    {"iata": "IAH", "city": "Houston"},
    {"iata": "LAS", "city": "Las Vegas"},
    {"iata": "MCO", "city": "Orlando"},
    {"iata": "EWR", "city": "Newark"},
    {"iata": "CLT", "city": "Charlotte"},
    {"iata": "MSP", "city": "Minneapolis"},
    {"iata": "DTW", "city": "Detroit"},
    {"iata": "PHL", "city": "Philadelphia"},
    {"iata": "LGA", "city": "LaGuardia"},
]


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
        """Async entry — invoked by pipecat on live LLM tool calls."""
        result = self.invoke(**(params.arguments or {}))
        # Guard against pipecat masking a falsy result (e.g. an empty match
        # list) as the literal "COMPLETED"; the LLM would read that as success.
        await params.result_callback(normalize_tool_result(result))

    def _do_work(self, p) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError(f"{type(self).__name__} must implement _do_work(p)")


class _Tau2ReadTool(_Tau2InvokeMixin, StandardSchemaTool):
    """Base for tau2 read-only tools — inherits ``StandardSchemaTool`` (no action recording)."""

    DESCRIPTION: ClassVar[str] = ""

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or self.DESCRIPTION)
        self.state = shared_state if shared_state is not None else {}


class _Tau2WriteTool(_Tau2InvokeMixin, WriteScenarioTool):
    """Base for tau2 write tools — inherits ``WriteScenarioTool`` (``_record_action`` + ``ACTION_TYPES``)."""

    ACTION_TYPES = TAU2_AIRLINE_ACTION_TYPES
    DESCRIPTION: ClassVar[str] = ""

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or self.DESCRIPTION)
        self.state = shared_state if shared_state is not None else {}


# =============================================================================
# Read tools
# =============================================================================


@register_schema_tool_for_eval(domain="tau2_airline")
class GetUserDetailsTool(_Tau2ReadTool):
    DESCRIPTION = "Get the details of a user, including their reservations."
    PARAMS_MODEL = GetUserDetailsParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {"user_id": {"type": "string", "description": "The user ID, such as 'sara_doe_496'."}}

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


@register_schema_tool_for_eval(domain="tau2_airline")
class GetReservationDetailsTool(_Tau2ReadTool):
    DESCRIPTION = "Get the details of a reservation."
    PARAMS_MODEL = GetReservationDetailsParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {"reservation_id": {"type": "string", "description": "The reservation ID, such as '8JX2WO'."}}

    @property
    def required_properties(self) -> List[str]:
        return ["reservation_id"]

    def _do_work(self, p: GetReservationDetailsParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        reservation = _get_reservation_dict(db, p.reservation_id)
        if reservation is None:
            return {
                "status": "error",
                "error_type": "not_found",
                "message": f"Reservation {p.reservation_id} not found",
            }
        return reservation


@register_schema_tool_for_eval(domain="tau2_airline")
class ListAllAirportsTool(_Tau2ReadTool):
    DESCRIPTION = "Returns a list of all available airports as IATA-code + city pairs."
    PARAMS_MODEL = ListAllAirportsParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def required_properties(self) -> List[str]:
        return []

    def _do_work(self, p: ListAllAirportsParams) -> dict:
        return {"status": "success", "airports": copy.deepcopy(_ALL_AIRPORTS)}


@register_schema_tool_for_eval(domain="tau2_airline")
class SearchDirectFlightTool(_Tau2ReadTool):
    DESCRIPTION = "Search for direct flights between two cities on a specific date."
    PARAMS_MODEL = SearchDirectFlightParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "origin": {"type": "string", "description": "The origin city airport in three letters, such as 'JFK'."},
            "destination": {
                "type": "string",
                "description": "The destination city airport in three letters, such as 'LAX'.",
            },
            "date": {
                "type": "string",
                "description": "The date of the flight in the format 'YYYY-MM-DD', such as '2024-01-01'.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["origin", "destination", "date"]

    def _do_work(self, p: SearchDirectFlightParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        results = _search_direct_flights(db, date=p.date, origin=p.origin, destination=p.destination)
        return {"status": "success", "flights": results}


@register_schema_tool_for_eval(domain="tau2_airline")
class SearchOnestopFlightTool(_Tau2ReadTool):
    DESCRIPTION = "Search for one-stop flights between two cities on a specific date."
    PARAMS_MODEL = SearchOnestopFlightParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "origin": {"type": "string", "description": "The origin city airport in three letters, such as 'JFK'."},
            "destination": {
                "type": "string",
                "description": "The destination city airport in three letters, such as 'LAX'.",
            },
            "date": {
                "type": "string",
                "description": "The date of the flight in the format 'YYYY-MM-DD', such as '2024-05-01'.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["origin", "destination", "date"]

    def _do_work(self, p: SearchOnestopFlightParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        results: List[List[dict]] = []
        for r1 in _search_direct_flights(db, date=p.date, origin=p.origin, destination=None):
            r1 = dict(r1)
            r1["date"] = p.date
            # Tau2 quirk: if arrival has a "+1" marker the second leg is the next day.
            # The May-2024 date assumption is verbatim from tau2.
            date2 = (
                f"2024-05-{int(p.date[-2:]) + 1}" if "+1" in (r1.get("scheduled_arrival_time_est") or "") else p.date
            )
            for r2 in _search_direct_flights(
                db,
                date=date2,
                origin=r1["destination"],
                destination=p.destination,
                leave_after=r1.get("scheduled_arrival_time_est"),
            ):
                r2 = dict(r2)
                r2["date"] = date2
                results.append([r1, r2])
        return {"status": "success", "onestop_flights": results}


@register_schema_tool_for_eval(domain="tau2_airline")
class GetFlightStatusTool(_Tau2ReadTool):
    DESCRIPTION = "Get the status of a flight on a given date."
    PARAMS_MODEL = GetFlightStatusParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "flight_number": {"type": "string", "description": "The flight number."},
            "date": {"type": "string", "description": "The date of the flight."},
        }

    @property
    def required_properties(self) -> List[str]:
        return ["flight_number", "date"]

    def _do_work(self, p: GetFlightStatusParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        instance = _get_flight_instance_dict(db, p.flight_number, p.date)
        if instance is None:
            return {
                "status": "error",
                "error_type": "not_found",
                "message": f"Flight {p.flight_number} not found on date {p.date}",
            }
        return {"status": "success", "flight_status": instance.get("status")}


# =============================================================================
# Write tools
# =============================================================================


@register_schema_tool_for_eval(domain="tau2_airline")
class BookReservationTool(_Tau2WriteTool):
    DESCRIPTION = (
        "Book a new flight reservation under a user account. Reserves seats on each leg, "
        "validates payment methods cover the total price, and creates a reservation record."
    )
    PARAMS_MODEL = BookReservationParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "user_id": {
                "type": "string",
                "description": "The ID of the user to book the reservation such as 'sara_doe_496'.",
            },
            "origin": {"type": "string", "description": "The IATA code for the origin city such as 'SFO'."},
            "destination": {"type": "string", "description": "The IATA code for the destination city such as 'JFK'."},
            "flight_type": {
                "type": "string",
                "enum": ["round_trip", "one_way"],
                "description": "The type of flight such as 'one_way' or 'round_trip'.",
            },
            "cabin": {
                "type": "string",
                "enum": ["business", "economy", "basic_economy"],
                "description": "The cabin class such as 'basic_economy', 'economy', or 'business'.",
            },
            "flights": {
                "type": "array",
                "description": "An array of objects containing details about each piece of flight.",
                "items": {
                    "type": "object",
                    "properties": {
                        "flight_number": {"type": "string", "description": "Flight number, such as 'HAT001'."},
                        "date": {
                            "type": "string",
                            "description": "Flight date in 'YYYY-MM-DD' format.",
                        },
                    },
                    "required": ["flight_number", "date"],
                },
            },
            "passengers": {
                "type": "array",
                "description": "An array of objects containing details about each passenger.",
                "items": {
                    "type": "object",
                    "properties": {
                        "first_name": {"type": "string"},
                        "last_name": {"type": "string"},
                        "dob": {"type": "string", "description": "Date of birth in YYYY-MM-DD format"},
                    },
                    "required": ["first_name", "last_name", "dob"],
                },
            },
            "payment_methods": {
                "type": "array",
                "description": "An array of objects containing details about each payment method.",
                "items": {
                    "type": "object",
                    "properties": {
                        "payment_id": {"type": "string"},
                        "amount": {"type": "integer"},
                    },
                    "required": ["payment_id", "amount"],
                },
            },
            "total_baggages": {
                "type": "integer",
                "description": "The total number of baggage items to book the reservation.",
            },
            "nonfree_baggages": {
                "type": "integer",
                "description": "The number of non-free baggage items to book the reservation.",
            },
            "insurance": {
                "type": "string",
                "enum": ["yes", "no"],
                "description": "Whether the reservation has insurance.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return [
            "user_id",
            "origin",
            "destination",
            "flight_type",
            "cabin",
            "flights",
            "passengers",
            "payment_methods",
            "total_baggages",
            "nonfree_baggages",
            "insurance",
        ]

    def _do_work(self, p: BookReservationParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        user = _get_user_dict(db, p.user_id)
        if user is None:
            return {"status": "error", "error_type": "not_found", "message": f"User {p.user_id} not found"}

        reservation_id = _get_new_reservation_id(db)
        if reservation_id is None:
            return {"status": "error", "error_type": "value_error", "message": "Too many reservations"}

        # Resolve flight availability + compute price
        total_price = 0
        reservation_flights: List[dict] = []
        flight_date_data_refs: List[dict] = []  # references for later seat deduction
        cabin_value = p.cabin.value if hasattr(p.cabin, "value") else p.cabin
        n_passengers = len(p.passengers)
        for flight_info in p.flights:
            flight = _get_flight_dict(db, flight_info.flight_number)
            if flight is None:
                return {
                    "status": "error",
                    "error_type": "not_found",
                    "message": f"Flight {flight_info.flight_number} not found",
                }
            flight_date_data = _get_flight_instance_dict(db, flight_info.flight_number, flight_info.date)
            if flight_date_data is None:
                return {
                    "status": "error",
                    "error_type": "not_found",
                    "message": f"Flight {flight_info.flight_number} not found on date {flight_info.date}",
                }
            if flight_date_data.get("status") != "available":
                return {
                    "status": "error",
                    "error_type": "value_error",
                    "message": f"Flight {flight_info.flight_number} not available on date {flight_info.date}",
                }
            available = (flight_date_data.get("available_seats") or {}).get(cabin_value, 0)
            if available < n_passengers:
                return {
                    "status": "error",
                    "error_type": "value_error",
                    "message": f"Not enough seats on flight {flight_info.flight_number}",
                }
            price = (flight_date_data.get("prices") or {}).get(cabin_value, 0)
            reservation_flights.append(
                ReservationFlight(
                    origin=flight.get("origin"),
                    destination=flight.get("destination"),
                    flight_number=flight_info.flight_number,
                    date=flight_info.date,
                    price=price,
                ).model_dump()
            )
            flight_date_data_refs.append(flight_date_data)
            total_price += price * n_passengers

        # Insurance + baggage fees (tau2: $30/passenger, $50/non-free bag)
        if p.insurance.value == "yes" if hasattr(p.insurance, "value") else p.insurance == "yes":
            total_price += 30 * n_passengers
        total_price += 50 * p.nonfree_baggages

        # Validate every payment method exists and has sufficient balance
        for payment in p.payment_methods:
            method = (user.get("payment_methods") or {}).get(payment.payment_id)
            if method is None:
                return {
                    "status": "error",
                    "error_type": "value_error",
                    "message": f"Payment method {payment.payment_id} not found",
                }
            if method.get("source") in ("gift_card", "certificate") and method.get("amount", 0) < payment.amount:
                return {
                    "status": "error",
                    "error_type": "value_error",
                    "message": f"Not enough balance in payment method {payment.payment_id}",
                }

        total_payment = sum(pay.amount for pay in p.payment_methods)
        if total_payment != total_price:
            return {
                "status": "error",
                "error_type": "value_error",
                "message": f"Payment amount does not add up, total price is {total_price}, but paid {total_payment}",
            }

        # All checks passed — deduct payment balances + (for certificates) remove method
        payment_history: List[dict] = []
        for payment in p.payment_methods:
            method = user["payment_methods"][payment.payment_id]
            if method.get("source") == "gift_card":
                method["amount"] -= payment.amount
            elif method.get("source") == "certificate":
                user["payment_methods"].pop(payment.payment_id)
            payment_history.append({"payment_id": payment.payment_id, "amount": payment.amount})

        # Deduct seats from each flight instance
        for flight_date_data in flight_date_data_refs:
            seats = flight_date_data.get("available_seats") or {}
            seats[cabin_value] = seats.get(cabin_value, 0) - n_passengers

        # Build reservation record
        reservation = Reservation(
            reservation_id=reservation_id,
            user_id=p.user_id,
            origin=p.origin,
            destination=p.destination,
            flight_type=p.flight_type,
            cabin=p.cabin,
            flights=reservation_flights,
            passengers=[passenger.model_dump() for passenger in p.passengers],
            payment_history=payment_history,
            created_at=_current_datetime(),
            total_baggages=p.total_baggages,
            nonfree_baggages=p.nonfree_baggages,
            insurance=p.insurance,
        ).model_dump()

        db.setdefault("reservations", {})[reservation_id] = reservation
        user.setdefault("reservations", []).append(reservation_id)

        self._record_action(
            {
                "action_type": "book_reservation",
                "name": "book_reservation",
                "arguments": p.model_dump(),
                "result": {"reservation_id": reservation_id},
            }
        )
        return reservation


@register_schema_tool_for_eval(domain="tau2_airline")
class CancelReservationTool(_Tau2WriteTool):
    DESCRIPTION = "Cancel the whole reservation. Reverses payment history and marks status='cancelled'."
    PARAMS_MODEL = CancelReservationParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {"reservation_id": {"type": "string", "description": "The reservation ID, such as 'ZFA04Y'."}}

    @property
    def required_properties(self) -> List[str]:
        return ["reservation_id"]

    def _do_work(self, p: CancelReservationParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        reservation = _get_reservation_dict(db, p.reservation_id)
        if reservation is None:
            return {
                "status": "error",
                "error_type": "not_found",
                "message": f"Reservation {p.reservation_id} not found",
            }

        # Reverse the payment history (tau2: append -amount entries, don't replace)
        refunds = [
            {"payment_id": pay.get("payment_id"), "amount": -pay.get("amount", 0)}
            for pay in reservation.get("payment_history") or []
        ]
        reservation.setdefault("payment_history", []).extend(refunds)
        reservation["status"] = "cancelled"
        # Note: tau2 deliberately skips seat-release on cancellation (their TODO).
        # Preserve that behavior so DB hashes match tau2's evaluator output.

        self._record_action(
            {
                "action_type": "cancel_reservation",
                "name": "cancel_reservation",
                "arguments": p.model_dump(),
                "result": {"reservation_id": p.reservation_id, "status": "cancelled"},
            }
        )
        return reservation


@register_schema_tool_for_eval(domain="tau2_airline")
class SendCertificateTool(_Tau2WriteTool):
    DESCRIPTION = (
        "Send a certificate to a user (e.g., as compensation). Adds a new certificate "
        "payment method to the user's profile. Be careful — irreversible."
    )
    PARAMS_MODEL = SendCertificateParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "user_id": {
                "type": "string",
                "description": "The ID of the user to book the reservation, such as 'sara_doe_496'.",
            },
            "amount": {"type": "integer", "description": "The amount of the certificate to send."},
        }

    @property
    def required_properties(self) -> List[str]:
        return ["user_id", "amount"]

    def _do_work(self, p: SendCertificateParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        user = _get_user_dict(db, p.user_id)
        if user is None:
            return {"status": "error", "error_type": "not_found", "message": f"User {p.user_id} not found"}

        # Mirror tau2: try the three preassigned ids in order, first unused wins.
        payment_methods = user.setdefault("payment_methods", {})
        for pid_suffix in _NEW_PAYMENT_IDS:
            payment_id = f"certificate_{pid_suffix}"
            if payment_id not in payment_methods:
                payment_methods[payment_id] = {
                    "source": "certificate",
                    "id": payment_id,
                    "amount": float(p.amount),
                }
                self._record_action(
                    {
                        "action_type": "send_certificate",
                        "name": "send_certificate",
                        "arguments": p.model_dump(),
                        "result": {"payment_id": payment_id, "amount": p.amount},
                    }
                )
                return {
                    "status": "success",
                    "message": f"Certificate {payment_id} added to user {p.user_id} with amount {p.amount}.",
                }
        return {"status": "error", "error_type": "value_error", "message": "Too many certificates"}


@register_schema_tool_for_eval(domain="tau2_airline")
class UpdateReservationBaggagesTool(_Tau2WriteTool):
    DESCRIPTION = "Update the baggage counts on a reservation. Charges $50 per additional non-free bag."
    PARAMS_MODEL = UpdateReservationBaggagesParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "reservation_id": {"type": "string", "description": "The reservation ID, such as 'ZFA04Y'."},
            "total_baggages": {
                "type": "integer",
                "description": "The updated total number of baggage items included in the reservation.",
            },
            "nonfree_baggages": {
                "type": "integer",
                "description": "The updated number of non-free baggage items included in the reservation.",
            },
            "payment_id": {
                "type": "string",
                "description": "The payment id stored in user profile, such as 'credit_card_7815826', 'gift_card_7815826'.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["reservation_id", "total_baggages", "nonfree_baggages", "payment_id"]

    def _do_work(self, p: UpdateReservationBaggagesParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        reservation = _get_reservation_dict(db, p.reservation_id)
        if reservation is None:
            return {
                "status": "error",
                "error_type": "not_found",
                "message": f"Reservation {p.reservation_id} not found",
            }
        user = _get_user_dict(db, reservation.get("user_id"))
        if user is None:
            return {
                "status": "error",
                "error_type": "not_found",
                "message": f"User {reservation.get('user_id')} not found",
            }

        delta_nonfree = max(0, p.nonfree_baggages - reservation.get("nonfree_baggages", 0))
        total_price = 50 * delta_nonfree

        payment_result = _payment_for_update(user, p.payment_id, total_price)
        if payment_result["status"] != "ok":
            return payment_result
        payment = payment_result.get("payment")
        if payment is not None:
            reservation.setdefault("payment_history", []).append(payment)

        reservation["total_baggages"] = p.total_baggages
        reservation["nonfree_baggages"] = p.nonfree_baggages

        self._record_action(
            {
                "action_type": "update_reservation_baggages",
                "name": "update_reservation_baggages",
                "arguments": p.model_dump(),
                "result": {"reservation_id": p.reservation_id, "charged": total_price},
            }
        )
        return reservation


@register_schema_tool_for_eval(domain="tau2_airline")
class UpdateReservationFlightsTool(_Tau2WriteTool):
    DESCRIPTION = (
        "Update the flight legs on an existing reservation. The flights array must include "
        "the ENTIRE new itinerary — even unchanged legs. Charges the price difference (if positive)."
    )
    PARAMS_MODEL = UpdateReservationFlightsParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "reservation_id": {"type": "string", "description": "The reservation ID, such as 'ZFA04Y'."},
            "cabin": {
                "type": "string",
                "enum": ["business", "economy", "basic_economy"],
                "description": "The cabin class of the reservation.",
            },
            "flights": {
                "type": "array",
                "description": (
                    "An array of objects containing details about each piece of flight in the ENTIRE new reservation. "
                    "Even if a flight segment is not changed, it should still be included in the array."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "flight_number": {"type": "string"},
                        "date": {"type": "string", "description": "YYYY-MM-DD"},
                    },
                    "required": ["flight_number", "date"],
                },
            },
            "payment_id": {
                "type": "string",
                "description": "The payment id stored in user profile, such as 'credit_card_7815826', 'gift_card_7815826'.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["reservation_id", "cabin", "flights", "payment_id"]

    def _do_work(self, p: UpdateReservationFlightsParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        reservation = _get_reservation_dict(db, p.reservation_id)
        if reservation is None:
            return {
                "status": "error",
                "error_type": "not_found",
                "message": f"Reservation {p.reservation_id} not found",
            }
        user = _get_user_dict(db, reservation.get("user_id"))
        if user is None:
            return {
                "status": "error",
                "error_type": "not_found",
                "message": f"User {reservation.get('user_id')} not found",
            }

        cabin_value = p.cabin.value if hasattr(p.cabin, "value") else p.cabin
        n_passengers = len(reservation.get("passengers") or [])
        existing_flights = reservation.get("flights") or []
        existing_cabin = reservation.get("cabin")

        new_reservation_flights: List[dict] = []
        total_price = 0
        for flight_info in p.flights:
            # If unchanged (same flight#, date, AND same cabin), keep the existing record's price.
            match = next(
                (
                    rf
                    for rf in existing_flights
                    if rf.get("flight_number") == flight_info.flight_number
                    and rf.get("date") == flight_info.date
                    and cabin_value == existing_cabin
                ),
                None,
            )
            if match is not None:
                total_price += match.get("price", 0) * n_passengers
                new_reservation_flights.append(copy.deepcopy(match))
                continue

            # Otherwise re-quote.
            flight = _get_flight_dict(db, flight_info.flight_number)
            if flight is None:
                return {
                    "status": "error",
                    "error_type": "not_found",
                    "message": f"Flight {flight_info.flight_number} not found",
                }
            flight_date_data = _get_flight_instance_dict(db, flight_info.flight_number, flight_info.date)
            if flight_date_data is None or flight_date_data.get("status") != "available":
                return {
                    "status": "error",
                    "error_type": "value_error",
                    "message": f"Flight {flight_info.flight_number} not available on date {flight_info.date}",
                }
            if (flight_date_data.get("available_seats") or {}).get(cabin_value, 0) < n_passengers:
                return {
                    "status": "error",
                    "error_type": "value_error",
                    "message": f"Not enough seats on flight {flight_info.flight_number}",
                }
            price = (flight_date_data.get("prices") or {}).get(cabin_value, 0)
            new_reservation_flights.append(
                ReservationFlight(
                    flight_number=flight_info.flight_number,
                    date=flight_info.date,
                    price=price,
                    origin=flight.get("origin"),
                    destination=flight.get("destination"),
                ).model_dump()
            )
            total_price += price * n_passengers

        # Deduct what was already paid for the existing flights
        total_price -= sum(rf.get("price", 0) for rf in existing_flights) * n_passengers

        payment_result = _payment_for_update(user, p.payment_id, total_price)
        if payment_result["status"] != "ok":
            return payment_result
        payment = payment_result.get("payment")
        if payment is not None:
            reservation.setdefault("payment_history", []).append(payment)

        reservation["flights"] = new_reservation_flights
        reservation["cabin"] = cabin_value
        # Note: tau2 deliberately defers flight-DB seat updates here ("assume it takes
        # time to be updated"). We preserve that to keep DB hashes comparable.

        self._record_action(
            {
                "action_type": "update_reservation_flights",
                "name": "update_reservation_flights",
                "arguments": p.model_dump(),
                "result": {"reservation_id": p.reservation_id, "charged": total_price},
            }
        )
        return reservation


@register_schema_tool_for_eval(domain="tau2_airline")
class UpdateReservationPassengersTool(_Tau2WriteTool):
    DESCRIPTION = "Update the passenger list on a reservation. Number of passengers must match the existing count."
    PARAMS_MODEL = UpdateReservationPassengersParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "reservation_id": {"type": "string", "description": "The reservation ID, such as 'ZFA04Y'."},
            "passengers": {
                "type": "array",
                "description": "An array of objects containing details about each passenger.",
                "items": {
                    "type": "object",
                    "properties": {
                        "first_name": {"type": "string"},
                        "last_name": {"type": "string"},
                        "dob": {"type": "string", "description": "YYYY-MM-DD"},
                    },
                    "required": ["first_name", "last_name", "dob"],
                },
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["reservation_id", "passengers"]

    def _do_work(self, p: UpdateReservationPassengersParams) -> dict:
        db = self.state.get("db")
        if not db:
            return _db_not_initialized()
        reservation = _get_reservation_dict(db, p.reservation_id)
        if reservation is None:
            return {
                "status": "error",
                "error_type": "not_found",
                "message": f"Reservation {p.reservation_id} not found",
            }
        existing_passengers = reservation.get("passengers") or []
        if len(p.passengers) != len(existing_passengers):
            return {
                "status": "error",
                "error_type": "value_error",
                "message": "Number of passengers does not match",
            }
        reservation["passengers"] = [passenger.model_dump() for passenger in p.passengers]

        self._record_action(
            {
                "action_type": "update_reservation_passengers",
                "name": "update_reservation_passengers",
                "arguments": p.model_dump(),
                "result": {"reservation_id": p.reservation_id},
            }
        )
        return reservation


# =============================================================================
# Generic tools (no DB mutation, but transfer_to_human_agents is action-recorded)
# =============================================================================


@register_schema_tool_for_eval(domain="tau2_airline")
class CalculateTool(_Tau2ReadTool):
    DESCRIPTION = (
        "Calculate the result of a mathematical expression. The expression can contain numbers, "
        "operators (+, -, *, /), parentheses, and spaces."
    )
    PARAMS_MODEL = CalculateParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "expression": {
                "type": "string",
                "description": (
                    "The mathematical expression to calculate, such as '2 + 2'. The expression can "
                    "contain numbers, operators (+, -, *, /), parentheses, and spaces."
                ),
            }
        }

    @property
    def required_properties(self) -> List[str]:
        return ["expression"]

    def _do_work(self, p: CalculateParams) -> dict:
        # Tau2's char allowlist — refuses anything not in this set.
        if not all(char in "0123456789+-*/(). " for char in p.expression):
            return {"status": "error", "error_type": "value_error", "message": "Invalid characters in expression"}
        try:
            result = round(float(eval(p.expression, {"__builtins__": None}, {})), 2)
        except Exception as e:
            return {"status": "error", "error_type": "value_error", "message": f"Invalid expression: {e}"}
        return {"status": "success", "result": str(result)}


@register_schema_tool_for_eval(domain="tau2_airline")
class TransferToHumanAgentsTool(_Tau2WriteTool):
    DESCRIPTION = (
        "Transfer the user to a human agent, with a summary of the user's issue. Only transfer if "
        "the user explicitly asks for a human or you cannot solve the issue with available tools."
    )
    PARAMS_MODEL = TransferToHumanAgentsParams

    @property
    def properties(self) -> Dict[str, Any]:
        return {"summary": {"type": "string", "description": "A summary of the user's issue."}}

    @property
    def required_properties(self) -> List[str]:
        return ["summary"]

    def _do_work(self, p: TransferToHumanAgentsParams) -> dict:
        # No DB mutation, but we still record the action so the gold-replay's
        # reference_answer includes it and live action-list comparison can match.
        # Tau2's evaluation_criteria.actions store compare_args=[] here, meaning
        # "match by name only, ignore the free-text summary."
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
# Convenience: ordered tool list (used by BaseScenario._build_tool_map)
# =============================================================================


TAU2_AIRLINE_TOOL_CLASSES: List[Type[StandardSchemaTool]] = [
    # reads
    GetUserDetailsTool,
    GetReservationDetailsTool,
    ListAllAirportsTool,
    SearchDirectFlightTool,
    SearchOnestopFlightTool,
    GetFlightStatusTool,
    # writes
    BookReservationTool,
    CancelReservationTool,
    SendCertificateTool,
    UpdateReservationBaggagesTool,
    UpdateReservationFlightsTool,
    UpdateReservationPassengersTool,
    # generics
    CalculateTool,
    TransferToHumanAgentsTool,
]


# Map from tau2 @is_tool method name → Tool class. Stable lookup for ``_build_tool_map``.
TAU2_AIRLINE_TOOL_NAME_TO_CLASS: Dict[str, Type[StandardSchemaTool]] = {
    "get_user_details": GetUserDetailsTool,
    "get_reservation_details": GetReservationDetailsTool,
    "list_all_airports": ListAllAirportsTool,
    "search_direct_flight": SearchDirectFlightTool,
    "search_onestop_flight": SearchOnestopFlightTool,
    "get_flight_status": GetFlightStatusTool,
    "book_reservation": BookReservationTool,
    "cancel_reservation": CancelReservationTool,
    "send_certificate": SendCertificateTool,
    "update_reservation_baggages": UpdateReservationBaggagesTool,
    "update_reservation_flights": UpdateReservationFlightsTool,
    "update_reservation_passengers": UpdateReservationPassengersTool,
    "calculate": CalculateTool,
    "transfer_to_human_agents": TransferToHumanAgentsTool,
}

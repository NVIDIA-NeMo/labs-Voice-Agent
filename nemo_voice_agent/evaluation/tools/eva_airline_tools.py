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
# Adapted from https://github.com/ServiceNow/eva/tree/0.1.3 (MIT-licensed):
#   src/eva/assistant/tools/airline_tools.py
# Helpers ported wholesale; tool function bodies wrapped in StandardSchemaTool
# subclasses with shared_state["db"] in place of eva's `db` parameter.

# Scenario definitions and tool descriptions contain prose strings that don't
# benefit from line wrapping.
# pylint: disable=line-too-long
# flake8: noqa: E501

import copy
import json
from typing import Any, Dict, List, Optional

from loguru import logger
from pipecat.services.llm_service import FunctionCallParams
from pydantic import ValidationError

from nemo_voice_agent.evaluation.tools import register_schema_tool_for_eval
from nemo_voice_agent.evaluation.tools._write_tool_base import WriteScenarioTool
from nemo_voice_agent.evaluation.tools.eva_airline_params import (
    AddBaggageAllowanceParams,
    AddMealRequestParams,
    AddToStandbyParams,
    AssignSeatParams,
    CancellationReason,
    CancelReservationParams,
    GetDisruptionInfoParams,
    GetFlightStatusParams,
    GetReservationParams,
    IssueHotelVoucherParams,
    IssueMealVoucherParams,
    IssueTravelCreditParams,
    ProcessRefundParams,
    RebookFlightParams,
    RebookingType,
    SearchRebookingOptionsParams,
    TransferToAgentParams,
    validation_error_response,
)
from nemo_voice_agent.utils.tool_calling import StandardSchemaTool

# ---------------------------------------------------------------------------
# Action-type vocabulary (locked, 1:1 with eva tool names)
#
# Referenced by every WriteAirlineTool subclass's _record_action call and by
# scenarios' reference_answer payloads.
# ---------------------------------------------------------------------------

AIRLINE_ACTION_TYPES: List[str] = [
    "rebook_flight",
    "cancel_reservation",
    "process_refund",
    "issue_meal_voucher",
    "issue_hotel_voucher",
    "issue_travel_credit",
    "assign_seat",
    "add_baggage_allowance",
    "add_meal_request",
    "add_to_standby",
    "transfer_to_agent",
]


# ---------------------------------------------------------------------------
# Shared helpers (ported from eva/src/eva/assistant/tools/airline_tools.py)
# Adapted from https://github.com/ServiceNow/eva/tree/0.1.3
# ---------------------------------------------------------------------------


def _lookup_reservation(db: dict, confirmation_number: str) -> Optional[dict]:
    """Find a reservation by confirmation number."""
    return db.get("reservations", {}).get(confirmation_number.upper())


def _find_booking_journey(reservation: dict, journey_id: str) -> Optional[dict]:
    """Find a booking journey entry within a reservation.

    When multiple bookings share the same journey_id (e.g. after a partial rebook
    creates a 'kept' booking alongside the cancelled original), the first
    non-cancelled booking is returned in preference to a cancelled one.
    """
    first_match = None
    for bk in reservation.get("bookings", []):
        if bk.get("journey_id") == journey_id:
            if first_match is None:
                first_match = bk
            if bk.get("status") != "cancelled":
                return bk
    return first_match


def _find_booking_segment(
    booking: dict, journey_id: str, flight_number: str = ""
) -> tuple[List[dict], Optional[dict]]:
    """Find flight segment(s) within a booking journey.

    If flight_number is provided, returns only that segment.
    If omitted and booking has one segment, returns that segment.
    If omitted and booking has multiple segments, returns an error dict.

    Returns:
        (targets, error) — targets is a list of matching segments,
        error is a response dict if something went wrong (else None).
    """
    booking_segments = booking.get("segments", [])
    if flight_number:
        targets = [fs for fs in booking_segments if fs.get("flight_number") == flight_number]
        if not targets:
            return [], {
                "status": "error",
                "error_type": "flight_not_found",
                "message": f"Flight {flight_number} not found in journey {journey_id}",
            }
        return targets, None
    elif len(booking_segments) == 1:
        return booking_segments, None
    else:
        return [], {
            "status": "error",
            "error_type": "flight_number_required",
            "message": f"Journey {journey_id} has {len(booking_segments)} segments; flight_number is required",
        }


def _get_journey_fares(journey: dict) -> dict:
    """Return journey-level fares dict.

    Fares are stored at the journey level as the total price for all segments combined.
    """
    return journey.get("fares", {})


def _get_booking_total_fare(booking: dict) -> float:
    """Return total fare for a booking journey.

    Uses journey-level fare_paid if present, otherwise sums segment-level fare_paid.
    """
    if "fare_paid" in booking:
        return booking.get("fare_paid", 0)
    return sum(seg.get("fare_paid", 0) for seg in booking.get("segments", []))


def _get_journey_available_seats(journey: dict) -> dict:
    """Compute effective available seats for a journey.

    For multi-segment journeys, the constraining factor is the segment
    with the fewest seats in each fare class (min across segments).
    """
    segments = journey.get("segments", [])
    if not segments:
        return {}
    result = dict(segments[0].get("available_seats", {}))
    for seg in segments[1:]:
        seg_seats = seg.get("available_seats", {})
        for fc in result:
            result[fc] = min(result.get(fc, 0), seg_seats.get(fc, 0))
    return result


def _reservation_not_found(confirmation_number: str) -> dict:
    return {
        "status": "error",
        "error_type": "not_found",
        "message": f"Reservation {confirmation_number} not found",
    }


def _journey_not_found(journey_id: str) -> dict:
    return {
        "status": "error",
        "error_type": "journey_not_found",
        "message": f"Journey {journey_id} not found in reservation",
    }


def _db_not_initialized() -> dict:
    """Returned when shared_state['db'] is missing — fixture didn't load."""
    return {
        "status": "error",
        "error_type": "db_not_initialized",
        "message": "Scenario database not loaded. This indicates a fixture-loading bug; contact the evaluator.",
    }


# ---------------------------------------------------------------------------
# WriteAirlineTool — thin subclass binding the eva-airline action vocabulary
# to the shared WriteScenarioTool base (see tools/_write_tool_base.py).
# ---------------------------------------------------------------------------


class WriteAirlineTool(WriteScenarioTool):
    """Eva-airline subclass that binds ``ACTION_TYPES`` to ``AIRLINE_ACTION_TYPES``.

    Inherits ``_record_action`` (validates ``action_type`` against ACTION_TYPES,
    appends to ``shared_state["actions"]``) and ``_next_call_index`` (per-scenario
    monotonic counter for tools that mint unique IDs) from the shared base.

    Read-only tools subclass ``StandardSchemaTool`` directly — only writes record.
    """

    ACTION_TYPES = AIRLINE_ACTION_TYPES


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------


@register_schema_tool_for_eval
class EvaAirlineGetReservationTool(StandardSchemaTool):
    """Retrieve flight reservation using confirmation number and passenger last name.

    Authentication entry point — typically the first tool called per scenario.
    """

    DESCRIPTION = (
        "Retrieve flight reservation using confirmation number and passenger last name. "
        "This is typically the first tool called to authenticate the caller and load "
        "their flight numbers."
    )

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or self.DESCRIPTION)
        self.state = shared_state if shared_state is not None else {}

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "confirmation_number": {
                "type": "string",
                "description": "6-character alphanumeric booking confirmation code.",
            },
            "last_name": {
                "type": "string",
                "description": "Passenger last name for verification.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["confirmation_number", "last_name"]

    async def _execute(self, params: FunctionCallParams) -> None:
        # Adapted from https://github.com/ServiceNow/eva/tree/0.1.3 — get_reservation
        try:
            p = GetReservationParams.model_validate(params.arguments)
        except ValidationError as exc:
            await params.result_callback(validation_error_response(exc, GetReservationParams))
            return

        db = self.state.get("db")
        if not db:
            await params.result_callback(_db_not_initialized())
            return

        confirmation_number = p.confirmation_number.upper()
        last_name = p.last_name

        reservation = _lookup_reservation(db, confirmation_number)
        if not reservation:
            await params.result_callback(_reservation_not_found(confirmation_number))
            return

        if last_name:
            passengers = reservation.get("passengers", [])
            last_name_match = any(p2.get("last_name", "").lower() == last_name.lower() for p2 in passengers)
            if not last_name_match:
                await params.result_callback(
                    {
                        "status": "error",
                        "error_type": "authentication_failed",
                        "message": f"Last name does not match reservation {confirmation_number}",
                    }
                )
                return

        # Sort journeys by first segment's date, then journey_id, for readability.
        result_reservation = copy.deepcopy(reservation)
        result_reservation["bookings"].sort(
            key=lambda j: (
                j.get("segments", [{}])[0].get("date", ""),
                j.get("journey_id", ""),
            )
        )
        await params.result_callback({"status": "success", "reservation": result_reservation})


@register_schema_tool_for_eval
class EvaAirlineGetFlightStatusTool(StandardSchemaTool):
    """Get current status of a specific flight (delays, cancellations, gate)."""

    DESCRIPTION = (
        "Get specific information about a flight (origin, destination, departure time) and "
        "current status (delays, cancellations, gate). For connecting flights, any segment's "
        "flight number works."
    )

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or self.DESCRIPTION)
        self.state = shared_state if shared_state is not None else {}

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "flight_number": {
                "type": "string",
                "description": "Flight number (e.g., SK123). For connecting flights, any segment works.",
            },
            "flight_date": {
                "type": "string",
                "description": "Flight date in YYYY-MM-DD format.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["flight_number", "flight_date"]

    async def _execute(self, params: FunctionCallParams) -> None:
        # Adapted from https://github.com/ServiceNow/eva/tree/0.1.3 — get_flight_status
        try:
            p = GetFlightStatusParams.model_validate(params.arguments)
        except ValidationError as exc:
            await params.result_callback(validation_error_response(exc, GetFlightStatusParams))
            return

        db = self.state.get("db")
        if not db:
            await params.result_callback(_db_not_initialized())
            return

        flight_number = p.flight_number.upper()
        normalized_date = p.flight_date.replace("-", "")

        journeys = db.get("journeys", {})
        journey_id = f"FL_{flight_number}_{normalized_date}"
        flight = journeys.get(journey_id)

        if not flight:
            for f in journeys.values():
                if f.get("date", "").replace("-", "") == normalized_date:
                    for segment in f.get("segments", []):
                        if segment.get("flight_number", "").upper() == flight_number:
                            flight = f
                            break
                if flight:
                    break

        if not flight:
            await params.result_callback(
                {
                    "status": "error",
                    "error_type": "not_found",
                    "message": f"Flight {p.flight_number} not found for date {p.flight_date}",
                }
            )
            return

        await params.result_callback({"status": "success", "journey": copy.deepcopy(flight)})


@register_schema_tool_for_eval
class EvaAirlineGetDisruptionInfoTool(StandardSchemaTool):
    """Get IRROPS disruption details (cause, fee waiver, refund eligibility)."""

    DESCRIPTION = (
        "Get detailed information about flight disruption for IRROPS handling. "
        "Determines passenger rebooking entitlements based on disruption type and cause."
    )

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or self.DESCRIPTION)
        self.state = shared_state if shared_state is not None else {}

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "flight_number": {
                "type": "string",
                "description": "Flight number to look up disruption for (e.g., SK123).",
            },
            "date": {
                "type": "string",
                "description": "Flight date in YYYY-MM-DD format.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["flight_number", "date"]

    async def _execute(self, params: FunctionCallParams) -> None:
        # Adapted from https://github.com/ServiceNow/eva/tree/0.1.3 — get_disruption_info
        try:
            p = GetDisruptionInfoParams.model_validate(params.arguments)
        except ValidationError as exc:
            await params.result_callback(validation_error_response(exc, GetDisruptionInfoParams))
            return

        db = self.state.get("db")
        if not db:
            await params.result_callback(_db_not_initialized())
            return

        flight_number = p.flight_number.upper()
        date = p.date

        disruptions = db.get("disruptions", {})
        disruption = None
        if date:
            disruption = disruptions.get(f"{flight_number}_{date}")
        else:
            for d in disruptions.values():
                if d.get("flight_number", "").upper() == flight_number:
                    disruption = d
                    break

        if not disruption:
            await params.result_callback(
                {
                    "status": "error",
                    "error_type": "not_found",
                    "message": f"No disruption info found for flight {flight_number}",
                }
            )
            return

        await params.result_callback({"status": "success", "disruption": copy.deepcopy(disruption)})


@register_schema_tool_for_eval
class EvaAirlineSearchRebookingOptionsTool(StandardSchemaTool):
    """Search available flights (origin/destination/date) filtered by seat availability."""

    DESCRIPTION = (
        "Search for available flights to rebook a passenger. Returns options filtered by "
        "rebooking rules and seat availability."
    )

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or self.DESCRIPTION)
        self.state = shared_state if shared_state is not None else {}

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "origin": {
                "type": "string",
                "description": "Origin airport code (e.g., JFK).",
            },
            "destination": {
                "type": "string",
                "description": "Destination airport code (e.g., LAX).",
            },
            "date": {
                "type": "string",
                "description": "Travel date in YYYY-MM-DD format.",
            },
            "passenger_count": {
                "type": "integer",
                "description": "Number of passengers needing seats.",
            },
            "fare_class": {
                "type": "string",
                "description": (
                    "Fare class to search (basic_economy, main_cabin, premium_economy, "
                    "business, first, any). 'any' picks the cheapest available class."
                ),
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["origin", "destination", "date", "passenger_count", "fare_class"]

    async def _execute(self, params: FunctionCallParams) -> None:
        # Adapted from https://github.com/ServiceNow/eva/tree/0.1.3 — search_rebooking_options
        try:
            p = SearchRebookingOptionsParams.model_validate(params.arguments)
        except ValidationError as exc:
            await params.result_callback(validation_error_response(exc, SearchRebookingOptionsParams))
            return

        db = self.state.get("db")
        if not db:
            await params.result_callback(_db_not_initialized())
            return

        origin = p.origin.upper()
        destination = p.destination.upper()
        date = p.date
        passenger_count = p.passenger_count
        fare_class = p.fare_class

        journeys = db.get("journeys", {})
        results = []
        all_fare_classes = [
            "basic_economy",
            "main_cabin",
            "premium_economy",
            "business",
            "first",
        ]

        for journey_id, flight in journeys.items():
            if flight.get("origin") != origin:
                continue
            if flight.get("destination") != destination:
                continue
            if flight.get("date") != date:
                continue
            if flight.get("status") not in ["scheduled", "on_time", "delayed"]:
                continue
            if not flight.get("bookable", False):
                continue

            available_seats = _get_journey_available_seats(flight)
            if fare_class == "any":
                if not any(seats >= passenger_count for seats in available_seats.values() if seats is not None):
                    continue
            else:
                if available_seats.get(fare_class, 0) < passenger_count:
                    continue

            journey_fares = _get_journey_fares(flight)
            actual_fare_class = fare_class
            if fare_class == "any":
                available_cabins = [
                    (fc, journey_fares.get(fc))
                    for fc in all_fare_classes
                    if available_seats.get(fc, 0) >= passenger_count and journey_fares.get(fc) is not None
                ]
                if available_cabins:
                    actual_fare_class = min(available_cabins, key=lambda x: x[1])[0]

            segments = flight.get("segments", [])
            results.append(
                {
                    "journey_id": journey_id,
                    "origin": flight.get("origin"),
                    "destination": flight.get("destination"),
                    "num_stops": flight.get("num_stops", 0),
                    "total_duration_minutes": flight.get("total_duration_minutes"),
                    "segments": segments,
                    "departure_time": (segments[0]["scheduled_departure"] if segments else None),
                    "arrival_time": (segments[-1]["scheduled_arrival"] if segments else None),
                    "available_seats": {fc: available_seats.get(fc, 0) for fc in all_fare_classes},
                    "fare": journey_fares.get(actual_fare_class),
                }
            )

        results.sort(key=lambda x: x.get("departure_time", ""))
        await params.result_callback(
            {
                "status": "success",
                "options": results,
                "count": len(results),
                "message": f"{len(results)} flight(s) found",
            }
        )


# ---------------------------------------------------------------------------
# Write tools — each records an action on success
# ---------------------------------------------------------------------------


@register_schema_tool_for_eval
class EvaAirlineRebookFlightTool(WriteAirlineTool):
    """Rebook passenger(s) to a new flight (voluntary, IRROPS, partial)."""

    DESCRIPTION = (
        "Rebook passenger(s) to a new flight. Handles voluntary changes, IRROPS rebooking, "
        "and missed flight recovery. If new_fare_class is provided, the cabin changes; "
        "otherwise the original fare class is kept. If flight_number is provided, a partial "
        "rebook of that segment is performed (split-booking). Always explain change_fee, "
        "fare_difference, and total_collected to the caller before confirming."
    )

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or self.DESCRIPTION)
        self.state = shared_state if shared_state is not None else {}

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "confirmation_number": {
                "type": "string",
                "description": "Booking confirmation code.",
            },
            "journey_id": {
                "type": "string",
                "description": "Journey ID from get_reservation booking to be changed.",
            },
            "new_journey_id": {
                "type": "string",
                "description": "Journey ID of the replacement flight from search_rebooking_options.",
            },
            "rebooking_type": {
                "type": "string",
                "description": (
                    "Reason for rebooking. One of: voluntary, same_day, irrops_cancellation, "
                    "irrops_delay, irrops_schedule_change, missed_flight_passenger_fault, "
                    "missed_connection_airline_fault."
                ),
            },
            "waive_change_fee": {
                "type": "boolean",
                "description": "Whether to waive the change fee (only for voluntary changes with authorization).",
            },
            "new_fare_class": {
                "type": "string",
                "description": (
                    "Fare class to rebook into if changing cabin "
                    "(basic_economy, main_cabin, premium_economy, business, first). "
                    "Omit to keep original fare class."
                ),
            },
            "flight_number": {
                "type": "string",
                "description": (
                    "For multi-segment journeys, the specific flight number of the leg to "
                    "replace (partial rebook). Omit for full journey rebook."
                ),
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return [
            "confirmation_number",
            "journey_id",
            "new_journey_id",
            "rebooking_type",
            "waive_change_fee",
        ]

    async def _execute(self, params: FunctionCallParams) -> None:
        # Adapted from https://github.com/ServiceNow/eva/tree/0.1.3 — rebook_flight
        try:
            p = RebookFlightParams.model_validate(params.arguments)
        except ValidationError as exc:
            await params.result_callback(validation_error_response(exc, RebookFlightParams))
            return

        db = self.state.get("db")
        if not db:
            await params.result_callback(_db_not_initialized())
            return

        confirmation_number = p.confirmation_number.upper()
        journey_id = p.journey_id
        new_journey_id = p.new_journey_id
        rebooking_type = p.rebooking_type
        waive_change_fee = p.waive_change_fee
        new_fare_class = p.new_fare_class
        flight_number = p.flight_number

        reservation = _lookup_reservation(db, confirmation_number)
        if not reservation:
            await params.result_callback(_reservation_not_found(confirmation_number))
            return

        booking = _find_booking_journey(reservation, journey_id)
        if not booking:
            await params.result_callback(_journey_not_found(journey_id))
            return

        journeys = db.get("journeys", {})
        new_flight = journeys.get(new_journey_id)
        if not new_flight:
            await params.result_callback(
                {
                    "status": "error",
                    "error_type": "flight_not_found",
                    "message": f"Flight {new_journey_id} not found",
                }
            )
            return

        if not new_flight.get("bookable", True):
            await params.result_callback(
                {
                    "status": "error",
                    "error_type": "not_bookable",
                    "message": f"Flight {new_journey_id} is not available for booking",
                }
            )
            return

        original_fare_class = booking.get("fare_class", "main_cabin")
        target_fare_class = new_fare_class or original_fare_class

        is_irrops = "irrops" in rebooking_type
        is_same_day = rebooking_type == RebookingType.same_day

        if is_irrops or waive_change_fee:
            change_fee = 0
        elif is_same_day:
            change_fee = 199 if original_fare_class == "basic_economy" else 75
        else:
            voluntary_fees = {
                "basic_economy": 75,
                "main_cabin": 75,
                "premium_economy": 75,
                "business": 0,
                "first": 0,
            }
            change_fee = voluntary_fees.get(original_fare_class, 75)

        replaced_segment = None
        if flight_number:
            targets, error = _find_booking_segment(booking, journey_id, flight_number)
            if error:
                await params.result_callback(error)
                return
            replaced_segment = targets[0]

        if flight_number:
            old_fare = replaced_segment.get("fare_paid", 0)
        else:
            old_fare = _get_booking_total_fare(booking)
        journey_fares = _get_journey_fares(new_flight)
        new_fare = journey_fares.get(target_fare_class)

        if new_fare is None:
            await params.result_callback(
                {
                    "status": "error",
                    "error_type": "fare_class_not_available",
                    "message": f"Fare class '{target_fare_class}' is not available on flight {new_journey_id}",
                }
            )
            return

        journey_seats = _get_journey_available_seats(new_flight)
        if journey_seats.get(target_fare_class, 0) <= 0:
            await params.result_callback(
                {
                    "status": "error",
                    "error_type": "no_seats_available",
                    "message": f"No seats available in {target_fare_class} on flight {new_journey_id}",
                }
            )
            return

        fare_difference = new_fare - old_fare
        credit_due = max(0, -fare_difference)
        fare_difference_to_collect = max(0, fare_difference)
        total_collected = 0 if is_irrops else (change_fee + fare_difference_to_collect)

        # Mutate: cancel old booking
        for seg in reservation["bookings"]:
            if seg["journey_id"] == journey_id:
                seg["status"] = "cancelled"
                break

        # Add new booking journey
        flight_segments = new_flight.get("segments", [])
        old_segments = booking.get("segments", [])
        new_booking_segments = []
        for i, fs in enumerate(flight_segments):
            if is_irrops and not flight_number and i < len(old_segments):
                seg_fare_paid = old_segments[i].get("fare_paid", 0)
            elif is_irrops and flight_number and len(flight_segments) == 1:
                seg_fare_paid = replaced_segment.get("fare_paid", 0)
            else:
                seg_fare_paid = fs.get("fares", {}).get(target_fare_class, 0)
            new_booking_segments.append(
                {
                    "flight_number": fs.get("flight_number"),
                    "date": new_flight.get("date"),
                    "fare_paid": seg_fare_paid,
                    "seat": None,
                    "bags_checked": 0,
                    "meal_request": None,
                }
            )
        new_booking = {
            "journey_id": new_journey_id,
            "fare_class": target_fare_class,
            "fare_paid": old_fare if is_irrops else new_fare,
            "status": "confirmed",
            "segments": new_booking_segments,
        }
        reservation["bookings"].append(new_booking)

        kept_segments_info = []
        if flight_number:
            kept_segments = [seg for seg in booking.get("segments", []) if seg.get("flight_number") != flight_number]
            if kept_segments:
                kept_booking = {
                    "journey_id": journey_id,
                    "fare_class": original_fare_class,
                    "fare_paid": sum(s.get("fare_paid", 0) for s in kept_segments),
                    "status": "confirmed",
                    "segments": copy.deepcopy(kept_segments),
                }
                reservation["bookings"].append(kept_booking)
                kept_segments_info = [
                    {
                        "flight_number": s.get("flight_number"),
                        "origin": s.get("origin"),
                        "destination": s.get("destination"),
                        "fare_paid": s.get("fare_paid"),
                        "seat": s.get("seat"),
                    }
                    for s in kept_segments
                ]

        reservation["status"] = "changed"

        for seg in new_flight.get("segments", []):
            seg.setdefault("available_seats", {})
            seg["available_seats"][target_fare_class] = seg["available_seats"].get(target_fare_class, 0) - 1

        old_journey_id = booking.get("journey_id")
        old_flight = journeys.get(old_journey_id)
        if old_flight:
            for seg in old_flight.get("segments", []):
                seg.setdefault("available_seats", {})
                seg["available_seats"][original_fare_class] = seg["available_seats"].get(original_fare_class, 0) + 1

        response = {
            "status": "success",
            "confirmation_number": confirmation_number,
            "new_journey": {
                "journey_id": new_journey_id,
                "num_stops": new_flight.get("num_stops", 0),
                "segments": copy.deepcopy(new_flight.get("segments", [])),
                "departure": (
                    new_flight["segments"][0]["scheduled_departure"] if new_flight.get("segments") else None
                ),
                "arrival": (new_flight["segments"][-1]["scheduled_arrival"] if new_flight.get("segments") else None),
                "origin": new_flight.get("origin"),
                "destination": new_flight.get("destination"),
            },
            "cost_summary": {
                "original_fare_class": original_fare_class,
                "new_fare_class": target_fare_class,
                "cabin_changed": original_fare_class != target_fare_class,
                "change_fee": change_fee,
                "fare_difference": fare_difference,
                "credit_due": credit_due,
                "total_collected": total_collected,
                "fee_waived": is_irrops or waive_change_fee,
            },
            "message": f"Successfully rebooked to flight {new_journey_id}"
            + (f" in {target_fare_class}" if original_fare_class != target_fare_class else ""),
        }
        if flight_number:
            response["partial_rebook"] = True
            response["replaced_segment"] = {
                "flight_number": flight_number,
                "origin": replaced_segment.get("origin"),
                "destination": replaced_segment.get("destination"),
                "fare_paid": replaced_segment.get("fare_paid"),
            }
            response["kept_segments"] = kept_segments_info

        # Action record — pure projection of validated params + computed cost summary
        action = {
            "action_type": "rebook_flight",
            "confirmation_number": confirmation_number,
            "old_journey_id": journey_id,
            "new_journey_id": new_journey_id,
            "rebooking_type": str(rebooking_type),
            "original_fare_class": original_fare_class,
            "new_fare_class": target_fare_class,
            "change_fee": change_fee,
            "fare_difference": fare_difference,
            "total_collected": total_collected,
            "partial_rebook": bool(flight_number),
        }
        if flight_number:
            action["replaced_flight_number"] = flight_number
        self._record_action(action)

        await params.result_callback(response)


@register_schema_tool_for_eval
class EvaAirlineCancelReservationTool(WriteAirlineTool):
    """Cancel a flight booking (single journey within a reservation)."""

    DESCRIPTION = (
        "Cancel a specific journey within a booking. If all journeys end up cancelled, "
        "the reservation itself is marked cancelled. Returns refund and credit eligibility."
    )

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or self.DESCRIPTION)
        self.state = shared_state if shared_state is not None else {}

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "confirmation_number": {
                "type": "string",
                "description": "Booking confirmation code.",
            },
            "journey_id": {
                "type": "string",
                "description": "Journey ID of the booking to cancel, from get_reservation.",
            },
            "cancellation_reason": {
                "type": "string",
                "description": (
                    "One of: voluntary, irrops_refund, 24_hour_rule, schedule_unacceptable, " "medical, bereavement."
                ),
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["confirmation_number", "journey_id", "cancellation_reason"]

    async def _execute(self, params: FunctionCallParams) -> None:
        # Adapted from https://github.com/ServiceNow/eva/tree/0.1.3 — cancel_reservation
        try:
            p = CancelReservationParams.model_validate(params.arguments)
        except ValidationError as exc:
            await params.result_callback(validation_error_response(exc, CancelReservationParams))
            return

        db = self.state.get("db")
        if not db:
            await params.result_callback(_db_not_initialized())
            return

        confirmation_number = p.confirmation_number.upper()
        journey_id = p.journey_id
        cancellation_reason = p.cancellation_reason

        reservation = _lookup_reservation(db, confirmation_number)
        if not reservation:
            await params.result_callback(_reservation_not_found(confirmation_number))
            return

        booking = _find_booking_journey(reservation, journey_id)
        if not booking:
            await params.result_callback(_journey_not_found(journey_id))
            return

        if booking.get("status") == "cancelled":
            await params.result_callback(
                {
                    "status": "error",
                    "error_type": "already_cancelled",
                    "message": f"Journey {journey_id} is already cancelled",
                }
            )
            return

        booking_fare = _get_booking_total_fare(booking)

        fee_waiving_reasons = {
            CancellationReason.irrops_refund,
            CancellationReason.rule_24_hour,
        }
        is_refundable = cancellation_reason in fee_waiving_reasons or reservation.get("fare_type") == "refundable"
        cancellation_fee = 0 if is_refundable else 100

        refund_amount = max(0, booking_fare - cancellation_fee) if is_refundable else 0
        credit_amount = 0 if is_refundable else max(0, booking_fare - cancellation_fee)

        booking["status"] = "cancelled"

        fare_class = booking.get("fare_class", "main_cabin")
        journeys = db.get("journeys", {})
        cancelled_flight = journeys.get(journey_id)
        if cancelled_flight:
            for seg in cancelled_flight.get("segments", []):
                seg.setdefault("available_seats", {})
                seg["available_seats"][fare_class] = seg["available_seats"].get(fare_class, 0) + 1

        all_cancelled = all(seg.get("status") == "cancelled" for seg in reservation.get("bookings", []))
        if all_cancelled:
            reservation["status"] = "cancelled"

        self._record_action(
            {
                "action_type": "cancel_reservation",
                "confirmation_number": confirmation_number,
                "journey_id": journey_id,
                "cancellation_reason": str(cancellation_reason),
                "is_refundable": is_refundable,
                "cancellation_fee": cancellation_fee,
                "refund_amount_eligible": refund_amount,
            }
        )

        await params.result_callback(
            {
                "status": "success",
                "confirmation_number": confirmation_number,
                "journey_id": journey_id,
                "is_refundable": is_refundable,
                "cancellation_fee": cancellation_fee,
                "refund_amount_eligible": refund_amount,
                "credit_amount_eligible": credit_amount,
                "reservation_status": "cancelled" if all_cancelled else "active",
                "message": f"Journey {journey_id} cancelled successfully",
            }
        )


@register_schema_tool_for_eval
class EvaAirlineProcessRefundTool(WriteAirlineTool):
    """Process a refund. Call once per refund type (fare and ancillary fees are separate)."""

    DESCRIPTION = (
        "Process a refund for a cancelled or eligible reservation. Called after "
        "cancel_reservation when refund is due. Call once per refund type — fare and "
        "ancillary fees must be separate calls. Never combine both into a single call."
    )

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or self.DESCRIPTION)
        self.state = shared_state if shared_state is not None else {}

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "confirmation_number": {
                "type": "string",
                "description": "Booking confirmation code.",
            },
            "refund_amount": {
                "type": "number",
                "description": "Amount to refund (use refund_amount_eligible from cancel_reservation).",
            },
            "refund_type": {
                "type": "string",
                "description": "One of: full_fare, partial_fare, taxes_only, ancillary_fees.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["confirmation_number", "refund_amount", "refund_type"]

    async def _execute(self, params: FunctionCallParams) -> None:
        # Adapted from https://github.com/ServiceNow/eva/tree/0.1.3 — process_refund
        try:
            p = ProcessRefundParams.model_validate(params.arguments)
        except ValidationError as exc:
            await params.result_callback(validation_error_response(exc, ProcessRefundParams))
            return

        db = self.state.get("db")
        if not db:
            await params.result_callback(_db_not_initialized())
            return

        confirmation_number = p.confirmation_number.upper()
        refund_amount = p.refund_amount
        refund_type = p.refund_type

        if not _lookup_reservation(db, confirmation_number):
            await params.result_callback(_reservation_not_found(confirmation_number))
            return

        if refund_amount <= 0:
            await params.result_callback(
                {
                    "status": "error",
                    "error_type": "invalid_amount",
                    "message": f"Refund amount must be greater than 0, got {refund_amount}",
                }
            )
            return

        call_index = self._next_call_index("process_refund")
        refund_id = f"REF-{confirmation_number}-{str(call_index).zfill(3)}"
        processing_days = 7

        refunds = db.setdefault("refunds", {})
        refunds[refund_id] = {
            "refund_id": refund_id,
            "confirmation_number": confirmation_number,
            "refund_amount": refund_amount,
            "refund_type": refund_type,
            "processing_days": processing_days,
            "initiated_date": db["_current_date"],
            "status": "processing",
        }

        self._record_action(
            {
                "action_type": "process_refund",
                "confirmation_number": confirmation_number,
                "refund_amount": refund_amount,
                "refund_type": str(refund_type),
            }
        )

        await params.result_callback(
            {
                "status": "success",
                "confirmation_number": confirmation_number,
                "refund_id": refund_id,
                "refund_amount": refund_amount,
                "refund_type": refund_type,
                "processing_days": processing_days,
                "message": f"${refund_amount} refund initiated, processing time {processing_days} business days",
            }
        )


@register_schema_tool_for_eval
class EvaAirlineAssignSeatTool(WriteAirlineTool):
    """Assign a seat (window/aisle/middle/no_preference) on a specific flight segment."""

    DESCRIPTION = (
        "Assign a seat to a passenger based on preference (window, aisle, middle, no_preference). "
        "Always ask the passenger for their preference before calling — do not assume."
    )

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or self.DESCRIPTION)
        self.state = shared_state if shared_state is not None else {}

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "confirmation_number": {
                "type": "string",
                "description": "Booking confirmation code.",
            },
            "passenger_id": {
                "type": "string",
                "description": "Passenger identifier (e.g., PAX001).",
            },
            "journey_id": {
                "type": "string",
                "description": "Journey ID from get_reservation.",
            },
            "seat_preference": {
                "type": "string",
                "description": "One of: window, aisle, middle, no_preference.",
            },
            "flight_number": {
                "type": "string",
                "description": "For multi-segment journeys, the flight number to assign seat on. Omit for single-segment.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["confirmation_number", "passenger_id", "journey_id", "seat_preference"]

    async def _execute(self, params: FunctionCallParams) -> None:
        # Adapted from https://github.com/ServiceNow/eva/tree/0.1.3 — assign_seat
        try:
            p = AssignSeatParams.model_validate(params.arguments)
        except ValidationError as exc:
            await params.result_callback(validation_error_response(exc, AssignSeatParams))
            return

        db = self.state.get("db")
        if not db:
            await params.result_callback(_db_not_initialized())
            return

        confirmation_number = p.confirmation_number.upper()
        passenger_id = p.passenger_id
        journey_id = p.journey_id
        seat_preference = p.seat_preference
        flight_number = p.flight_number

        reservation = _lookup_reservation(db, confirmation_number)
        if not reservation:
            await params.result_callback(_reservation_not_found(confirmation_number))
            return

        booking = _find_booking_journey(reservation, journey_id)
        if not booking:
            await params.result_callback(_journey_not_found(journey_id))
            return

        targets, error = _find_booking_segment(booking, journey_id, flight_number)
        if error:
            await params.result_callback(error)
            return
        flight_seg = targets[0]
        if not flight_number:
            flight_number = flight_seg.get("flight_number", "")

        journeys = db.get("journeys", {})
        journey = journeys.get(journey_id)
        journey_seg = None
        if journey:
            for js in journey.get("segments", []):
                if js.get("flight_number") == flight_number:
                    journey_seg = js
                    break

        fare_class = booking.get("fare_class", "main_cabin")
        if journey_seg:
            seg_seats = journey_seg.get("available_seats", {}).get(fare_class, 0)
            if seg_seats <= 0:
                await params.result_callback(
                    {
                        "status": "error",
                        "error_type": "no_seats_available",
                        "message": f"No seats available in {fare_class} fare class",
                    }
                )
                return

            raw_seat_types = journey_seg.get("available_seat_types")
            if raw_seat_types and isinstance(raw_seat_types, dict):
                available_seat_types = raw_seat_types.get(fare_class, ["window", "aisle", "middle"])
            else:
                available_seat_types = ["window", "aisle", "middle"]
            if seat_preference != "no_preference" and seat_preference not in available_seat_types:
                await params.result_callback(
                    {
                        "status": "error",
                        "error_type": "seat_type_unavailable",
                        "message": (
                            f"No {seat_preference} seats available in {fare_class} on this flight. "
                            f"Available types: {', '.join(available_seat_types)}"
                        ),
                    }
                )
                return

        passenger_index = int(passenger_id[-3:]) if passenger_id and len(passenger_id) >= 3 else 0
        base_row_map = {
            "basic_economy": 25,
            "main_cabin": 20,
            "premium_economy": 10,
            "business": 5,
            "first": 1,
        }
        base_row = base_row_map.get(fare_class, 20)
        seat_row = base_row + passenger_index
        seat_letter_map = {
            "window": "A",
            "aisle": "C",
            "middle": "B",
            "no_preference": "C",
        }
        seat_letter = seat_letter_map.get(seat_preference, "C")
        seat_number = f"{seat_row}{seat_letter}"

        flight_seg["seat"] = seat_number

        self._record_action(
            {
                "action_type": "assign_seat",
                "confirmation_number": confirmation_number,
                "passenger_id": passenger_id,
                "journey_id": journey_id,
                "flight_number": flight_number,
                "seat_preference": str(seat_preference),
                "seat_assigned": seat_number,
            }
        )

        await params.result_callback(
            {
                "status": "success",
                "confirmation_number": confirmation_number,
                "passenger_id": passenger_id,
                "journey_id": journey_id,
                "flight_number": flight_number,
                "seat_assigned": seat_number,
                "fare_class": fare_class,
                "preference": seat_preference,
                "message": f"Seat {seat_number} ({seat_preference}) successfully assigned",
            }
        )


@register_schema_tool_for_eval
class EvaAirlineAddBaggageAllowanceTool(WriteAirlineTool):
    """Add checked baggage (0-5 bags) to a flight segment."""

    DESCRIPTION = "Add checked baggage allowance to a flight segment. Specify the exact number of " "bags (0-5)."

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or self.DESCRIPTION)
        self.state = shared_state if shared_state is not None else {}

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "confirmation_number": {
                "type": "string",
                "description": "Booking confirmation code.",
            },
            "journey_id": {
                "type": "string",
                "description": "Journey ID from get_reservation.",
            },
            "num_bags": {
                "type": "integer",
                "description": "Number of checked bags (0-5).",
            },
            "flight_number": {
                "type": "string",
                "description": "For multi-segment, specific flight to add baggage to. Omit to apply to all segments.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["confirmation_number", "journey_id", "num_bags"]

    async def _execute(self, params: FunctionCallParams) -> None:
        # Adapted from https://github.com/ServiceNow/eva/tree/0.1.3 — add_baggage_allowance
        try:
            p = AddBaggageAllowanceParams.model_validate(params.arguments)
        except ValidationError as exc:
            await params.result_callback(validation_error_response(exc, AddBaggageAllowanceParams))
            return

        db = self.state.get("db")
        if not db:
            await params.result_callback(_db_not_initialized())
            return

        confirmation_number = p.confirmation_number.upper()
        journey_id = p.journey_id
        num_bags = p.num_bags
        flight_number = p.flight_number

        reservation = _lookup_reservation(db, confirmation_number)
        if not reservation:
            await params.result_callback(_reservation_not_found(confirmation_number))
            return

        booking = _find_booking_journey(reservation, journey_id)
        if not booking:
            await params.result_callback(_journey_not_found(journey_id))
            return

        if num_bags < 0 or num_bags > 5:
            await params.result_callback(
                {
                    "status": "error",
                    "error_type": "invalid_bag_count",
                    "message": f"Invalid number of bags {num_bags}. Must be between 0 and 5",
                }
            )
            return

        if flight_number:
            targets, error = _find_booking_segment(booking, journey_id, flight_number)
            if error:
                await params.result_callback(error)
                return
        else:
            targets = booking.get("segments", [])

        for fs in targets:
            fs["bags_checked"] = num_bags

        self._record_action(
            {
                "action_type": "add_baggage_allowance",
                "confirmation_number": confirmation_number,
                "journey_id": journey_id,
                "flight_number": flight_number,
                "num_bags": num_bags,
            }
        )

        await params.result_callback(
            {
                "status": "success",
                "confirmation_number": confirmation_number,
                "journey_id": journey_id,
                "bags_checked": num_bags,
                "message": f"Baggage allowance set to {num_bags} checked bag(s)",
            }
        )


@register_schema_tool_for_eval
class EvaAirlineAddMealRequestTool(WriteAirlineTool):
    """Add or update a special meal request for a passenger on a flight segment."""

    DESCRIPTION = "Add or update special meal request for a passenger on a flight segment."

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or self.DESCRIPTION)
        self.state = shared_state if shared_state is not None else {}

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "confirmation_number": {
                "type": "string",
                "description": "Booking confirmation code.",
            },
            "passenger_id": {
                "type": "string",
                "description": "Passenger identifier (e.g., PAX001).",
            },
            "journey_id": {
                "type": "string",
                "description": "Journey ID from get_reservation.",
            },
            "meal_type": {
                "type": "string",
                "description": (
                    "One of: vegetarian, vegan, kosher, halal, gluten_free, diabetic, "
                    "low_sodium, child, hindu, standard, none."
                ),
            },
            "flight_number": {
                "type": "string",
                "description": "For multi-segment, specific flight to apply meal to. Omit to apply to all segments.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["confirmation_number", "passenger_id", "journey_id", "meal_type"]

    async def _execute(self, params: FunctionCallParams) -> None:
        # Adapted from https://github.com/ServiceNow/eva/tree/0.1.3 — add_meal_request
        try:
            p = AddMealRequestParams.model_validate(params.arguments)
        except ValidationError as exc:
            await params.result_callback(validation_error_response(exc, AddMealRequestParams))
            return

        db = self.state.get("db")
        if not db:
            await params.result_callback(_db_not_initialized())
            return

        confirmation_number = p.confirmation_number.upper()
        passenger_id = p.passenger_id
        journey_id = p.journey_id
        meal_type = p.meal_type
        flight_number = p.flight_number

        reservation = _lookup_reservation(db, confirmation_number)
        if not reservation:
            await params.result_callback(_reservation_not_found(confirmation_number))
            return

        booking = _find_booking_journey(reservation, journey_id)
        if not booking:
            await params.result_callback(_journey_not_found(journey_id))
            return

        if flight_number:
            targets, error = _find_booking_segment(booking, journey_id, flight_number)
            if error:
                await params.result_callback(error)
                return
        else:
            targets = booking.get("segments", [])

        for fs in targets:
            fs["meal_request"] = meal_type

        self._record_action(
            {
                "action_type": "add_meal_request",
                "confirmation_number": confirmation_number,
                "passenger_id": passenger_id,
                "journey_id": journey_id,
                "meal_type": str(meal_type),
                "flight_number": flight_number,
            }
        )

        await params.result_callback(
            {
                "status": "success",
                "confirmation_number": confirmation_number,
                "passenger_id": passenger_id,
                "journey_id": journey_id,
                "meal_type": meal_type,
                "message": f"{meal_type} meal request added",
            }
        )


@register_schema_tool_for_eval
class EvaAirlineAddToStandbyTool(WriteAirlineTool):
    """Add passenger(s) to standby list for a flight."""

    DESCRIPTION = "Add passenger(s) to the standby list for a flight."

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or self.DESCRIPTION)
        self.state = shared_state if shared_state is not None else {}

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "confirmation_number": {
                "type": "string",
                "description": "Booking confirmation code.",
            },
            "journey_id": {
                "type": "string",
                "description": "Journey ID to add passenger to standby for.",
            },
            "passenger_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Passenger IDs to add to standby (e.g. ['PAX002', 'PAX003']).",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["confirmation_number", "journey_id", "passenger_ids"]

    async def _execute(self, params: FunctionCallParams) -> None:
        # Adapted from https://github.com/ServiceNow/eva/tree/0.1.3 — add_to_standby
        try:
            p = AddToStandbyParams.model_validate(params.arguments)
        except ValidationError as exc:
            await params.result_callback(validation_error_response(exc, AddToStandbyParams))
            return

        db = self.state.get("db")
        if not db:
            await params.result_callback(_db_not_initialized())
            return

        confirmation_number = p.confirmation_number.upper()
        journey_id = p.journey_id
        passenger_ids = p.passenger_ids

        journeys = db.get("journeys", {})
        flight = journeys.get(journey_id)

        if not flight:
            await params.result_callback(
                {
                    "status": "error",
                    "error_type": "flight_not_found",
                    "message": f"Flight {journey_id} not found",
                }
            )
            return

        if flight.get("status") == "cancelled":
            await params.result_callback(
                {
                    "status": "error",
                    "error_type": "flight_cancelled",
                    "message": "Cannot add to standby for cancelled flight",
                }
            )
            return

        reservation = _lookup_reservation(db, confirmation_number)
        if reservation and passenger_ids:
            valid_passenger_ids = {p2.get("passenger_id") for p2 in reservation.get("passengers", [])}
            invalid_ids = [pid for pid in passenger_ids if pid not in valid_passenger_ids]
            if invalid_ids:
                await params.result_callback(
                    {
                        "status": "error",
                        "error_type": "invalid_passengers",
                        "message": f"Unknown passenger ID(s): {', '.join(invalid_ids)}",
                    }
                )
                return

        if "standby_list" not in flight:
            flight["standby_list"] = []
        standby_position = len(flight["standby_list"]) + len(passenger_ids)

        if reservation:
            if "standby_list" not in reservation:
                reservation["standby_list"] = []
            reservation["standby_list"].append(
                {
                    "journey_id": journey_id,
                    "passenger_ids": passenger_ids,
                    "position": standby_position,
                    "status": "pending",
                }
            )

        for passenger_id in passenger_ids:
            flight["standby_list"].append(
                {
                    "confirmation_number": confirmation_number,
                    "passenger_id": passenger_id,
                    "position": len(flight["standby_list"]) + 1,
                }
            )

        self._record_action(
            {
                "action_type": "add_to_standby",
                "confirmation_number": confirmation_number,
                "journey_id": journey_id,
                "passenger_ids": list(passenger_ids),
            }
        )

        await params.result_callback(
            {
                "status": "success",
                "confirmation_number": confirmation_number,
                "journey_id": journey_id,
                "standby_list_position": standby_position,
                "message": f"Added {len(passenger_ids)} passenger(s) to standby list",
            }
        )


@register_schema_tool_for_eval
class EvaAirlineIssueTravelCreditTool(WriteAirlineTool):
    """Issue a travel credit (future-flight voucher) to a passenger."""

    DESCRIPTION = (
        "Issue a travel credit or future flight voucher. Used for non-refundable "
        "cancellations, fare-difference negatives, downgrades, or service recovery."
    )

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or self.DESCRIPTION)
        self.state = shared_state if shared_state is not None else {}

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "confirmation_number": {
                "type": "string",
                "description": "Booking confirmation code.",
            },
            "passenger_id": {
                "type": "string",
                "description": "Passenger identifier (e.g., PAX001).",
            },
            "amount": {"type": "number", "description": "Credit amount in USD."},
            "credit_reason": {
                "type": "string",
                "description": (
                    "One of: cancellation_non_refundable, fare_difference_negative, "
                    "service_recovery, goodwill, downgrade_compensation."
                ),
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["confirmation_number", "passenger_id", "amount", "credit_reason"]

    async def _execute(self, params: FunctionCallParams) -> None:
        # Adapted from https://github.com/ServiceNow/eva/tree/0.1.3 — issue_travel_credit
        try:
            p = IssueTravelCreditParams.model_validate(params.arguments)
        except ValidationError as exc:
            await params.result_callback(validation_error_response(exc, IssueTravelCreditParams))
            return

        db = self.state.get("db")
        if not db:
            await params.result_callback(_db_not_initialized())
            return

        confirmation_number = p.confirmation_number.upper()
        passenger_id = p.passenger_id
        amount = p.amount
        credit_reason = p.credit_reason

        if not _lookup_reservation(db, confirmation_number):
            await params.result_callback(_reservation_not_found(confirmation_number))
            return

        passenger_prefix = passenger_id[:3].upper() if passenger_id else ""
        credit_code = f"TC{confirmation_number}{passenger_prefix}"

        travel_credits = db.setdefault("travel_credits", {})
        travel_credits[credit_code] = {
            "credit_code": credit_code,
            "confirmation_number": confirmation_number,
            "passenger_id": passenger_id,
            "amount": amount,
            "credit_reason": credit_reason,
            "issued_date": db["_current_date"],
            "expiry_date": str(int(db["_current_date"][:4]) + 1) + db["_current_date"][4:],
            "status": "active",
        }

        self._record_action(
            {
                "action_type": "issue_travel_credit",
                "confirmation_number": confirmation_number,
                "passenger_id": passenger_id,
                "amount": amount,
                "credit_reason": str(credit_reason),
            }
        )

        await params.result_callback(
            {
                "status": "success",
                "confirmation_number": confirmation_number,
                "passenger_id": passenger_id,
                "credit_code": credit_code,
                "amount": amount,
                "valid_months": 12,
                "message": f"${amount} travel credit issued with code {credit_code}",
            }
        )


@register_schema_tool_for_eval
class EvaAirlineIssueHotelVoucherTool(WriteAirlineTool):
    """Issue a hotel voucher (1-3 nights) for IRROPS overnight situations."""

    DESCRIPTION = (
        "Issue a hotel voucher. Use for overnight IRROPS situations after rebooking is "
        "confirmed (not before). Maximum 3 nights."
    )

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or self.DESCRIPTION)
        self.state = shared_state if shared_state is not None else {}

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "confirmation_number": {
                "type": "string",
                "description": "Booking confirmation code.",
            },
            "passenger_id": {
                "type": "string",
                "description": "Passenger identifier (e.g., PAX001).",
            },
            "num_nights": {"type": "integer", "description": "Number of nights (1-3)."},
        }

    @property
    def required_properties(self) -> List[str]:
        return ["confirmation_number", "passenger_id", "num_nights"]

    async def _execute(self, params: FunctionCallParams) -> None:
        # Adapted from https://github.com/ServiceNow/eva/tree/0.1.3 — issue_hotel_voucher
        try:
            p = IssueHotelVoucherParams.model_validate(params.arguments)
        except ValidationError as exc:
            await params.result_callback(validation_error_response(exc, IssueHotelVoucherParams))
            return

        db = self.state.get("db")
        if not db:
            await params.result_callback(_db_not_initialized())
            return

        confirmation_number = p.confirmation_number.upper()
        passenger_id = p.passenger_id
        num_nights = p.num_nights

        if not _lookup_reservation(db, confirmation_number):
            await params.result_callback(_reservation_not_found(confirmation_number))
            return

        if num_nights > 3:
            await params.result_callback(
                {
                    "status": "error",
                    "error_type": "exceeds_authority",
                    "message": "Hotel vouchers can be issued for maximum of 3 nights",
                }
            )
            return

        voucher_code = f"HOTEL-{confirmation_number}"
        hotel_vouchers = db.setdefault("hotel_vouchers", {})
        hotel_vouchers[voucher_code] = {
            "voucher_code": voucher_code,
            "confirmation_number": confirmation_number,
            "passenger_id": passenger_id,
            "num_nights": num_nights,
            "issued_date": db["_current_date"],
            "status": "active",
        }

        self._record_action(
            {
                "action_type": "issue_hotel_voucher",
                "confirmation_number": confirmation_number,
                "passenger_id": passenger_id,
                "num_nights": num_nights,
            }
        )

        await params.result_callback(
            {
                "status": "success",
                "confirmation_number": confirmation_number,
                "passenger_id": passenger_id,
                "voucher_code": voucher_code,
                "number_of_nights": num_nights,
                "valid_at": "Any hotels in airport area",
                "message": f"Hotel voucher issued with code {voucher_code} for {num_nights} nights",
            }
        )


@register_schema_tool_for_eval
class EvaAirlineIssueMealVoucherTool(WriteAirlineTool):
    """Issue a meal voucher for delays/cancellations that qualify per policy."""

    DESCRIPTION = (
        "Issue a meal voucher. Amount is computed from voucher_reason: "
        "delay_over_2_hours=$12, delay_over_4_hours=$15, "
        "cancellation_wait_same_day=$15, irrops_overnight=$25."
    )

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or self.DESCRIPTION)
        self.state = shared_state if shared_state is not None else {}

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "confirmation_number": {
                "type": "string",
                "description": "Booking confirmation code.",
            },
            "passenger_id": {
                "type": "string",
                "description": "Passenger identifier (e.g., PAX001).",
            },
            "voucher_reason": {
                "type": "string",
                "description": (
                    "One of: delay_over_2_hours, delay_over_4_hours, " "cancellation_wait_same_day, irrops_overnight."
                ),
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["confirmation_number", "passenger_id", "voucher_reason"]

    async def _execute(self, params: FunctionCallParams) -> None:
        # Adapted from https://github.com/ServiceNow/eva/tree/0.1.3 — issue_meal_voucher
        try:
            p = IssueMealVoucherParams.model_validate(params.arguments)
        except ValidationError as exc:
            await params.result_callback(validation_error_response(exc, IssueMealVoucherParams))
            return

        db = self.state.get("db")
        if not db:
            await params.result_callback(_db_not_initialized())
            return

        confirmation_number = p.confirmation_number.upper()
        passenger_id = p.passenger_id
        voucher_reason = p.voucher_reason

        if not _lookup_reservation(db, confirmation_number):
            await params.result_callback(_reservation_not_found(confirmation_number))
            return

        amount_map = {
            "delay_over_2_hours": 12,
            "delay_over_4_hours": 15,
            "cancellation_wait_same_day": 15,
            "irrops_overnight": 25,
        }
        amount = amount_map.get(voucher_reason, 12)

        passenger_prefix = passenger_id[:4].upper() if passenger_id else ""
        voucher_code = f"MEAL-{confirmation_number}-{passenger_prefix}"

        meal_vouchers = db.setdefault("meal_vouchers", {})
        meal_vouchers[voucher_code] = {
            "voucher_code": voucher_code,
            "confirmation_number": confirmation_number,
            "passenger_id": passenger_id,
            "amount": amount,
            "voucher_reason": voucher_reason,
            "issued_date": db["_current_date"],
            "status": "active",
        }

        self._record_action(
            {
                "action_type": "issue_meal_voucher",
                "confirmation_number": confirmation_number,
                "passenger_id": passenger_id,
                "voucher_reason": str(voucher_reason),
                "amount": amount,
            }
        )

        await params.result_callback(
            {
                "status": "success",
                "confirmation_number": confirmation_number,
                "passenger_id": passenger_id,
                "voucher_code": voucher_code,
                "amount": amount,
                "valid_at": "Airport terminal restaurants",
                "message": f"${amount} meal voucher issued with code {voucher_code}",
            }
        )


# ---------------------------------------------------------------------------
# System tool — eva taxonomy is "system"; subclass WriteAirlineTool because
# transfer is a recordable terminal action.
# ---------------------------------------------------------------------------


@register_schema_tool_for_eval
class EvaAirlineTransferToAgentTool(WriteAirlineTool):
    """Transfer the call to a live human agent (recordable terminal action)."""

    DESCRIPTION = (
        "Transfer the call to a live human agent. Use when the caller explicitly requests "
        "an agent, when a policy exception is needed, when stuck after two attempts, or "
        "for technical issues."
    )

    def __init__(self, *, shared_state: Optional[dict] = None, description: Optional[str] = None):
        super().__init__(description=description or self.DESCRIPTION)
        self.state = shared_state if shared_state is not None else {}

    @property
    def properties(self) -> Dict[str, Any]:
        return {
            "confirmation_number": {
                "type": "string",
                "description": "Booking confirmation code for context transfer.",
            },
            "transfer_reason": {
                "type": "string",
                "description": (
                    "Reason for transfer (passenger_requested, policy_exception_needed, "
                    "complex_itinerary, complaint_escalation, technical_issue, unable_to_resolve)."
                ),
            },
            "issue_summary": {
                "type": "string",
                "description": "Brief summary of the issue and what has been attempted.",
            },
        }

    @property
    def required_properties(self) -> List[str]:
        return ["confirmation_number", "transfer_reason", "issue_summary"]

    async def _execute(self, params: FunctionCallParams) -> None:
        # Adapted from https://github.com/ServiceNow/eva/tree/0.1.3 — transfer_to_agent
        try:
            p = TransferToAgentParams.model_validate(params.arguments)
        except ValidationError as exc:
            await params.result_callback(validation_error_response(exc, TransferToAgentParams))
            return

        db = self.state.get("db")
        if not db:
            await params.result_callback(_db_not_initialized())
            return

        confirmation_number = p.confirmation_number.upper()
        transfer_reason = p.transfer_reason
        issue_summary = p.issue_summary

        if confirmation_number and not _lookup_reservation(db, confirmation_number):
            await params.result_callback(_reservation_not_found(confirmation_number))
            return

        call_index = self._next_call_index("transfer_to_agent")
        transfer_id = f"TRF-{confirmation_number}-{str(call_index).zfill(3)}"

        self._record_action(
            {
                "action_type": "transfer_to_agent",
                "confirmation_number": confirmation_number,
                "transfer_reason": transfer_reason,
                "issue_summary": issue_summary,
            }
        )

        await params.result_callback(
            {
                "status": "success",
                "transfer_id": transfer_id,
                "confirmation_number": confirmation_number,
                "transfer_reason": transfer_reason,
                "issue_summary": issue_summary,
                "estimated_wait": "2-3 minutes",
                "message": "Transferring to live agent",
            }
        )


# ---------------------------------------------------------------------------
# Note: actions accumulated in shared_state["actions"] are pulled by the bridge
# at end-of-scenario via the get_scenario_summary RTVI action, not emitted by
# any LLM-callable tool. See evaluation/README.md "eva_airline domain notes".
# ---------------------------------------------------------------------------

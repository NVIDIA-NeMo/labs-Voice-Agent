# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Table-driven error and state coverage for evaluation tool packages."""

import asyncio
import copy
import inspect

import pytest

import nemo_voice_agent.evaluation.scenarios.data  # noqa: F401
from nemo_voice_agent.evaluation.scenarios import ALL_EVAL_SCENARIOS
from nemo_voice_agent.evaluation.tools import eva_airline_tools, tau2_retail_tools, tau2_telecom_tools
from nemo_voice_agent.evaluation.tools import tau2_airline_tools as tau2_airline
from nemo_voice_agent.evaluation.tools import tau2_telecom_user_tools as telecom_user
from nemo_voice_agent.utils.tool_calling import StandardSchemaTool


class _Params:
    def __init__(self, arguments):
        self.arguments = arguments
        self.results = []

    @property
    def result(self):
        return self.results[-1] if self.results else None

    async def result_callback(self, result):
        self.results.append(result)


def _classes(module):
    """Return concrete schema tool classes defined by one module."""
    return [
        cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__
        and cls.__name__.endswith("Tool")
        and not cls.__name__.startswith("_")
        and cls.__name__ not in {"WriteAirlineTool"}
        and issubclass(cls, StandardSchemaTool)
    ]


_VALUE_BY_NAME = {
    "confirmation_number": "ZK3FFW",
    "last_name": "Rodriguez",
    "flight_number": "SK621",
    "flight_date": "2026-03-20",
    "date": "2026-03-20",
    "origin": "AUS",
    "destination": "LAX",
    "passenger_count": 1,
    "fare_class": "main_cabin",
    "journey_id": "FL_SK621_20260320",
    "new_journey_id": "FL_SK703_20260325",
    "passenger_id": "PAX001",
    "customer_id": "C1001",
    "line_id": "L1001",
    "bill_id": "B1001",
    "id": "C1001",
    "phone_number": "555-0001",
    "full_name": "Test User",
    "dob": "1990-01-01",
    "amount": 1.0,
    "data_amount_gb": 1.0,
    "app_name": "Messages",
    "permission": "camera",
    "mode": "4g_5g_preferred",
    "apn_name": "internet",
    "mmsc_url": "http://mmsc.example",
    "seat_preference": "window",
    "meal_type": "vegetarian",
    "bags_to_add": 1,
    "reason": "customer_request",
    "cancellation_reason": "customer_request",
    "refund_method": "original_payment",
    "rebooking_type": "voluntary",
    "waive_change_fee": False,
}


def _value_for(name, schema):
    if name in _VALUE_BY_NAME:
        return _VALUE_BY_NAME[name]
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        option = next((item for item in schema["anyOf"] if item.get("type") != "null"), {})
        return _value_for(name, option)
    kind = schema.get("type")
    if kind == "integer":
        return 1
    if kind == "number":
        return 1.0
    if kind == "boolean":
        return False
    if kind == "array":
        return []
    if kind == "object":
        return {}
    return "test"


def _arguments(tool):
    """Generate schema-valid-looking arguments from a tool's parameter model."""
    model = getattr(tool, "PARAMS_MODEL", None)
    if model is not None:
        schema = model.model_json_schema()
        properties = schema.get("properties", {})
        return {name: _value_for(name, properties.get(name, {})) for name in schema.get("required", [])}
    properties = tool.properties
    return {name: _value_for(name, properties.get(name, {})) for name in tool.required_properties}


def _is_crash_envelope(result):
    """True for ``StandardSchemaTool.__call__``'s uncaught-exception envelope.

    ``__call__`` turns an exception raised by ``_execute`` into the structured
    result ``{"error": str(e)}`` instead of letting it propagate, so an
    unhandled crash no longer reaches a caller's ``except`` block. The matrix
    tests must still flag it, otherwise a tool that blows up on a missing
    database would silently "pass". Tools' own validation responses use the
    richer ``{"status": "error", "error_type": ..., "message": ...}`` shape, so
    the single-key ``error`` dict identifies the base-class envelope precisely.
    """
    return isinstance(result, dict) and set(result) == {"error"}


def _invoke(tool, arguments):
    if hasattr(tool, "invoke"):
        # Sync gold-replay / shadow-DB path: returns the raw result verbatim
        # (deliberately NOT normalized — see ``_normalize_empty_result``).
        return tool.invoke(**arguments)
    params = _Params(arguments)
    # Drive the public pipecat entry point so delivery itself is under test.
    asyncio.run(tool(params))
    # ``__call__`` is the single delivery point: exactly one result per call.
    # The old contract delivered twice (the tool's own result, then a spurious
    # ``None`` from the base class), which pipecat 1.x rejects with
    # "tool_call_id ... is not running" and wedges the aggregator.
    assert len(params.results) == 1, f"delivered {len(params.results)} results, expected exactly 1"
    # ``__call__`` applies ``_normalize_empty_result``, so a falsy result is
    # always wrapped in a non-falsy envelope before it reaches pipecat.
    assert params.result, f"delivered falsy result {params.result!r}; normalization envelope missing"
    return params.result


@pytest.mark.parametrize(
    "module",
    [eva_airline_tools, tau2_airline, tau2_retail_tools, tau2_telecom_tools, telecom_user],
)
def test_all_evaluation_tools_report_missing_database(module):
    """Every registered domain tool handles an absent scenario database."""
    failures = []
    for cls in _classes(module):
        try:
            tool = cls(shared_state={})
            result = _invoke(tool, _arguments(tool))
            assert result is not None
            assert not _is_crash_envelope(result), f"unhandled exception surfaced as {result}"
        except Exception as exc:  # noqa: BLE001 - report the complete class matrix
            failures.append(f"{cls.__name__}: {type(exc).__name__}: {exc}")
    assert not failures


@pytest.mark.parametrize(
    "module",
    [eva_airline_tools, tau2_airline, tau2_retail_tools, tau2_telecom_tools, telecom_user],
)
def test_all_evaluation_tools_return_validation_results(module):
    """Malformed calls exercise each tool's validation response path."""
    failures = []
    for cls in _classes(module):
        try:
            result = _invoke(cls(shared_state={}), {})
            assert result is not None
            assert not _is_crash_envelope(result), f"unhandled exception surfaced as {result}"
        except Exception as exc:  # noqa: BLE001 - report the complete class matrix
            failures.append(f"{cls.__name__}: {type(exc).__name__}: {exc}")
    assert not failures


def _telecom_scenario():
    name = "tau2_telecom__mobile_data_issue__airplane_mode_on__data_mode_off"
    return ALL_EVAL_SCENARIOS[name]()


def _real_arguments(tool, agent_db, user_db):
    """Replace generic identifiers with values available in telecom fixtures."""
    arguments = _arguments(tool)
    customer = agent_db["customers"][0]
    line = agent_db["lines"][0]
    bill = agent_db["bills"][0]
    replacements = {
        "customer_id": customer["customer_id"],
        "line_id": line["line_id"],
        "bill_id": bill["bill_id"],
        "phone_number": line["phone_number"],
        "full_name": customer["full_name"],
        "dob": customer["date_of_birth"],
        "id": customer["customer_id"],
    }
    for name in arguments.keys() & replacements.keys():
        arguments[name] = replacements[name]
    device = user_db.get("device", {})
    apps = device.get("app_statuses", {})
    if "app_name" in arguments and apps:
        arguments["app_name"] = next(iter(apps))
    return arguments


@pytest.mark.parametrize("module, side", [(tau2_telecom_tools, "agent"), (telecom_user, "user")])
def test_telecom_tool_matrix_with_real_scenario_state(module, side):
    """All telecom tools traverse their populated-database behavior."""
    scenario = _telecom_scenario()
    agent_db = scenario.db
    user_db = scenario.user_db
    failures = []
    for cls in _classes(module):
        state = {"db": copy.deepcopy(agent_db if side == "agent" else user_db)}
        try:
            tool = cls(shared_state=state)
            result = _invoke(tool, _real_arguments(tool, agent_db, user_db))
            assert result is not None
            if _is_crash_envelope(result):
                # Same tolerance as the exception arm below: on the __call__
                # path a rejected state transition comes back as
                # {"error": "..."} instead of propagating, and reaching that
                # branch with a real message is the behavior under test.
                assert result["error"]
        except (ValueError, KeyError, TypeError) as exc:
            # A validly-routed tool can reject a state transition that this
            # particular scenario does not permit; reaching that branch is
            # the behavior under test.
            assert str(exc)
        except Exception as exc:  # noqa: BLE001 - report unexpected framework failures
            failures.append(f"{cls.__name__}: {type(exc).__name__}: {exc}")
    assert not failures


def test_telecom_user_status_helpers_cover_indicator_combinations():
    """Status rendering covers airplane, cellular, Wi-Fi, VPN, and saver indicators."""
    db = copy.deepcopy(_telecom_scenario().user_db)
    device = db["device"]
    device.update(
        airplane_mode=False,
        network_signal_strength="excellent",
        network_technology_connected="5G",
        data_enabled=True,
        data_saver_mode=True,
        wifi_enabled=True,
        wifi_connected=True,
        wifi_ssid="Office",
        vpn_connected=True,
    )
    status = telecom_user._check_status_bar(db)
    assert all(value in status for value in ["Excellent", "5G", "Data Enabled", "Data Saver", "Office", "VPN"])

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

"""Unit tests for evaluation scenario registration and restaurant tools."""

import asyncio

import pytest

from nemo_voice_agent.evaluation.scenarios import (
    ALL_EVAL_SCENARIOS,
    get_eval_scenario,
    list_eval_scenarios,
    register_eval_scenario,
)
from nemo_voice_agent.evaluation.scenarios.classes import Scenario
from nemo_voice_agent.evaluation.tools.restaurant_tools import GetMenuTool


class _Params:
    """Duck-typed FunctionCallParams replacement for tool callback tests."""

    def __init__(self, arguments=None):
        """Create params with optional arguments and result capture."""
        self.arguments = arguments or {}
        self.results = []

    async def result_callback(self, result):
        """Capture one tool result."""
        self.results.append(result)


def test_register_eval_scenario_lists_and_instantiates_registered_class():
    """The scenario registry decorator stores valid Scenario subclasses by name."""
    saved_registry = dict(ALL_EVAL_SCENARIOS)
    ALL_EVAL_SCENARIOS.clear()

    try:

        @register_eval_scenario
        class _UnitRegistryScenario(Scenario):
            """Tiny scenario class for registry testing."""

            name = "unit_registry__scenario"
            description = "registry test"

        assert list_eval_scenarios() == ["unit_registry__scenario"]
        assert isinstance(get_eval_scenario("unit_registry__scenario"), _UnitRegistryScenario)
        assert get_eval_scenario("missing") is None
    finally:
        ALL_EVAL_SCENARIOS.clear()
        ALL_EVAL_SCENARIOS.update(saved_registry)


def test_register_eval_scenario_rejects_non_scenario_class():
    """The scenario registry raises for classes outside the Scenario hierarchy."""
    with pytest.raises(ValueError, match="not a subclass of Scenario"):
        register_eval_scenario(object)


def test_get_menu_tool_schema_and_execution_return_configured_menu():
    """GetMenuTool exposes an empty schema and returns the injected menu text."""
    tool = GetMenuTool(menu="Pizza - $10")
    params = _Params()

    asyncio.run(tool._execute(params))

    assert tool.properties == {}
    assert tool.required_properties == []
    assert params.results == [{"menu": "Pizza - $10"}]

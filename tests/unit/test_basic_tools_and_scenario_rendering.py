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

"""Unit tests for basic evaluation tools and Scenario prompt/rendering helpers."""

import asyncio
import json

import pytest

from nemo_voice_agent.evaluation.scenarios.classes import Actions, Persona, Resources, Scenario, SuccessSignal, Task
from nemo_voice_agent.evaluation.tools.basic_tools import (
    GetCityWeatherTool,
    PlaceOrderTool,
    ReadFileTool,
    SaveQuestionAnswerTool,
)


class _CapturingRTVI:
    """Fake RTVI transport used to inspect messages emitted by tools."""

    def __init__(self):
        """Initialize an empty message list."""
        self.messages = []

    async def push_transport_message(self, message, exclude_none=True):
        """Record the message passed to the RTVI processor."""
        self.messages.append(message)


class _Params:
    """Duck-typed FunctionCallParams replacement for basic tool tests."""

    def __init__(self, arguments=None):
        """Create params with arguments and an empty result list."""
        self.arguments = arguments or {}
        self.results = []

    async def result_callback(self, result):
        """Capture one tool result."""
        self.results.append(result)


def test_get_city_weather_tool_returns_deterministic_fixture_weather():
    """The eval weather tool returns canned sunny weather without external API calls."""
    tool = GetCityWeatherTool()
    params = _Params({"city_name": "Paris"})

    asyncio.run(tool._execute(params))

    assert params.results == [
        {
            "city": "Paris",
            "weather": "sunny",
            "temperature": "20 degrees Celsius",
            "uv_index": "low",
        }
    ]


def test_read_file_tool_returns_content_and_error_payload(tmp_path):
    """ReadFileTool returns file contents and reports missing-file errors via callback."""
    file_path = tmp_path / "note.txt"
    file_path.write_text("hello")
    tool = ReadFileTool()
    ok_params = _Params({"file_path": str(file_path)})
    missing_params = _Params({"file_path": str(tmp_path / "missing.txt")})

    asyncio.run(tool._execute(ok_params))
    asyncio.run(tool._execute(missing_params))

    assert ok_params.results == [{"file_path": str(file_path), "content": "hello"}]
    assert "error" in missing_params.results[0]


def test_place_order_tool_validates_items_and_emits_summary_message():
    """PlaceOrderTool validates item totals and sends a tagged scenario summary."""
    rtvi = _CapturingRTVI()
    tool = PlaceOrderTool(rtvi=rtvi, auto_validate=True, valid_item_names=["pizza"])
    params = _Params(
        {
            "items": [{"name": "pizza", "unit_price": "7.50", "quantity": "2"}],
            "customer_name": "Ada",
            "customer_phone": "123-456-7890",
            "total_price": "15.00",
        }
    )

    asyncio.run(tool._execute(params))

    assert params.results[0]["success"] is True
    assert json.loads(params.results[0]["order_details"])["customer_name"] == "Ada"
    assert rtvi.messages[0].data.text.startswith("<final_response>")
    assert "pizza" in rtvi.messages[0].data.text


@pytest.mark.parametrize(
    "item, total_price, match",
    [
        ({}, "0.00", "must have a `name` key"),
        ({"name": "salad", "unit_price": "1.00", "quantity": "1"}, "1.00", "not on the menu"),
        ({"name": "pizza", "unit_price": "-1.00", "quantity": "1"}, "-1.00", "unit price"),
        ({"name": "pizza", "unit_price": "1.00", "quantity": "-1"}, "-1.00", "quantity"),
        ({"name": "pizza", "unit_price": "1.00", "quantity": "1"}, "2.00", "total price"),
    ],
)
def test_place_order_tool_rejects_invalid_orders(item, total_price, match):
    """Invalid order fields raise clear ValueError messages before callback emission."""
    tool = PlaceOrderTool(rtvi=_CapturingRTVI(), auto_validate=True, valid_item_names=["pizza"])
    params = _Params({"items": [item], "customer_name": "Ada", "total_price": total_price})

    with pytest.raises(ValueError, match=match):
        asyncio.run(tool._execute(params))


def test_save_question_answer_tool_sends_json_summary():
    """SaveQuestionAnswerTool serializes the QA pair into a scenario-summary message."""
    rtvi = _CapturingRTVI()
    tool = SaveQuestionAnswerTool(rtvi=rtvi)
    params = _Params({"question": "Q?", "answer": "A."})

    asyncio.run(tool._execute(params))

    assert params.results == [{"success": True, "message": "Question and answer logged."}]
    text = rtvi.messages[0].data.text
    assert text.startswith("<final_response>")
    assert json.loads(text.removeprefix("<final_response>").removesuffix("</final_response>")) == {
        "question": "Q?",
        "answer": "A.",
    }


class _RenderableScenario(Scenario):
    """Small concrete scenario used to exercise prompt rendering and save()."""

    name = "render__scenario"
    domain = "render"
    success_signals = (SuccessSignal.CLEAN_EXIT,)

    @property
    def user_persona(self):
        """Return a user persona with language/accent fields."""
        return Persona(
            role="customer",
            name="Casey",
            background="You need help.",
            personality="Brief.",
            language="English",
            accent="Canadian",
        )

    @property
    def agent_persona(self):
        """Return an agent persona with only an accent field."""
        return Persona(role="assistant", name="Alex", accent="Australian")

    @property
    def user_task(self):
        """Return a user task with background and goal."""
        return Task(background="You are at home.", goal="Ask for help.")

    @property
    def agent_task(self):
        """Return an agent task with a goal."""
        return Task(goal="Help the user.")

    @property
    def user_actions(self):
        """Return instructions and guidelines for rendering."""
        return Actions(instructions=["Say hello", "Ask"], guidelines=["Stay concise"])

    @property
    def agent_actions(self):
        """Return agent instructions only."""
        return Actions(instructions=["Greet", "Answer"])

    @property
    def user_resources(self):
        """Return user resources with tools, docs, information, and sections."""
        return Resources(
            tools={"ReadFileTool": {"file_path": "doc.txt"}},
            documents={"doc": "doc.txt"},
            information=["Use the account number."],
            info_sections={"Known": "Known fact", "Empty": ""},
        )

    @property
    def agent_resources(self):
        """Return empty agent resources."""
        return Resources()


def test_scenario_prompt_helpers_render_sections_and_tools():
    """Scenario prompt helpers combine persona, task, actions, resources, and tool JSON."""
    scenario = _RenderableScenario(reference_answer={"message": "done"}, max_duration=3)

    user_prompt = scenario.get_user_prompt()
    agent_prompt = scenario.get_agent_prompt()

    assert "Your name is Casey." in user_prompt
    assert "You speak English with a Canadian accent." in user_prompt
    assert "Step 1: Say hello" in user_prompt
    assert "### Known\nKnown fact" in user_prompt
    assert json.loads(scenario.get_user_tools()) == {"ReadFileTool": {"file_path": "doc.txt"}}
    assert "You speak with a Australian accent." in agent_prompt
    assert scenario.sync_state({}, {}) == {"agent": {}, "user": {}}


def test_scenario_save_writes_prompts_metadata_tools_and_reference(tmp_path):
    """Scenario.save writes all reviewable scenario config files to the output directory."""
    scenario = _RenderableScenario(
        reference_answer="finished",
        max_duration=3,
        clean_text=True,
        disallow_extra_items=True,
        nl_assertions=["Mention the answer."],
        db_state_assertions=[{"func_name": "check", "arguments": {}, "assert_value": True}],
        initialization_actions=[{"side": "agent", "func_name": "seed", "arguments": {}}],
        expected_scenario_db={"db": 1},
        expected_user_db={"user": 2},
    )

    scenario.save(tmp_path)

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    reference = json.loads((tmp_path / "reference_answer.json").read_text())

    assert (tmp_path / "user_prompt.txt").exists()
    assert (tmp_path / "agent_prompt.txt").exists()
    assert json.loads(json.loads((tmp_path / "user_tools.json").read_text())) == {
        "ReadFileTool": {"file_path": "doc.txt"}
    }
    assert json.loads(json.loads((tmp_path / "agent_tools.json").read_text())) == {}
    assert metadata["name"] == "render__scenario"
    assert metadata["clean_text"] is True
    assert metadata["disallow_extra_items"] is True
    assert metadata["success_signals"] == ["clean_exit"]
    assert metadata["nl_assertions"] == ["Mention the answer."]
    assert "expected_db_hash" in metadata
    assert "expected_user_db_hash" in metadata
    assert reference == {"message": "finished"}

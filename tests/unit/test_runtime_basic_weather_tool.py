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

"""Unit tests for runtime basic direct tools that normally call external services."""

import asyncio
from types import SimpleNamespace

from nemo_voice_agent.utils.tool_calling import basic_tools


class _FakeLLM:
    """Fake LLM service that captures frames pushed by the weather tool."""

    def __init__(self):
        """Initialize an empty frame capture."""
        self.frames = []

    async def push_frame(self, frame, direction=None):
        """Record one frame and optional direction."""
        self.frames.append((frame, direction))


class _Params:
    """Duck-typed FunctionCallParams replacement for direct weather tool tests."""

    def __init__(self):
        """Create params with fake LLM and result capture."""
        self.llm = _FakeLLM()
        self.results = []

    async def result_callback(self, result):
        """Capture one tool result."""
        self.results.append(result)


class _FakeWeatherClient:
    """Async context manager that returns a deterministic weather forecast."""

    def __init__(self, *args, **kwargs):
        """Accept the same constructor shape as python_weather.Client."""

    async def __aenter__(self):
        """Return this fake client from the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        """Exit the async context manager without suppressing exceptions."""
        return False

    async def get(self, city_name):
        """Return a fake forecast for the requested city."""
        return SimpleNamespace(
            description="Sunny",
            temperature=22,
            wind_speed=8,
            wind_direction=SimpleNamespace(name="NORTH"),
            precipitation=0,
            humidity=55,
            visibility=10,
            ultraviolet=3,
        )


def test_tool_get_city_weather_returns_forecast_and_pushes_progress_frames(monkeypatch):
    """The runtime weather tool emits progress frames and returns normalized forecast fields."""
    monkeypatch.setattr(basic_tools.python_weather, "Client", _FakeWeatherClient)
    params = _Params()

    asyncio.run(basic_tools.tool_get_city_weather(params, "Paris"))

    assert len(params.llm.frames) == 2
    assert params.results == [
        {
            "city": "Paris",
            "description": "Sunny",
            "temperature": "22 degrees Celsius",
            "wind_speed": "8 kilometers per hour",
            "wind_direction": "NORTH",
            "precipitation": "0 millimeters",
            "humidity": "55 percent",
            "visibility": "10 kilometers",
            "uv_index": "3",
        }
    ]


def test_tool_get_city_weather_reports_timeout(monkeypatch):
    """A weather API timeout is converted into a structured error result."""

    async def _raise_timeout(coro, timeout):
        """Raise the timeout branch used by the tool."""
        await coro
        raise asyncio.TimeoutError

    monkeypatch.setattr(basic_tools.python_weather, "Client", _FakeWeatherClient)
    monkeypatch.setattr(basic_tools.asyncio, "wait_for", _raise_timeout)
    params = _Params()

    asyncio.run(basic_tools.tool_get_city_weather(params, "Berlin"))

    assert params.results == [{"error": "python_weather API request timed out after 10.0 seconds for `Berlin`"}]


def test_tool_get_city_weather_reports_client_errors(monkeypatch):
    """Unexpected weather client errors are returned through result_callback."""

    async def _raise_error(coro, timeout):
        """Raise a deterministic client error after closing the coroutine."""
        await coro
        raise ValueError("service unavailable")

    monkeypatch.setattr(basic_tools.python_weather, "Client", _FakeWeatherClient)
    monkeypatch.setattr(basic_tools.asyncio, "wait_for", _raise_error)
    params = _Params()

    asyncio.run(basic_tools.tool_get_city_weather(params, "Tokyo"))

    assert params.results == [{"error": "Error fetching weather forecast for `Tokyo`: service unavailable"}]

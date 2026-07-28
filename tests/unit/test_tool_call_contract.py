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

"""The StandardSchemaTool contract: exactly one result per tool call.

A tool that delivers its result twice used to be invisible — pipecat <1.0
ignored the duplicate. Pipecat 1.x tracks in-flight tool calls and rejects the
second one ("tool_call_id ... is not running"), which strands the aggregator's
deferred context push so the LLM is never re-invoked with the tool output. The
agent goes quiet with no error.

These tests pin the contract that makes that impossible: ``_execute`` is pure
(arguments in, result out) and ``__call__`` is the single delivery point.
"""

import ast
import asyncio
import pathlib

import pytest

from nemo_voice_agent.utils.tool_calling.base import StandardSchemaTool


class _RecordingParams:
    """Minimal FunctionCallParams stand-in that counts deliveries."""

    def __init__(self, arguments=None):
        self.arguments = arguments or {}
        self.delivered = []
        self.function_name = "test_tool"
        self.context = None
        self.llm = None

    async def result_callback(self, result, **kwargs):
        self.delivered.append(result)


class _Tool(StandardSchemaTool):
    """Concrete tool used to exercise the base contract."""

    def __init__(self, *, behavior=None, **kwargs):
        super().__init__(description="test tool", **kwargs)
        self._behavior = behavior or (lambda **kw: {"ok": True, "got": kw})

    @property
    def properties(self):
        return {"value": {"type": "string", "description": "anything"}}

    @property
    def required_properties(self):
        return []

    async def _execute(self, **kwargs):
        return self._behavior(**kwargs)


def _run(tool, arguments=None):
    params = _RecordingParams(arguments)
    asyncio.run(tool(params))
    return params


def test_result_is_delivered_exactly_once():
    """The core invariant. A second delivery is what wedges pipecat 1.x."""
    params = _run(_Tool(), {"value": "x"})

    assert len(params.delivered) == 1
    assert params.delivered[0] == {"ok": True, "got": {"value": "x"}}


def test_arguments_are_passed_as_keywords_not_params():
    """_execute receives plain call arguments, never the framework object."""
    seen = {}

    def behavior(**kw):
        seen.update(kw)
        return {"ok": True}

    _run(_Tool(behavior=behavior), {"a": 1, "b": "two"})

    assert seen == {"a": 1, "b": "two"}


def test_exception_becomes_a_single_structured_error():
    """A raising tool still delivers once, so the aggregator never hangs."""

    def boom(**_kw):
        raise RuntimeError("kaboom")

    params = _run(_Tool(behavior=boom), {})

    assert len(params.delivered) == 1
    assert params.delivered[0] == {"error": "kaboom"}


@pytest.mark.parametrize(
    "empty, expected_key",
    [([], "results"), ({}, "results"), (None, "result"), ("", "result")],
)
def test_empty_results_are_normalized_centrally(empty, expected_key):
    """The empty-result guard is applied by __call__, so no tool can forget it.

    Pipecat rewrites a falsy result to the literal "COMPLETED", which the LLM
    reads as success — the bug behind an agent believing a lookup that matched
    nobody had succeeded.
    """
    params = _run(_Tool(behavior=lambda **_kw: empty), {})

    assert len(params.delivered) == 1
    delivered = params.delivered[0]
    assert delivered, "result must not be falsy or pipecat masks it as COMPLETED"
    assert expected_key in delivered


def test_after_result_hook_runs_after_delivery():
    """Post-delivery side effects (e.g. the exit message) keep their ordering."""
    order = []

    class _WithHook(_Tool):
        async def _execute(self, **kwargs):
            order.append("execute")
            return {"ok": True}

        async def _after_result(self, params):
            order.append("after_result")

    params = _RecordingParams({})

    async def _record(result, **kwargs):
        order.append("deliver")
        params.delivered.append(result)

    params.result_callback = _record
    asyncio.run(_WithHook()(params))

    assert order == ["execute", "deliver", "after_result"]
    assert len(params.delivered) == 1


def test_no_execute_implementation_touches_the_framework():
    """Structural guard over the whole package.

    This is the check that would have caught the original bug: every _execute
    must be pure. If one takes `params` or calls `result_callback`, it is
    almost certainly delivering a result that __call__ will then deliver again.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    violations = []

    for path in (repo_root / "nemo_voice_agent").rglob("*.py"):
        source = path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.AsyncFunctionDef) and node.name == "_execute"):
                continue
            rel = f"{path.relative_to(repo_root)}:{node.lineno}"
            if "params" in [arg.arg for arg in node.args.args]:
                violations.append(f"{rel} takes `params`")
            segment = ast.get_source_segment(source, node) or ""
            if "result_callback" in segment:
                violations.append(f"{rel} delivers its own result")

    assert not violations, "_execute must be pure (arguments in, result out):\n" + "\n".join(violations)


def test_error_message_containing_braces_is_not_mangled_by_the_logger():
    """A tool error whose text contains `{}` must still be delivered.

    loguru treats extra kwargs as `str.format()` arguments, so the stdlib
    idiom `logger.error(msg, exc_info=True)` raises IndexError from inside the
    error handler whenever the message has braces — masking the real failure.
    Real tools hit this: PlaceOrderTool reports "the item is: {}."
    """

    def boom(**_kw):
        raise ValueError("Each item must have a `name` key, but the item is: {}.")

    params = _run(_Tool(behavior=boom), {})

    assert len(params.delivered) == 1
    assert params.delivered[0]["error"].endswith("the item is: {}.")

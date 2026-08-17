{/*
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/}

# Writing Your Own Tools

NeMo Labs Voice Agent gives you three ways to expose a Python callable to the LLM. Read
[Tool Calling](./tool-calling.md) first for backend requirements and the config flags that gate registration.

| Mechanism | Where the schema comes from | Register with | Use it when |
| --- | --- | --- | --- |
| Direct function | Inferred from the Python signature + docstring | `register_direct_tools_to_llm(tools=[...])` | The tool is a standalone function with simple, annotatable arguments. |
| Component-owned tool | Same inference, but the functions are methods on a pipeline component | `ToolCallingMixin` + `register_direct_tools_to_llm(tool_mixins=[...])` | The tool must mutate a service's state (TTS speed, ASR language, turn-taking behavior). |
| Schema tool | An explicit `FunctionSchema` you declare | `register_schema_tools_to_llm(...)` | You need control over the JSON Schema, or you are writing an eval-domain tool. |

The first two paths both hand pipecat a `DirectFunction`, so they share one contract.

## The direct-function contract

`pipecat/adapters/schemas/direct_function.py` validates and introspects every direct function at
registration time; violating the first two rules raises at startup, not at call time.

1. The function must be `async`.
2. Its first parameter must be named exactly `params` (it receives a `FunctionCallParams`).
3. It must deliver its result exactly once, by awaiting `params.result_callback(...)`.

Everything the LLM sees is derived automatically:

| Source in your code | What the LLM receives |
| --- | --- |
| The Python function name | The tool name. There is no way to override it on this path. |
| The docstring summary/body | The tool description. |
| Each `Args:` entry | That parameter's `description`. |
| Each type annotation | That parameter's JSON Schema `type` (unannotated parameters get an empty schema). |
| A parameter with no default | An entry in the schema's `required` list. |
| The `params` parameter | Nothing — it is skipped. |

Because the docstring *is* the prompt, write it for the model: say when the tool should be called, when it
should not, and what the agent should do with the result.

## 1. Direct functions

The shipped example is `tool_get_city_weather` in `nemo_voice_agent/utils/tool_calling/basic_tools.py`. It
pushes a `TTSSpeakFrame` filler ("Looking up weather data for ... Please wait a moment.") before its network
call so the user is not left in silence, wraps the HTTP call in its own `asyncio.wait_for`, and returns
`{"error": ...}` through `result_callback` on timeout or failure.

A minimal tool of your own:

```python
# my_tools.py
from pipecat.services.llm_service import FunctionCallParams


async def tool_get_store_hours(params: FunctionCallParams, store_id: str, day: str = "today"):
    """Look up the opening hours of a retail store.

    Call this whenever the user asks when a specific store opens or closes.
    Do not call it for general questions about the company.

    Args:
        store_id: The store identifier, for example "SF-014".
        day: Day of the week, or "today". Defaults to "today".
    """
    await params.result_callback({"store_id": store_id, "day": day, "opens": "09:00", "closes": "21:00"})
```

Wire it into the pipeline by adding it to the existing `register_direct_tools_to_llm` call in
`examples/generic_voice_agent/server/server.py` (guarded by `llm.enable_tool_calling`):

```python
from my_tools import tool_get_store_hours
from nemo_voice_agent.utils.tool_calling.mixins import register_direct_tools_to_llm

register_direct_tools_to_llm(
    llm=llm,
    context=context,
    tool_mixins=[tts],
    tools=[tool_get_city_weather, tool_get_store_hours],
)
```

The full signature (keyword-only, from `nemo_voice_agent/utils/tool_calling/mixins.py`):

```python
def register_direct_tools_to_llm(
    *,
    llm: OpenAILLMService,
    context: LLMContext,
    tool_mixins: list[ToolCallingMixin] = [],
    tools: list[DirectFunction] = [],
    cancel_on_interruption: bool = True,
) -> None: ...
```

Tools already present on the context are preserved — the helper appends to `context.tools` before calling
`context.set_tools(...)`, so calling it twice accumulates rather than replaces.

## 2. Component-owned tools

When a tool has to reach into a live pipeline component, put it on the component. Mix
`ToolCallingMixin` (`nemo_voice_agent/utils/tool_calling/mixins.py`) into the service and implement
`setup_tool_calling`, which registers each bound method:

```python
from pipecat.services.llm_service import FunctionCallParams

from nemo_voice_agent.utils.tool_calling.mixins import ToolCallingMixin


class MyService(SomeBaseService, ToolCallingMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setup_tool_calling()  # must be called from __init__

    def setup_tool_calling(self):
        self.register_direct_function("tool_set_level", self.tool_set_level)

    async def tool_set_level(self, params: FunctionCallParams, level: int):
        """Set the processing level of the service.

        Args:
            level: An integer between 0 and 10.
        """
        self._level = level
        await params.result_callback({"success": True, "message": f"Level set to {level}"})
```

The mixin surface is three members:

| Member | Behavior |
| --- | --- |
| `setup_tool_calling(self)` | You implement it. The base raises `NotImplementedError`, so a mixed-in class that registers nothing must still override it — `BaseNemoTTSService` does so with a `pass` body, which is why the Magpie and FastPitch subclasses can stay tool-free. |
| `register_direct_function(self, function_name, function)` | Stores the callable in `self.direct_functions`, creating the dict on first use. |
| `available_tools` | Property returning `dict[str, DirectFunction]`; this is what `register_direct_tools_to_llm` iterates over. |

**Gotcha:** the `function_name` you pass to `register_direct_function` is only a bookkeeping key.
`register_direct_tools_to_llm` appends the *callable*, and pipecat names the tool after the Python method,
so a mismatched key silently has no effect on what the model sees. Keep them identical.

The canonical example is `KokoroTTSService` in `nemo_voice_agent/pipecat/services/nemo/tts.py`, which
registers six voice-control methods (`tool_tts_set_speed`, `tool_tts_reset_speed`, `tool_tts_speak_faster`,
`tool_tts_speak_slower`, `tool_tts_set_voice`, `tool_tts_reset_voice`). `BaseNemoTTSService` calls
`setup_tool_calling()` from its constructor, so a subclass only overrides the method. Two patterns there
worth copying: `tool_tts_set_voice` pushes an `LLMTextFrame("Just a moment.")` before reloading the model,
and runs the blocking reload through `asyncio.to_thread` so the event loop keeps serving audio.

Pass the instance in `tool_mixins`. Anything in that list that is not a `ToolCallingMixin` is skipped with a
warning — which is why the example server can pass `tool_mixins=[tts]` unconditionally even when `tts.type`
resolves to the hosted NVIDIA service.

## 3. Schema tools

`StandardSchemaTool` in `nemo_voice_agent/utils/tool_calling/base.py` is the explicit-schema path. Subclass
it and implement three members; the base builds the `FunctionSchema` and owns delivery:

```python
class StandardSchemaTool:
    def __init__(self, *, description: Optional[str] = None, name: Optional[str] = None): ...

    @property
    def properties(self) -> Dict[str, Any]: ...

    @property
    def required_properties(self) -> List[str]: ...

    async def _execute(self, **kwargs: Any) -> Any: ...

    async def _after_result(self, params: FunctionCallParams) -> None: ...
```

Key differences from the direct path:

- The tool name comes from an optional class-level `name` attribute, falling back to the class name — so
  snake_case tool names are possible here.
- `_execute` is **pure**: it takes the call arguments as keyword arguments and *returns* a result. It must
  not touch `params` and must not call `result_callback`. `__call__` delivers exactly once, and converts a
  raised exception into `{"error": ...}`. `tests/unit/test_tool_call_contract.py` pins this.
- Falsy results are wrapped before delivery, because pipecat rewrites any falsy result to the literal string
  `"COMPLETED"` — which a model reads as success. A bare `[]` from a lookup that matched nothing becomes an
  explicit "No matching records found." envelope instead.
- `register_schema_tools_to_llm` also installs a catch-all handler for unregistered tool names that returns
  the list of tools the model can actually see.

Register with `register_schema_tools_to_llm(llm, context, tools, cancel_on_interruption=True,
keep_existing_tools=True, register_unknown_tool_handler=True)`. This is the path the evaluation domains use —
see [Authoring Evaluation Tools](../evaluate/authoring-tools.md).

## Interruption and long-running tools

`cancel_on_interruption` defaults to `True` on both registration helpers, and that is what you want for an
ordinary tool. Setting it to `False` does more than survive a barge-in: in pipecat 1.x it also marks the tool
**asynchronous**, so the LLM does not wait for the result. The model immediately receives a
`{"status": "running"}` placeholder and the real payload is injected later as a `developer` message. Only opt
in for a genuinely fire-and-forget tool. For bounded waits, use `llm.function_call_timeout_secs`
(default `10.0`) — see [Tool Calling](./tool-calling.md).

## Testing your tool

Both paths are testable without a GPU or a running LLM: build a duck-typed stand-in for
`FunctionCallParams` that records `result_callback` calls, then await the tool.
`tests/unit/test_runtime_basic_weather_tool.py` does this for the weather tool;
`tests/unit/test_tool_call_contract.py` does it for the schema-tool contract.

```bash
uv run pytest tests/unit/test_runtime_basic_weather_tool.py tests/unit/test_tool_call_contract.py -m "not gpu"
```

Two repo-wide requirements before committing: every `.py` file other than `__init__.py` needs the
SPDX/Apache header in its first 10 lines (CI hard-fails without it), and ruff is the only formatter and
linter — run `uv run ruff format my_tools.py` and `uv run ruff check --fix my_tools.py`.

## Next steps

- [Tool Calling](./tool-calling.md) — backend support, parser flags, and the shipped demo tools.
- [Authoring Evaluation Tools](../evaluate/authoring-tools.md) — the schema-tool path in depth.
- [Prompts](../configure/prompts.md) — tuning `system_prompt_suffix` so the model reaches for tools at the right time.
- [Server Configuration](../configure/server-config.md) — where `enable_tool_calling` lives and how layering works.

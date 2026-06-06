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

"""Factory helpers for the common RTVI actions used by voice-agent bots.

The actions are parameterized so the same factory works for bots with different
pipeline shapes: pass in whichever aggregators, services, and handlers the bot
actually has. ``None`` entries in ``resettable_services`` are silently skipped.

The reset and update-prompt actions need to queue an ``EndTaskFrame`` onto a
``PipelineTask`` that is typically created *after* the RTVI processor (because
the task needs ``rtvi`` in its observer list). ``TaskRef`` is a tiny holder the
bot sets after constructing the task.
"""

import asyncio
import copy
import dataclasses
import json
from typing import Any, Callable, List, Optional

from loguru import logger
from pipecat.frames.frames import EndTaskFrame
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frameworks.rtvi import RTVIAction, RTVIProcessor
from pipecat.services.ai_service import AIService


@dataclasses.dataclass
class TaskRef:
    """Mutable handle to a PipelineTask and its running flag.

    Construct early, hand to RTVI action factories, then populate once the task
    exists. ``running`` is flipped by the bot runner during shutdown so handlers
    can avoid queueing frames onto a dead task.
    """

    task: Optional[PipelineTask] = None
    running: bool = False


@dataclasses.dataclass
class SharedStateRef:
    """Mutable handle to the per-scenario ``shared_state`` dict.

    The same dict that's passed to tool constructors is also published here, so
    other RTVI action handlers (specifically ``get_scenario_summary``) can read
    ``shared_state["actions"]`` and ``shared_state["db"]`` without needing tool
    references. ``state`` is reset (re-pointed at a new dict) every time
    ``update_system_prompt`` runs.
    """

    state: dict = dataclasses.field(default_factory=dict)


async def _maybe_end_task(task_ref: TaskRef) -> None:
    if task_ref is not None and task_ref.running:
        await task_ref.task.queue_frames([EndTaskFrame()])


def _reset_services(services: List[AIService]) -> None:
    for service in services:
        if service is not None and hasattr(service, "reset"):
            service.reset()


def create_reset_context_action(
    task_ref: TaskRef,
    user_aggregator,
    assistant_aggregator,
    original_messages: List[dict],
    resettable_services: List[AIService],
) -> RTVIAction:
    """Build the ``context.reset`` action.

    ``original_messages`` is captured by reference so the action always resets to
    whatever ``update_system_prompt`` last wrote.
    """

    async def handler(rtvi_processor: RTVIProcessor, service: str, arguments: dict[str, Any]) -> bool:
        logger.info("Resetting conversation context...")
        try:
            await _maybe_end_task(task_ref)
            user_aggregator.reset()
            assistant_aggregator.reset()
            user_aggregator.set_messages(copy.deepcopy(original_messages))
            assistant_aggregator.set_messages(copy.deepcopy(original_messages))
            _reset_services(resettable_services)
            logger.info("Conversation context reset successfully")
            return True
        except Exception as e:
            logger.error(f"Error resetting context: {e}")
            return False

    return RTVIAction(
        service="context",
        action="reset",
        result="bool",
        arguments=[],
        handler=handler,
    )


def create_update_system_prompt_action(
    task_ref: TaskRef,
    user_aggregator,
    assistant_aggregator,
    original_messages: List[dict],
    resettable_services: List[Any],
    *,
    system_role: str,
    system_prompt_suffix: str,
    enable_tool_calling: bool = False,
    llm=None,
    context=None,
    rtvi: Optional[RTVIProcessor] = None,
    tool_factory: Optional[Callable[..., Any]] = None,
    register_schema_tools: Optional[Callable[..., Any]] = None,
    shared_state_ref: Optional[SharedStateRef] = None,
) -> RTVIAction:
    """Build the ``context.update_system_prompt`` action.

    Tool registration is optional. When ``enable_tool_calling`` is True and a
    ``tools`` JSON string is supplied by the caller, ``tool_factory`` is invoked
    per tool to produce schema tools, then ``register_schema_tools`` swaps them
    onto ``llm`` / ``context``. This keeps the factory decoupled from
    evaluation-specific tool registries.

    The action accepts an optional ``shared_state_init`` argument (JSON string)
    used to initialize the per-scenario ``shared_state`` dict before tools are
    instantiated. The bridge populates it from ``Scenario.setup_shared_state``.
    Two supported shapes (both via ``shared_state_init``):
      - **Inline**: ``{"db": {...full content...}, ...}``. Used as-is.
      - **Path-based fallback**: ``{"db_path": "rel/path.json", ...}``. Resolved
        against ``EVAL_DATA_ROOT`` and replaced under the de-suffixed key
        (``db_path`` → ``db``). Missing files raise ``FileNotFoundError`` loudly.

    If ``shared_state_ref`` is provided, the resolved ``shared_state`` is
    published to it so other action handlers (``get_scenario_summary``) can
    read the same dict. Only consumed when tool calling is enabled.
    """

    async def handler(rtvi_processor: RTVIProcessor, service: str, arguments: dict[str, Any]) -> bool:
        try:
            await _maybe_end_task(task_ref)

            new_prompt = arguments.get("prompt", "")
            new_tools_json = arguments.get("tools", "{}")
            if not new_prompt:
                logger.error("No prompt provided in update_system_prompt action")
                return False

            logger.info(f"Updating system prompt to: {new_prompt[:100]}...")

            if arguments.get("add_suffix", True) and system_prompt_suffix:
                new_prompt = f"{new_prompt}\n{system_prompt_suffix}"

            new_messages = [{"role": system_role, "content": new_prompt}]

            original_messages.clear()
            original_messages.extend(new_messages)

            user_aggregator.reset()
            assistant_aggregator.reset()
            user_aggregator.set_messages(copy.deepcopy(new_messages))
            assistant_aggregator.set_messages(copy.deepcopy(new_messages))

            if (
                enable_tool_calling
                and new_tools_json
                and tool_factory is not None
                and register_schema_tools is not None
            ):
                # Registry namespace the bot should look tools up in. Defaults
                # to "default" (set by the bridge for non-tau2/eva scenarios).
                # Per-tool fallback to "default" happens inside ``tool_factory``
                # for shared harness tools (EndConversationTool, etc.).
                tool_domain = arguments.get("tool_domain", "default")
                logger.info(f"Registering new tools in domain={tool_domain!r}...")
                new_tools = json.loads(new_tools_json)

                # Initialize shared_state from the optional shared_state_init
                # payload produced by Scenario.setup_shared_state(). Inline DB
                # content (state["db"]) is the primary path; path-based loading
                # (state["db_path"]) is a fallback for fixtures too large to
                # ship inline.
                shared_state: dict = json.loads(arguments.get("shared_state_init", "{}"))
                if "db_path" in shared_state:
                    # Lazy import to avoid coupling rtvi_actions to evaluation/.
                    from nemo_voice_agent.evaluation import get_eval_data_root

                    db_path = shared_state.pop("db_path")
                    full_path = get_eval_data_root() / db_path
                    if not full_path.exists():
                        raise FileNotFoundError(
                            f"Scenario DB not found at {full_path} (from db_path={db_path!r}). "
                            f"Check EVAL_DATA_ROOT (currently resolves to {get_eval_data_root()})."
                        )
                    shared_state["db"] = json.loads(full_path.read_text())
                    logger.info(f"Loaded scenario DB from {full_path} into shared_state['db']")

                # Publish the dict so sibling action handlers (e.g. get_scenario_summary)
                # can read the same shared_state without needing tool references.
                if shared_state_ref is not None:
                    shared_state_ref.state = shared_state

                new_schema_tools = [
                    tool_factory(
                        tool_name,
                        domain=tool_domain,
                        rtvi=rtvi,
                        shared_state=shared_state,
                        **tool_args,
                    )
                    for tool_name, tool_args in new_tools.items()
                ]
                register_schema_tools(
                    llm=llm,
                    context=context,
                    tools=new_schema_tools,
                    cancel_on_interruption=False,
                    keep_existing_tools=False,
                )
            else:
                logger.info(
                    "Tool calling disabled, no tools provided, or tool_factory not configured; skipping tool registration."
                )

            logger.debug(f"user context tools: {user_aggregator._context.tools}")
            logger.debug(f"assistant context tools: {assistant_aggregator._context.tools}")

            _reset_services(resettable_services)

            logger.info("System prompt updated and context reset successfully")
            return True
        except Exception as e:
            logger.error(f"Error updating system prompt: {e}")
            return False

    return RTVIAction(
        service="context",
        action="update_system_prompt",
        result="bool",
        arguments=[
            {"name": "prompt", "type": "string", "required": True},
            {"name": "tools", "type": "string", "required": False, "default": "{}"},
            {"name": "add_suffix", "type": "bool", "required": False, "default": True},
            {
                "name": "shared_state_init",
                "type": "string",
                "required": False,
                "default": "{}",
            },
        ],
        handler=handler,
    )


def create_get_context_history_action(
    task_ref: TaskRef,
    assistant_aggregator,
) -> RTVIAction:
    """Build the ``context.get_context_history`` action.

    Returns the assistant aggregator's full message list, stringified to match
    the shape evaluation clients expect.

    **Race-safety against in-flight function calls.** When the bridge fetches
    the context immediately after receiving the ``<exit>`` server message (the
    `SendExitMessageTool` / `EndConversationTool` flow), the agent's final
    ``tool_call`` may still be in-flight in the aggregator's pipeline — the
    exit message arrives via the ``bot_server_message`` channel while the
    tool_call frame is still being committed via the
    ``FunctionCallInProgressFrame`` → ``FunctionCallResultFrame`` cycle. If we
    read the context before those frames commit, the captured message list is
    missing the final assistant turn. The judge then sees a stale context
    that doesn't contain the EndConversationTool tool_call and (incorrectly)
    deducts points for "didn't call EndConversationTool". 

    Fix: poll the aggregator until its in-progress map drains, with a hard
    deadline so a stuck tool can never deadlock scenario cleanup. The common
    case (no pending calls) returns immediately — zero added latency.

    """

    async def handler(rtvi_processor: RTVIProcessor, service: str, arguments: dict[str, Any]) -> dict:
        # Wait for any in-flight function calls to commit to the context before
        # snapshotting. ``has_function_calls_in_progress`` is pipecat's
        # public @property on ``LLMResponseAggregator`` (NOT a method —
        # access as an attribute, no parens, or it raises
        # ``'bool' object is not callable`` at runtime); the bounded 3 s
        # deadline ensures we always return — a stuck tool downgrades to a
        # warning, not a deadlock. 
        if hasattr(assistant_aggregator, "has_function_calls_in_progress"):
            deadline = asyncio.get_event_loop().time() + 3.0
            waited_initial = assistant_aggregator.has_function_calls_in_progress
            while assistant_aggregator.has_function_calls_in_progress:
                if asyncio.get_event_loop().time() >= deadline:
                    logger.warning(
                        "get_context_history: function calls still in progress after 3s; "
                        "returning context anyway (may be stale)"
                    )
                    break
                await asyncio.sleep(0.05)
            if waited_initial:
                logger.debug("get_context_history: aggregator drained, snapshotting context")
        try:
            messages = assistant_aggregator._context.get_messages()
            logger.debug(f"Returning context history: {len(messages)} messages")
            return {"context": str(messages)}
        except Exception as e:
            logger.error(f"Error getting context history: {e}")
            return {"context": []}

    return RTVIAction(
        service="context",
        action="get_context_history",
        result="object",
        arguments=[],
        handler=handler,
    )


def create_get_scenario_summary_action(
    task_ref: TaskRef,
    shared_state_ref: SharedStateRef,
) -> RTVIAction:
    """Build the ``context.get_scenario_summary`` action.

    Returns ``{"actions": [...], "db_hash": "<sha>"}`` from the per-scenario
    shared state by default. With ``include_db=true`` in the request payload,
    also returns the inline ``db`` dict alongside the hash — used by the
    runner's ``db_state_assertions`` aggregation (predicates need the actual
    DB values, not just the hash). Auto-aggregating tools (e.g.
    ``WriteScenarioTool`` subclasses) populate ``shared_state["actions"]`` on
    each successful mutation; the inbound fixture-loading flow populates
    ``shared_state["db"]``. The bridge calls this action after ``<exit>``
    (or scenario timeout) to retrieve the final artifacts without depending
    on any LLM-callable summary tool.

    **One DB per bot.** Each bot's ``shared_state["db"]`` IS this bot's DB —
    the agent bot's ``db`` is the agent-facing DB; the user bot's ``db`` is
    the user-facing DB. The naming distinction (``db`` vs ``user_db``) lives
    at the bridge/runner boundary: the bridge calls ``get_scenario_summary``
    once per bot and labels the responses by which WS it pulled from
    (the user-side pull is added when the first telecom scenario is
    ported). The bot itself doesn't know its own side and doesn't need to.

    **Hash-only outbound by default (not inline DB).** The DB itself stays on
    the bot server; only the SHA-256 of the canonicalized DB travels through
    the WebSocket. This keeps the response payload under a few KB regardless
    of DB size (tau2's airline DB is 7 MB inline; serialized via the previous
    inline-DB scheme it exceeded pipecat's 1 MB WebSocket frame limit and
    closed the connection with code 1009). Both the bot and the runner
    import the same ``get_dict_hash`` from ``nemo_voice_agent.evaluation.db_hash``
    so the canonical hashing rule (float normalization, order-independent
    list fields, excluded keys) is identical on both sides.

    **Inline DB opt-in (``include_db=true``).** Telecom's ``db_state_assertions``
    surface needs the runner to invoke predicate functions on the actual DB
    state, not just compare hashes. The bridge sets ``include_db=True`` when
    ``scenario.db_state_assertions`` is truthy. Telecom's per-bot DBs are
    small (~5 KB MockPhone state on the user side, modest customer/line data
    on the agent side) so the WS frame limit is not a concern; for retail
    (7 MB DB) the bridge leaves the flag at the default ``false`` and the
    existing hash-out behavior is preserved.

    Trade-off (when ``include_db=false``): the runner can no longer compute a
    per-field ``compute_db_diff`` on mismatch since it never sees the actual
    DB. For debugging hash mismatches in non-telecom domains, set
    ``include_db=true`` temporarily on the bridge call.

    Mirrors how ``get_context_history`` is consumed by the bridge.
    """
    # Lazy import to avoid coupling rtvi_actions (a pipecat-side module) to
    # the evaluation/ subpackage at module-load time.
    from nemo_voice_agent.evaluation.db_hash import get_dict_hash

    async def handler(rtvi_processor: RTVIProcessor, service: str, arguments: dict[str, Any]) -> dict:
        try:
            include_db = bool(arguments.get("include_db", False))
            actions = shared_state_ref.state.get("actions", [])
            db = shared_state_ref.state.get("db") or {}
            db_hash = get_dict_hash(db) if db else None
            logger.debug(
                f"Returning scenario summary: {len(actions)} action(s), "
                f"db_hash={db_hash}, include_db={include_db}"
            )
            await _maybe_end_task(task_ref)
            response: dict[str, Any] = {
                "actions": actions,
                "db_hash": db_hash,
            }
            if include_db:
                # Telecom-only path. Inline DB is needed when the runner
                # evaluates db_state_assertions — predicates take
                # ``(db: dict, **arguments) -> bool``, so the hash is
                # insufficient. The caller (bridge) is responsible for
                # only requesting this for domains where the DB is small
                # enough to safely cross the 1 MB pipecat WS frame limit.
                response["db"] = db or None
            return response
        except Exception as e:
            logger.error(f"Error getting scenario summary: {e}")
            return {"actions": [], "db_hash": None}

    return RTVIAction(
        service="context",
        action="get_scenario_summary",
        result="object",
        arguments=[
            {
                "name": "include_db",
                "type": "bool",
                "required": False,
                "default": False,
            },
        ],
        handler=handler,
    )


def create_apply_initialization_actions_action(
    shared_state_ref: SharedStateRef,
) -> RTVIAction:
    """Build the ``context.apply_initialization_actions`` action.

    Symmetric pre-scenario counterpart to ``get_scenario_summary``: takes a
    list of state-mutation steps and applies them to the bot's live
    ``shared_state["db"]`` / ``shared_state["user_db"]`` before the conversation
    starts. Used by telecom scenarios to seed device state (e.g.
    ``set_user_info``, ``turn_roaming_off``) and customer records
    (``enable_roaming``) so the agent talks to a meaningful starting state.

    **Bot-side dispatch (vs runner-side for ``db_state_assertions``).** Init
    actions mutate live DB dicts; the mutations have to land in the same dict
    instance the live LLM tools will read/write during the conversation —
    that's the bot's ``shared_state``, not a snapshot in the runner. Mirrors
    upstream tau2-bench's ``Environment.run_env_function_call`` which
    dispatches against live toolkit instances.

    **Payload shape:**

    .. code-block:: json

        {
          "domain": "tau2_telecom",
          "actions": [
            {"side": "user",  "func_name": "set_user_info",
             "arguments": {"name": "John Smith", "phone_number": "555-..."}},
            {"side": "agent", "func_name": "enable_roaming",
             "arguments": {"customer_id": "C1001", "line_id": "L1002"}}
          ]
        }

    The bridge splits the upstream ``initialization_actions`` list by ``side``
    and sends each subset to the matching bot (agent bot gets ``"agent"`` entries,
    user bot gets ``"user"`` entries). Per-bot calls therefore typically carry
    only one side's actions, but the dispatcher tolerates mixed lists for
    flexibility (e.g. when the same bot owns both DBs in a single-process
    test setup).

    **Returns** ``{"success": bool, "errors": list[str]}``. ``success`` is
    ``True`` only when every action dispatched cleanly. The bridge treats any
    ``success=False`` as a framework-class failure and aborts the scenario
    without scoring it — partial seeding produces noise, not signal.
    """
    # Lazy import to avoid coupling rtvi_actions (a pipecat-side module) to
    # the evaluation/ subpackage at module-load time. Same pattern as the
    # ``get_dict_hash`` import in ``create_get_scenario_summary_action``.
    from nemo_voice_agent.evaluation.initialization_functions import (
        apply_initialization_actions as _apply,
    )

    async def handler(rtvi_processor: RTVIProcessor, service: str, arguments: dict[str, Any]) -> dict:
        try:
            domain = arguments.get("domain", "default")
            actions = arguments.get("actions") or []
            if not isinstance(actions, list):
                return {
                    "success": False,
                    "errors": [f"`actions` payload must be a list, got {type(actions).__name__}."],
                }
            # Each bot owns exactly one DB at ``shared_state["db"]``. The
            # agent bot's ``db`` is the agent-facing DB; the user bot's
            # ``db`` is the user-facing DB. The bridge already filtered the
            # upstream action list to only the entries that belong to this
            # bot before sending — so all actions in the payload apply to
            # this single ``db``. The handler is side-agnostic.
            db = shared_state_ref.state.get("db") if shared_state_ref.state else None
            logger.info(
                f"[APPLY INIT] domain={domain!r}, actions={len(actions)}, "
                f"db_present={db is not None}"
            )
            if db is None:
                return {
                    "success": False,
                    "errors": [
                        f"No shared_state['db'] available; cannot apply "
                        f"{len(actions)} initialization action(s). "
                        f"update_system_prompt didn't seed the DB on this bot."
                    ],
                }
            result = _apply(domain=domain, actions=actions, db=db)
            if result["success"]:
                logger.info(f"[APPLY INIT] success ({len(actions)} action(s) applied)")
            else:
                logger.warning(
                    f"[APPLY INIT] failure with {len(result['errors'])} error(s): "
                    f"{result['errors']}"
                )
            return result
        except Exception as e:
            logger.error(f"Error applying initialization actions: {e}")
            return {"success": False, "errors": [f"{type(e).__name__}: {e}"]}

    return RTVIAction(
        service="context",
        action="apply_initialization_actions",
        result="object",
        arguments=[
            {"name": "domain", "type": "string", "required": False, "default": "default"},
            {"name": "actions", "type": "array", "required": True},
        ],
        handler=handler,
    )

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

    **Scenario-start lifecycle gate.** This handler is the bot-side signal
    that a new scenario is starting. In addition to swapping the prompt + tool
    surface, it RESETS ``shared_state_ref.state`` so any prior scenario's
    data (``db``, ``actions`` log, etc.) doesn't bleed into the new one. The
    reset is done via ``dict.clear()`` rather than reassignment so the dict
    identity is preserved — any tool that already holds a reference to
    ``shared_state`` continues to see the same object after the clear, and
    subsequent mutations by ``apply_initialization`` propagate correctly.

    Scenario fixture data (``db``, ``db_path``, ``actions`` list, custom
    keys from ``Scenario.setup_shared_state``) is NOT loaded here — that
    moves to ``create_apply_initialization_action`` which the bridge calls
    immediately after this one. This handler only stashes bot-side runtime
    sentinels (``__rtvi__``, ``__tool_domain__``) into the freshly-cleared
    dict so write tools have them available at call time.
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

            # Scenario-start reset of shared_state. Preserve the dict
            # identity so tools registered earlier (in a previous
            # scenario) keep their reference valid. Tools access state
            # lazily via ``self.state.get("db")`` etc., so seeing the
            # post-clear empty dict + later ``apply_initialization``
            # mutations is the expected sequence. ``SharedStateRef.state``
            # is always a real dict (default_factory=dict) so we don't
            # need to handle the ``None`` case.
            if shared_state_ref is not None:
                shared_state = shared_state_ref.state
                shared_state.clear()
            else:
                shared_state = {}

            # Stash bot-side runtime sentinels. These come from the BOT
            # (the rtvi processor instance, the active tool domain) —
            # not from scenario fixture data — so they belong here and
            # not in ``apply_initialization``.
            #
            # ``__rtvi__``: read by ``WriteScenarioTool._record_action``
            # to push ``action-applied`` RTVIServerMessages used by the
            # bridge's cross-side sync pipeline. Sentinel name starts
            # with ``__`` so it won't appear in JSON-serialized DB
            # snapshots (``db_hash`` filters dunder keys).
            #
            # ``__tool_domain__``: included in the ``action-applied``
            # payload so the bridge routes sync deltas to the correct
            # domain's applier.
            shared_state["__rtvi__"] = rtvi
            shared_state["__tool_domain__"] = arguments.get("tool_domain", "default")

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
            {"name": "tool_domain", "type": "string", "required": False, "default": "default"},
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


def create_apply_initialization_action(
    shared_state_ref: SharedStateRef,
) -> RTVIAction:
    """Build the ``context.apply_initialization`` action.

    Bot-side scenario-state initializer. The bridge calls this once per
    scenario, immediately after ``update_system_prompt``, to populate
    the empty ``shared_state`` dict with everything tools need at call
    time. Does three things in order:

    1. **Parse** ``shared_state_init`` JSON into ``shared_state_ref.state``
       (merge, not overwrite — preserves bot-side sentinels stashed by
       ``update_system_prompt`` like ``__rtvi__`` and ``__tool_domain__``).
       Custom keys from ``Scenario.setup_shared_state`` flow through here.
    2. **Load DB**: if the merged state has a ``db_path`` key, resolve it
       against ``EVAL_DATA_ROOT`` and replace with the loaded ``db`` dict.
       Idempotent — skipped when ``db`` is already present (e.g., when
       the bridge sent an inline DB instead of a path).
    3. **Apply init functions**: dispatch each ``{func_name, arguments}``
       record against the now-loaded ``db``. Side-agnostic — the bridge
       pre-filters by ``side`` before sending.

    Defensive create-if-missing: if ``shared_state_ref.state`` is
    ``None`` (e.g., in tests that skip ``update_system_prompt``), the
    handler creates an empty dict and proceeds. In production the bridge
    always calls ``update_system_prompt`` first, which creates + clears
    the dict and stashes runtime sentinels.

    **Payload shape:**

    .. code-block:: json

        {
          "domain": "tau2_telecom",
          "shared_state_init": "{\\"db_path\\": \\"tau2_telecom/db.json\\"}",
          "actions": [
            {"side": "user",  "func_name": "set_user_info",
             "arguments": {"name": "John Smith", "phone_number": "555-..."}},
            {"side": "agent", "func_name": "enable_roaming",
             "arguments": {"customer_id": "C1001", "line_id": "L1002"}}
          ]
        }

    The bridge sends this action per-bot. ``shared_state_init`` carries
    per-side scenario fixture data (path or inline DB, optional initial
    actions list, etc.); ``actions`` carries the per-side filtered subset
    of upstream init mutations. Both are independently optional — a
    scenario with no init functions and an inline DB still benefits
    from a single ``apply_initialization`` call that does only the
    DB-load step.

    **Returns** ``{"success": bool, "errors": list[str]}``. ``success``
    is ``True`` only when DB load and every action dispatch completed
    cleanly. The bridge treats any ``success=False`` as a
    framework-class failure and aborts the scenario without scoring it
    — partial seeding produces noise, not signal.
    """
    # Lazy imports to avoid coupling rtvi_actions (a pipecat-side module) to
    # the evaluation/ subpackage at module-load time. Same pattern as the
    # ``get_dict_hash`` import in ``create_get_scenario_summary_action``.
    from nemo_voice_agent.evaluation.initialization_functions import (
        apply_initialization_actions as _apply,
    )

    async def handler(rtvi_processor: RTVIProcessor, service: str, arguments: dict[str, Any]) -> dict:
        try:
            domain = arguments.get("domain", "default")
            shared_state_init_raw = arguments.get("shared_state_init", "{}")
            actions = arguments.get("actions") or []
            if not isinstance(actions, list):
                return {
                    "success": False,
                    "errors": [f"`actions` payload must be a list, got {type(actions).__name__}."],
                }

            # 1. Merge scenario fixture data into shared_state. Preserve
            #    dict identity — tools registered in update_system_prompt
            #    hold a reference; reassigning would break them. The
            #    merge into existing keys preserves bot-side runtime
            #    sentinels (``__rtvi__``, ``__tool_domain__``) that
            #    update_system_prompt stashed. ``SharedStateRef.state``
            #    is always a real dict (default_factory=dict).
            shared_state = shared_state_ref.state
            try:
                init_payload = json.loads(shared_state_init_raw)
            except json.JSONDecodeError as exc:
                return {
                    "success": False,
                    "errors": [f"Invalid shared_state_init JSON: {exc}"],
                }
            if not isinstance(init_payload, dict):
                return {
                    "success": False,
                    "errors": [
                        f"shared_state_init must decode to a dict, got "
                        f"{type(init_payload).__name__}."
                    ],
                }
            shared_state.update(init_payload)

            # 2. Resolve db_path → db. Idempotent: if ``db`` is already
            #    present (e.g., bridge sent inline DB content rather
            #    than a path), skip the load. Missing files raise loudly.
            if "db_path" in shared_state and "db" not in shared_state:
                # Lazy import to avoid coupling rtvi_actions to evaluation/.
                from nemo_voice_agent.evaluation import get_eval_data_root, load_db_artifact

                db_path = shared_state.pop("db_path")
                full_path = get_eval_data_root() / db_path
                try:
                    shared_state["db"] = load_db_artifact(full_path)
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        f"Scenario DB not found for db_path={db_path!r}. "
                        f"Check EVAL_DATA_ROOT (currently resolves to {get_eval_data_root()}). "
                        f"Underlying error: {exc}"
                    ) from exc
                logger.info(f"Loaded scenario DB for db_path={db_path!r} into shared_state['db']")
            elif "db_path" in shared_state:
                # ``db`` already present — drop the now-redundant path so
                # subsequent calls don't keep trying to resolve it.
                shared_state.pop("db_path", None)

            # 3. Each bot owns exactly one DB at ``shared_state["db"]``;
            #    the bridge already filtered the upstream action list to
            #    only entries belonging to this bot before sending. The
            #    handler is side-agnostic.
            db = shared_state.get("db")
            logger.info(
                f"[APPLY INIT] domain={domain!r}, actions={len(actions)}, "
                f"db_present={db is not None}"
            )
            if actions and db is None:
                return {
                    "success": False,
                    "errors": [
                        f"No shared_state['db'] available; cannot apply "
                        f"{len(actions)} initialization action(s). "
                        f"Bridge must send db_path (or inline db) in shared_state_init."
                    ],
                }
            if not actions:
                # No init mutations to apply; the DB-load step above is
                # the entire purpose of this call. Return success so the
                # bridge can proceed to the conversation.
                logger.info(f"[APPLY INIT] no actions to apply (DB-load-only call)")
                return {"success": True, "errors": []}

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
            logger.error(f"Error applying initialization: {e}")
            return {"success": False, "errors": [f"{type(e).__name__}: {e}"]}

    return RTVIAction(
        service="context",
        action="apply_initialization",
        result="object",
        arguments=[
            {"name": "domain", "type": "string", "required": False, "default": "default"},
            {"name": "shared_state_init", "type": "string", "required": False, "default": "{}"},
            {"name": "actions", "type": "array", "required": False, "default": []},
        ],
        handler=handler,
    )


def create_apply_sync_delta_action(
    shared_state_ref: SharedStateRef,
) -> RTVIAction:
    """Build the ``context.apply_sync_delta`` action.

    Bot-side endpoint of the cross-side state-propagation pipeline.
    The bridge calls this after an ``action-applied`` event from the
    *other* bot — it runs ``scenario.sync_state(agent_db, user_db)`` on
    its in-process shadow DBs, then pushes any non-empty per-side delta
    to the corresponding bot via this action.

    **Payload shape:**

    .. code-block:: json

        {
          "domain": "tau2_telecom",
          "delta": {
            "surroundings.payment_request": {"bill_id": "B1002", ...},
            "surroundings.line_active": true
          }
        }

    The bot's handler dispatches via
    ``nemo_voice_agent.evaluation.sync_appliers.apply_sync_delta`` —
    looks up the domain-specific applier (falls back to the generic
    dotted-path setter) and mutates ``shared_state["db"]`` in place.

    For single-side domains (eva / airline / retail) this action would
    never be called by the bridge — only scenarios overriding
    ``Scenario.sync_state`` trigger the pipeline. But the action is
    registered domain-agnostically so the same bot binary can run any
    scenario.

    **Returns** ``{"success": bool, "errors": list[str]}``. ``success``
    is ``True`` when the delta applied cleanly. Errors are
    informational — the bridge logs them and continues (a malformed
    delta should not crash the bot or stall the conversation).
    """
    # Lazy import — same rationale as create_apply_initialization_actions.
    from nemo_voice_agent.evaluation.sync_appliers import apply_sync_delta as _apply_delta

    async def handler(rtvi_processor: RTVIProcessor, service: str, arguments: dict[str, Any]) -> dict:
        try:
            domain = arguments.get("domain", "default")
            delta = arguments.get("delta") or {}
            if not isinstance(delta, dict):
                return {
                    "success": False,
                    "errors": [f"`delta` payload must be a dict, got {type(delta).__name__}."],
                }
            db = shared_state_ref.state.get("db") if shared_state_ref.state else None
            logger.info(
                f"[APPLY SYNC] domain={domain!r}, delta_keys={list(delta.keys())}, "
                f"db_present={db is not None}"
            )
            if db is None:
                return {
                    "success": False,
                    "errors": [
                        f"No shared_state['db'] available; cannot apply sync delta "
                        f"with {len(delta)} key(s). update_system_prompt didn't "
                        f"seed the DB on this bot."
                    ],
                }
            _apply_delta(domain=domain, db=db, delta=delta)
            logger.info(f"[APPLY SYNC] success ({len(delta)} key(s) applied)")
            return {"success": True, "errors": []}
        except Exception as e:
            logger.error(f"Error applying sync delta: {e}")
            return {"success": False, "errors": [f"{type(e).__name__}: {e}"]}

    return RTVIAction(
        service="context",
        action="apply_sync_delta",
        result="object",
        arguments=[
            {"name": "domain", "type": "string", "required": False, "default": "default"},
            {"name": "delta", "type": "object", "required": True},
        ],
        handler=handler,
    )

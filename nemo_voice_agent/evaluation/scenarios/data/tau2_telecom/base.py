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
# Scenario fixtures live under evaluation/data/tau2_telecom/ — adapted from
# https://github.com/sierra-research/tau2-bench/tree/voice-user-sim-v1.0
# (MIT-licensed). See evaluation/data/README.md for the upstream pin
# (commit 17e07b1).

"""Tau2-telecom scenario base.

``Tau2TelecomBaseScenario`` extends ``Tau2BaseScenario`` with three
telecom-specific concerns the airline/retail bases don't share:

1. **Dual DB.** A separate user-side ``TelecomUserDB`` (mock phone state +
   user surroundings) lives alongside the agent-side ``TelecomDB``.
   ``has_user_state = True`` triggers user-side seeding in
   ``_gold_replay`` and tells the bridge to dual-pull.
2. **Multi-file policy.** The agent system prompt is two files concatenated:
   ``main_policy.md`` + ``tech_support_{manual,workflow}.md``. We
   override ``policy`` to do the concat using a markdown horizontal-rule
   separator (no XML tags — matches airline/retail's single-markdown
   convention; both upstream files are well-formed standalone markdown).
3. **``db_state_assertions`` + ``initialization_actions`` translation.**
   Upstream uses ``env_type ∈ {"user", "assistant"}`` on both fields;
   the registry/framework uses ``side ∈ {"user", "agent"}``. We translate
   at scenario load time so the runner / bridge / registries all see a
   uniform shape. The helper lives in ``tau2_common`` for reuse if
   another tau2 domain adds these fields.

Subclasses only need to set ``name`` and ``tau2_id``; everything else
derives from the upstream task JSON.
"""

# pylint: disable=line-too-long
# flake8: noqa: E501

import json
from functools import cached_property
from typing import Any, ClassVar, Dict, List, Optional

from nemo_voice_agent.evaluation import get_eval_data_root
from nemo_voice_agent.evaluation.scenarios.classes import Actions, Resources
from nemo_voice_agent.evaluation.scenarios.data.tau2_common import (
    Tau2BaseScenario,
    _normalize_env_record,
)


# Procedural script for the user-sim: the conversation arc telecom
# scenarios assume. Keeps the model anchored in the
# describe → follow → report loop so it doesn't free-fire its
# phone-control tools (the failure mode we saw in
# eval_20260608_084540: user-sim fired all 4 tools on agent's
# greeting turn, then narrated as the agent).
TELECOM_USER_INSTRUCTIONS: List[str] = [
    (
        "Greet the agent and explain your issue in natural language. Describe the symptom "
        "you're experiencing without proposing solutions."
    ),
    (
        "Follow the agent's diagnostic instructions step by step. When the agent asks you "
        "to perform a specific action on your phone, use the corresponding tool — one per "
        "agent instruction."
    ),
    (
        "Verbally report what each tool returned back to the agent on your next spoken turn. "
        "Let the agent decide what to try next."
    ),
    (
        "Confirm with the agent when the symptom is gone, and say goodbye when the agent "
        "has finished resolving the issue."
    ),
]


# Behavioral constraint: never call a tool proactively. Pairs with
# ``TELECOM_USER_INSTRUCTIONS`` — the script defines the rhythm, this
# guideline forbids the failure mode (free-firing tools, narrating as
# the agent). Appended to the parent's ``guidelines`` (which already
# carries ``VOICE_ALPHANUMERIC_RULE``).
TELECOM_PASSIVE_TOOL_USE_GUIDELINE: str = (
    "Passive tool use only: never call a tool unless the agent has just instructed you "
    "to take a specific action. You are the customer, not the troubleshooter — the agent "
    "diagnoses, you describe and follow. Ground all spoken claims about your phone's "
    "state on the actual return value of the most recent tool call; never narrate as the "
    "agent (phrases like \"the issue is resolved\" or \"is there anything else I can help "
    "with?\" are the agent's lines, not yours)."
)
from nemo_voice_agent.evaluation.tools.tau2_telecom_sync import sync_telecom_state
from nemo_voice_agent.evaluation.tools.tau2_telecom_tools import (
    TAU2_TELECOM_AGENT_TOOL_NAME_TO_CLASS,
)
from nemo_voice_agent.evaluation.tools.tau2_telecom_user_tools import (
    TAU2_TELECOM_USER_TOOL_NAME_TO_CLASS,
)


# Telecom-specific addenda appended to the agent prompt AFTER the parent's
# voice-realization notes. These compensate for two structural gaps vs
# upstream tau2's text-mode evaluation:
#
# 1. **Tool availability** — upstream's policy text (``main_policy.md`` +
#    ``tech_support_manual.md``) references many user-side phone tools as
#    if callable (e.g. ``check_status_bar()``, ``toggle_airplane_mode()``).
#    In upstream text-mode the agent's OpenAI tool-call schema only exposes
#    the 13 agent-side tools, so the LLM can't hallucinate calls to
#    user-side names — schema validation catches them. In voice mode we
#    don't have shared conversation history (the agent never sees the
#    user-sim's tool calls), and the LLM was observed hallucinating
#    ``check_network_status`` directly. This block makes the split explicit.
#
# 2. **Stay-on-task** — upstream's policy lists "overdue bill payment",
#    "line suspension", "plan options" as supported flows. When the agent
#    discovers an unrelated overdue bill while troubleshooting mobile
#    data, it sometimes pivots to billing. Without cross-side
#    propagation of ``send_payment_request`` to ``surroundings.payment_request``
#    (a future bridge fix mirroring upstream ``sync_tools``), that pivot
#    deadlocks. This guideline tells the agent to stick to the user's
#    primary stated problem.

TELECOM_AGENT_TOOL_AVAILABILITY_NOTE = (
    "Tool availability for THIS voice scenario. The policy text above references many "
    "tools — but in this voice scenario you have ONLY the following set of agent-side "
    "tools registered, listed by the exact name you call them by:\n"
    "{agent_tool_names}\n\n"
    "All other tools mentioned in the policy text are on the USER'S PHONE. The user "
    "has these phone-control tools available to them and will operate their own device:\n"
    "{user_tool_names}\n\n"
    "When the policy says \"use `check_status_bar()`\" or \"guide the user to use "
    "`toggle_airplane_mode()`\", you are NOT supposed to call that tool yourself — "
    "you do not have it. Instead, instruct the user verbally and wait for them to "
    "report the result. Calling a user-side tool name will fail with an "
    "\"unknown_tool\" error; do not attempt it."
)


TELECOM_AGENT_STAY_ON_TASK_GUIDELINE = (
    "Stay on the user's stated problem. Address the PRIMARY issue the user reported "
    "first. Do not pivot to unrelated issues you discover incidentally (e.g. an "
    "overdue bill on the account, an expired contract on a sibling line, a "
    "suspended secondary line) unless one of these holds:\n"
    "- The discovered issue is directly causing the user's stated symptom (e.g. the "
    "  current line is suspended due to an overdue bill, so the user has no service).\n"
    "- The user explicitly asks about that other issue.\n"
    "If you discover something unrelated, you may briefly mention it at the end of "
    "the conversation (after the primary issue is resolved), but do not derail the "
    "current troubleshooting flow."
)


# Bridges a gap upstream policy.md leaves open: the policy says
# "outside the home network" without ever specifying what the home
# network is. Phone numbers (10-digit US format), customer addresses
# (US states), and billing (USD) all imply US-based, but the LLM
# needs that stated explicitly to confidently classify a user's
# named location (e.g. "France", "Mexico") as abroad.
TELECOM_AGENT_HOME_NETWORK_NOTE = (
    "Home network context + location-probe rule. This telecom company operates in "
    "the United States — phone numbers are 10-digit US format (e.g. 555-123-4567), "
    "customer addresses are US states, billing is in USD. \n\n" 
    "Always ASK the user where they are physically located right now "
    "for any connectivity complaint like no data, slow "
    "data, MMS not sending, no service, weak signal, etc. The user may not "
    "volunteer their location proactively; you must ask. Example opener: "
    "\"Before we check other settings on your account, can you tell me where you are right now? "
    "Are you home in the US, or traveling abroad?\"\n\n"
    "If the user reports being abroad with no data, IMMEDIATELY check `line.roaming_enabled` "
    "via `get_details_by_id(<line_id>)`. If `line.roaming_enabled` is False, "
    "explain that roaming is disabled on their line and ASK the user if they want to enable it. "
    "If the user says yes, call `enable_roaming` to enable roaming. "
)


class Tau2TelecomBaseScenario(Tau2BaseScenario):
    """Base class for scenarios ported from tau2-bench/telecom (voice-user-sim-v1.0).

    Subclasses only declare ``name`` and ``tau2_id`` (the upstream task id
    string). For example::

        @register_eval_scenario
        class Tau2TelecomMobileDataIssueAirplaneOnDataOff(Tau2TelecomBaseScenario):
            name = "tau2_telecom__mobile_data_issue__airplane_mode_on__data_mode_off"
            tau2_id = "[mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:None]"
    """

    domain: str = "tau2_telecom"

    # Triggers dual-DB seeding in gold replay (``_gold_replay`` calls
    # ``setup_shared_state(state, side="user")`` when this is True).
    has_user_state: bool = True

    # Default policy variant. Upstream offers two — ``"manual"`` (long-form
    # documentation) and ``"workflow"`` (procedural step-by-step). Matches
    # upstream's ``get_environment_manual_policy`` default. Subclass can
    # override to ``"workflow"`` per-scenario family if needed.
    policy_variant: ClassVar[str] = "manual"

    # -----------------------------------------------------------------------
    # Policy (overrides parent — two files concatenated)
    # -----------------------------------------------------------------------

    @cached_property
    def policy(self) -> str:
        """Concatenate ``main_policy.md`` + ``tech_support_{variant}.md``.

        Both files are well-formed standalone markdown (their own ``#`` /
        ``##`` / ``###`` header hierarchies). We join with a markdown
        horizontal-rule separator so the LLM has a clean section
        boundary without XML tags. The upstream telecom wraps each file
        in ``<main_policy>``/``<tech_support_policy>`` tags; we drop
        those because (a) airline / retail's single-markdown-string
        convention doesn't use XML, (b) markdown structure already
        provides hierarchy, (c) it saves ~30 tokens per request.
        """
        root = get_eval_data_root() / self.domain
        main_policy = (root / "main_policy.md").read_text().strip()
        tech_support = (root / f"tech_support_{self.policy_variant}.md").read_text().strip()
        return main_policy + "\n\n---\n\n" + tech_support

    # -----------------------------------------------------------------------
    # DBs — dual (agent-side ``db`` + user-side ``user_db``)
    # -----------------------------------------------------------------------

    @cached_property
    def user_db(self) -> Dict[str, Any]:
        """Initial user-side DB state — loaded once per process from ``user_db.json``.

        The Pydantic round-trip in ``scripts/prepare_tau2_data/prepare_telecom.py``
        materializes default fields (``surroundings`` block,
        ``signal_strength`` per-network table, ``app_statuses`` defaults)
        that the raw TOML doesn't carry, so downstream init / predicate
        code can read fields via dict access without defensive
        ``.get(default)`` calls.

        ``setup_shared_state`` deepcopies this when seeding user-side
        state so per-scenario mutations don't leak.
        """
        return json.loads((get_eval_data_root() / self.domain / "user_db.json").read_text())

    # -----------------------------------------------------------------------
    # db_state_assertions / initialization_actions — translate upstream shape
    # -----------------------------------------------------------------------

    @cached_property
    def db_state_assertions(self) -> Optional[List[Dict[str, Any]]]:
        """Per-task ``db_state_assertions`` records, translated from
        upstream's ``env_assertions``.

        Reads ``evaluation_criteria.env_assertions`` and runs each record
        through ``_normalize_env_record`` to rename ``env_type → side`` +
        ``"assistant" → "agent"``. Returns ``None`` (not ``[]``) when the
        task has no assertions so the runner's ``if scenario.db_state_assertions:``
        guard correctly skips verdict aggregation.
        """
        criteria = self.tau2_task.get("evaluation_criteria") or {}
        records = criteria.get("env_assertions")
        if not records:
            return None
        return [_normalize_env_record(rec) for rec in records]

    @cached_property
    def initialization_actions(self) -> Optional[List[Dict[str, Any]]]:
        """Per-task ``initialization_actions`` records, translated from upstream.

        Source: ``initial_state.initialization_actions``. Same translation
        as ``db_state_assertions`` — ``env_type → side``, ``"assistant"
        → "agent"``. Returns ``None`` when the task has none.
        """
        initial_state = self.tau2_task.get("initial_state") or {}
        records = initial_state.get("initialization_actions")
        if not records:
            return None
        return [_normalize_env_record(rec) for rec in records]

    @cached_property
    def nl_assertions(self) -> Optional[List[str]]:
        """Per-task ``nl_assertions`` for the LLM judge. ``None`` when absent."""
        criteria = self.tau2_task.get("evaluation_criteria") or {}
        assertions = criteria.get("nl_assertions")
        if not assertions:
            return None
        return list(assertions)

    # -----------------------------------------------------------------------
    # shared_state seeding — agent + user sides
    # -----------------------------------------------------------------------

    def setup_shared_state(self, state: dict, side: str) -> None:
        """Seed ``state["db_path"]`` per side (live-runtime path).

        - ``side="agent"`` → ``tau2_telecom/db.json`` (TelecomDB)
        - ``side="user"``  → ``tau2_telecom/user_db.json`` (TelecomUserDB)

        Path-based seeding (not inline DB) — the bot-server's
        ``rtvi_actions.create_update_system_prompt_action`` handler
        resolves the path against ``EVAL_DATA_ROOT`` and loads the file
        into ``shared_state["db"]`` on the bot side. Each bot ends up
        with its own DB at ``state["db"]`` (per-bot single-DB
        convention).

        Gold replay runs in-process and bypasses this method's
        user-side branch — the parent class's ``_gold_replay``
        inline-loads ``self.user_db`` directly into ``gold_state["user_db"]``.
        """
        if side == "agent":
            state["db_path"] = f"{self.domain}/db.json"
        elif side == "user":
            state["db_path"] = f"{self.domain}/user_db.json"

    # -----------------------------------------------------------------------
    # _build_tool_map — consumed by Tau2BaseScenario._gold_replay
    # -----------------------------------------------------------------------

    def _build_tool_map(self, state: dict) -> Dict[str, Any]:
        """Tool name → instance map for BOTH user-side and agent-side tools.

        Consumed by ``Tau2BaseScenario._gold_replay`` to dispatch each
        reference action by name. Telecom is the first dual-side
        domain, so the map must cover both surfaces:

          - **User-side tools** operate on the user_db. In gold-replay
            the same ``state`` carries both ``state["db"]`` (agent's
            TelecomDB) and ``state["user_db"]`` (user's TelecomUserDB).
            We bridge by wrapping ``state`` in a user-view dict where
            ``state["db"]`` aliases ``state["user_db"]`` (gold replay)
            or falls back to ``state["db"]`` (live runtime, where the
            user bot's state["db"] IS the user_db).
          - **Agent-side tools** operate on the agent TelecomDB
            directly via ``state["db"]`` — no view wrapping needed.

        The shared ``state["actions"]`` list collects both sides' write
        records; the gold-replay caller (``Tau2BaseScenario._gold_replay``)
        side-tags them after replay so the runner's comparator sees
        per-side action streams that align with the per-action ``side``
        field in ``reference_answer``.

        Tool-name collisions between sides are impossible — upstream's
        user-side and agent-side surfaces have disjoint method names
        (user: ``toggle_*`` / ``check_*`` / ``run_speed_test``; agent:
        ``get_*`` / ``send_payment_request`` / ``refuel_data`` etc.).
        """
        actions = state.setdefault("actions", [])
        # User-side view: state["db"] aliases state["user_db"] in gold-replay,
        # or falls back to state["db"] in live runtime.
        user_db = state.get("user_db") if state.get("user_db") is not None else state.get("db")
        user_view: Dict[str, Any] = {"db": user_db, "actions": actions}
        # Agent-side: pass state through unchanged (state["db"] is already the agent DB).
        agent_view: Dict[str, Any] = {"db": state.get("db"), "actions": actions}
        tool_map: Dict[str, Any] = {
            name: cls(shared_state=user_view) for name, cls in TAU2_TELECOM_USER_TOOL_NAME_TO_CLASS.items()
        }
        tool_map.update(
            {name: cls(shared_state=agent_view) for name, cls in TAU2_TELECOM_AGENT_TOOL_NAME_TO_CLASS.items()}
        )
        return tool_map

    # -----------------------------------------------------------------------
    # user_actions — passive tool-use script + guideline
    # -----------------------------------------------------------------------

    @cached_property
    def user_actions(self) -> Actions:
        """User-side actions — telecom-specific procedural script + passive-tool-use constraint.

        OVERRIDES ``Tau2BaseScenario.user_actions`` (which returns just the
        ``VOICE_ALPHANUMERIC_RULE`` guideline). Telecom is the first dual-side
        domain whose user-sim has LLM-callable tools, and the default
        tau2 ``task_instructions`` (preserved in ``user_persona.personality``)
        was written for text-mode where the user-sim chains tool calls
        inside one turn. In voice mode the user-sim has to talk to the
        agent first — hence the explicit script + constraint here.

        - ``instructions`` carry the procedural sequence (describe →
          follow → report → close). The model uses these to anchor
          itself in the conversational rhythm.
        - ``guidelines`` carry behavioral constraints (passive tool
          use, voice-readable identifiers). Stay constraint-shaped so
          the model can check any single turn against them.

        Inherits the parent's guidelines (``VOICE_ALPHANUMERIC_RULE``)
        and appends the telecom-specific passive-tool-use rule.
        """
        parent = super().user_actions
        return Actions(
            instructions=list(TELECOM_USER_INSTRUCTIONS),
            guidelines=list(parent.guidelines) + [TELECOM_PASSIVE_TOOL_USE_GUIDELINE],
        )

    # -----------------------------------------------------------------------
    # user_resources — register the 4 ported user-side LLM tools
    # -----------------------------------------------------------------------

    @cached_property
    def user_resources(self) -> Resources:
        """User-side resources — phone-control LLM tools + ``known_info`` /
        ``unknown_info`` from the parent.

        OVERRIDES the parent's user_resources (which returns the same
        info-sections but with empty ``tools={}``). Telecom is the
        first domain that populates ``user_resources.tools`` — the
        user-sim's LLM needs phone-control tools (toggle_data,
        toggle_airplane_mode, etc.) to actually act on the device the
        agent is troubleshooting.

        Tool kwargs are empty because tau2 tools take only
        ``shared_state`` which the bot server injects automatically.
        """
        parent = super().user_resources
        # Keys are snake_case tool names (from each tool's class-level
        # ``name`` attribute). Single source of truth: same key drives
        # the bot's ``tool_factory`` registry lookup, the LLM-visible
        # function-call name (matching how the policy.md text references
        # tools — ``toggle_airplane_mode()``, ``check_status_bar()``,
        # etc.), and ``_build_tool_map``'s gold-replay dispatch.
        return Resources(
            tools={name: {} for name in TAU2_TELECOM_USER_TOOL_NAME_TO_CLASS},
            documents={},
            information=parent.information,
            info_sections=parent.info_sections,
        )

    # -----------------------------------------------------------------------
    # agent_resources — minimal stub until agent-side telecom tools land
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # sync_state — cross-side state propagation (mirrors upstream sync_tools)
    # -----------------------------------------------------------------------

    def sync_state(self, agent_db: dict, user_db: dict) -> Dict[str, Dict[str, Any]]:
        """Delegate to the pure ``sync_telecom_state`` function.

        Called by the bridge after each write action on either bot. The
        actual reconciliation logic — propagating ``line.status``,
        ``line.roaming_enabled``, data-usage, and payment-request state
        between the two DBs — lives in ``sync_telecom_state`` so it can
        be unit-tested without bridge or scenario plumbing.

        See ``Scenario.sync_state`` for the framework contract and
        ``sync_telecom_state``'s docstring for the propagation paths.
        """
        return sync_telecom_state(agent_db, user_db)

    # -----------------------------------------------------------------------
    # get_agent_prompt — extend parent with telecom-specific addenda
    # -----------------------------------------------------------------------

    def get_agent_prompt(self) -> str:
        """Parent's policy + voice notes + telecom-specific addenda.

        Appends three telecom-only blocks to the parent's
        ``policy + Additional Notes`` output:

        - ``TELECOM_AGENT_TOOL_AVAILABILITY_NOTE``: explicit enumeration
          of which tools the agent CAN call vs which live on the user's
          phone. Substitutes the actual snake_case tool names so the
          model can map policy.md references unambiguously.
        - ``TELECOM_AGENT_STAY_ON_TASK_GUIDELINE``: prevents drift to
          unrelated discovered issues (e.g. overdue bills).
        - ``TELECOM_AGENT_HOME_NETWORK_NOTE``: states the home country
          (US) and frontloads the "check roaming first when abroad"
          diagnostic rule — disambiguates user-reported locations
          ("France", "overseas") that policy.md mentions only
          generically as "outside the home network".

        See module-level docstrings on the three constants for the
        upstream-vs-voice-mode structural reasoning.
        """
        agent_names = sorted(TAU2_TELECOM_AGENT_TOOL_NAME_TO_CLASS.keys()) + ["EndConversationTool"]
        user_names = sorted(TAU2_TELECOM_USER_TOOL_NAME_TO_CLASS.keys())
        availability = TELECOM_AGENT_TOOL_AVAILABILITY_NOTE.format(
            agent_tool_names="\n".join(f"- `{n}`" for n in agent_names),
            user_tool_names="\n".join(f"- `{n}`" for n in user_names),
        )
        return (
            super().get_agent_prompt()
            + "\n\n"
            + availability
            + "\n\n"
            + TELECOM_AGENT_STAY_ON_TASK_GUIDELINE
            + "\n\n"
            + TELECOM_AGENT_HOME_NETWORK_NOTE
        )

    @cached_property
    def agent_resources(self) -> Resources:
        """Agent-side resources — full tau2_telecom tool surface + ``EndConversationTool``.

        OVERRIDES ``Tau2BaseScenario.agent_resources``. Tool keys are
        snake_case ``cls.name`` strings matching upstream method names —
        the same identifiers the policy.md text references (e.g.
        ``get_customer_by_phone``, ``send_payment_request``). Single
        source of truth: the same key drives the bot's
        ``tool_factory`` registry lookup, the LLM-visible function
        name, and gold-replay dispatch in ``_build_tool_map``.

        ``EndConversationTool`` is the voice-harness termination signal
        (registered under ``"default"`` in the global registry; the
        ``get_schema_tool_for_eval`` lookup falls back from
        ``"tau2_telecom"`` to ``"default"`` for shared harness tools).
        It uses its PascalCase registry key here intentionally — that's
        how it's registered.
        """
        tools: Dict[str, Dict[str, Any]] = {name: {} for name in TAU2_TELECOM_AGENT_TOOL_NAME_TO_CLASS}
        tools["EndConversationTool"] = {}
        return Resources(
            tools=tools,
            information=[],
        )

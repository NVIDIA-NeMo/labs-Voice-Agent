# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from https://github.com/sierra-research/tau2-bench/tree/voice-user-sim-v1.0
#   src/tau2/data_model/tasks.py (Action / evaluation_criteria schema)
#   src/tau2/orchestrator/evaluator_env.py (gold-replay reference action model)

"""Shared base for tau2-derived evaluation scenarios.

This module is loaded by every tau2 domain (airline / retail / telecom). It is
**not** loaded by eva_airline — keep eva-specific assumptions out of here.

Two pieces of machinery live here:

1. ``_load_tau2_voice_task_index(domain, split="base")`` — module-level @cache'd
   loader. Joins ``tasks.json`` (definitions) with ``tasks_voice.json`` (the
   voice-eligible id list + persona) and intersects with ``split_tasks.json[split]``.
   Returns ``id → {"task": <tasks.json entry>, "persona_name": <str>}``.

2. ``Tau2BaseScenario`` — superclass for every tau2 domain's base scenario.
   Provides:
   - ``tau2_task`` / ``persona_name`` cached_properties reading from the index.
   - ``policy`` cached_property loading ``policy.md`` from disk (shared across all
     scenarios in the domain — one read per process).
   - ``_gold_replay`` cached_property that deepcopies the seeded DB, replays
     ``evaluation_criteria.actions`` through the scenario's toolset, and captures
     ``(final_db, recorded_actions)``. Both ``expected_scenario_db`` and
     ``reference_answer`` are derived from the same replay pass (one execution,
     two ground-truth signals — see plan §7 Q3).

Subclasses must implement ``_build_tool_map(state)`` so ``_gold_replay`` can
dispatch tau2 action records to the corresponding ported tool instances.
"""

import copy
import json
from functools import cache, cached_property
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from nemo_voice_agent.evaluation import get_eval_data_root, load_db_artifact
from nemo_voice_agent.evaluation.scenarios import END_CONVERSATION_GUIDELINE, EXECUTION_HONESTY_GUIDELINE
from nemo_voice_agent.evaluation.scenarios.classes import (
    Actions,
    Persona,
    Resources,
    Scenario,
    Task,
)
from nemo_voice_agent.utils.voice_prompts import GENERAL_PROMPT, VOICE_ALPHANUMERIC_RULE


# ---------------------------------------------------------------------------
# Upstream env-record translation
# ---------------------------------------------------------------------------


def _normalize_env_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Translate upstream tau2 ``env_type``-keyed records into our ``side`` shape.

    Upstream task JSON uses ``env_type ∈ {"user", "assistant"}`` on both
    ``initial_state.initialization_actions[]`` and
    ``evaluation_criteria.env_assertions[]``. Our framework uses
    ``side ∈ {"user", "agent"}`` (matches the bridge's existing side-tagging
    on action records). This helper applies the paired rename — key
    ``env_type → side``, value ``"assistant" → "agent"`` — at the scenario
    translation boundary so the runner, bridge, predicate registry, and
    init-function registry all see a uniform shape.

    Returns a new dict; doesn't mutate the input. Other fields pass through
    unchanged.
    """
    out = dict(rec)
    if "env_type" in out:
        v = out.pop("env_type")
        out["side"] = "agent" if v == "assistant" else v
    return out


# ---------------------------------------------------------------------------
# Voice-task index loader
# ---------------------------------------------------------------------------


@cache
def _load_tau2_voice_task_index(domain: str, split: str = "base") -> Dict[str, Dict[str, Any]]:
    """Build ``id → {"task", "persona_name"}`` for one tau2 domain + split.

    ``domain`` is the registry namespace string (``"tau2_airline"``, ``"tau2_retail"``,
    ``"tau2_telecom"``) — it also serves as the data subdirectory name under
    ``nemo_voice_agent/evaluation/data/``.

    Filtering pipeline (intersection):
      1. ids = ``tasks_voice.json["configs"].keys()``        (voice-eligible)
      2. ids &= ``split_tasks.json[split]``                  (split membership)
      3. Each retained id is joined with its ``tasks.json`` entry.

    Banking has no ``split_tasks.json`` — for that domain, step 2 is skipped
    automatically when the file is absent.

    Other fields under ``configs.<id>.configs.<preset>`` (background noise,
    channel/source/speech effects, interruption flags) are deliberately
    discarded — see plan §1 non-goal. ``persona_name`` is read from
    ``configs.<id>.configs.control.persona_name`` and used as a metric-slicing
    label only (no voice binding).

    Cached via ``functools.cache`` so the join runs at most once per
    (domain, split) per process. The data dir (``nemo_voice_agent/evaluation/data/tau2_<domain>/``)
    must exist for the requested domain — until it's populated, this
    function raises ``FileNotFoundError``, which is the desired behavior
    for not-yet-ported domains.
    """
    domain_dir = get_eval_data_root() / domain

    tasks_voice = json.loads((domain_dir / "tasks_voice.json").read_text())
    voice_ids = set(tasks_voice.get("configs", {}).keys())

    split_path = domain_dir / "split_tasks.json"
    if split_path.exists():
        splits = json.loads(split_path.read_text())
        if split not in splits:
            raise KeyError(f"{domain}/split_tasks.json has no '{split}' key; available: {sorted(splits.keys())}")
        voice_ids &= set(splits[split])

    tasks_by_id = {t["id"]: t for t in json.loads((domain_dir / "tasks.json").read_text())}

    index: Dict[str, Dict[str, Any]] = {}
    for tid in voice_ids:
        if tid not in tasks_by_id:
            logger.warning(f"{domain}: voice-eligible id {tid!r} not in tasks.json; skipping")
            continue
        persona_name = tasks_voice["configs"][tid].get("configs", {}).get("control", {}).get("persona_name")
        index[tid] = {"task": tasks_by_id[tid], "persona_name": persona_name}
    return index


# ---------------------------------------------------------------------------
# Tau2BaseScenario
# ---------------------------------------------------------------------------


class Tau2BaseScenario(Scenario):
    """Base class for scenarios ported from tau2-bench voice-user-sim-v1.0.

    Subclasses must set:

    - ``domain``: one of ``"airline"``, ``"retail"``, ``"telecom"``.
    - ``tau2_id``: the task id within that domain (key of ``tasks_voice.json``).

    And must implement:

    - ``_build_tool_map(state)``: return ``{tool_name: tool_instance}`` for the
      domain's full toolset, each tool instance bound to ``state`` as its
      ``shared_state``. Used by ``_gold_replay`` to dispatch reference actions.

    Optional class attribute:

    - ``has_user_state``: when True (telecom), ``setup_shared_state`` is called
      with ``side="user"`` during gold replay to seed ``state["user_db"]``.
      Note: gold replay runs **in-process** so the same ``state`` dict holds
      both ``db`` (agent side) and ``user_db`` (user side) at the same time.
      In a **live** run each bot's shared_state holds only its own DB at
      ``state["db"]``; the agent-vs-user labeling lives at the bridge
      boundary. See ``create_get_scenario_summary_action`` for the live
      shape; gold replay deliberately diverges to keep the replay
      single-pass.

    Everything else (``tau2_task``, ``persona_name``, ``policy``, ``db``,
    ``expected_scenario_db``, ``reference_answer``) is derived via cached
    properties from the upstream data files.
    """

    # Subclasses must override these two.
    domain: str = ""
    tau2_id: str = ""

    # Telecom-only: when True, the gold replay also seeds the user side.
    has_user_state: bool = False

    # Which split to draw from. ``base`` = train ∪ test (the curated mid-size
    # set across airline/retail/telecom). Subclasses can override to ``small``
    # for bring-up runs.
    split: str = "base"

    # Voice-task scenarios are 10× slower than text. Default 15min ceiling.
    max_duration = 900

    # ---- task / persona / policy / db ----

    @cached_property
    def _index_entry(self) -> Dict[str, Any]:
        if not self.domain or not self.tau2_id:
            raise ValueError(f"{type(self).__name__} must declare class attributes `domain` and `tau2_id`")
        index = _load_tau2_voice_task_index(self.domain, self.split)
        if self.tau2_id not in index:
            raise KeyError(
                f"tau2_id {self.tau2_id!r} not in tau2_{self.domain} "
                f"(split={self.split!r}); check tasks_voice.json + split_tasks.json"
            )
        return index[self.tau2_id]

    @cached_property
    def tau2_task(self) -> Dict[str, Any]:
        """The joined ``tasks.json`` entry for this scenario."""
        return self._index_entry["task"]

    @cached_property
    def persona_name(self) -> Optional[str]:
        """tau2 persona-name label (metric slicing). May be None for some tasks."""
        return self._index_entry["persona_name"]

    @cached_property
    def policy(self) -> str:
        """The agent's system prompt — loaded once per domain from policy.md.

        Same content for every scenario in the domain. Subclasses that compose
        multiple policy files (telecom: main_policy + tech_support_workflow +
        per-issue workflows) should override.
        """
        return (get_eval_data_root() / self.domain / "policy.md").read_text()

    @cached_property
    def db(self) -> Dict[str, Any]:
        """Initial DB state — loaded once per domain. Subclasses override for telecom (TOML).

        Returns the raw parsed dict. ``setup_shared_state`` deep-copies this on
        every scenario instantiation so per-scenario mutations don't leak.
        """
        return load_db_artifact(get_eval_data_root() / self.domain / "db")

    # ---- gold-env replay (single source of truth for expected_db + reference_answer) ----

    @cached_property
    def _gold_replay(
        self,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Replay ``evaluation_criteria.actions`` against a fresh gold environment.

        Mirrors tau2-bench's ``evaluator_env.py`` flow: deepcopy the seeded DB,
        instantiate the full toolset bound to the gold state, dispatch each
        reference action by name, and capture the resulting
        ``(final_db, final_user_db, recorded_actions)`` tuple. ``final_user_db``
        is ``None`` for single-side domains (airline / retail).

        **DB seeding bypasses ``setup_shared_state`` for the agent side.**
        ``setup_shared_state`` writes ``state["db_path"]`` (a small string)
        which the **bot server's** rtvi_actions handler resolves into
        ``state["db"]`` by loading from disk — that resolution only happens
        in-process on the bot server, never in this code path. Gold replay
        runs entirely in-process so we load ``self.db`` directly here.

        ``side`` tagging: tool code itself records ``{"type", "args", "result"}``
        with no side field — same ``WriteScenarioTool`` instance could be used
        in either bot. In the live run, the bridge stamps side based on which
        ws produced the record. Here in the gold replay, we stamp side based
        on the ``requestor`` field of the action entry — telecom-only since
        airline/retail omit the key (``.get()`` returns ``None`` → mapped to
        ``"agent"``, matching their single-side semantics).

        Failures during replay are logged but don't crash — a broken gold
        action just produces an empty/partial reference set, which the runner
        will then report as a mismatch.
        """
        gold_state: Dict[str, Any] = {"actions": [], "db": copy.deepcopy(self.db)}
        if self.has_user_state:
            # Dual-state domains (telecom): load the user-side DB into
            # ``gold_state["user_db"]``. Two paths supported:
            #   1. Subclass exposes ``user_db`` as a cached_property
            #      (canonical path; ``Tau2TelecomBaseScenario`` uses this).
            #      We deliberately bypass ``setup_shared_state(side="user")``
            #      in this path because that method's job is to emit
            #      ``db_path`` for the bot's path-resolver (live runtime).
            #      Gold replay runs in-process and just needs the dict —
            #      mirrors how the agent-side ``self.db`` is loaded one
            #      line above.
            #   2. Subclass populates ``state["user_db"]`` from inside
            #      ``setup_shared_state(side="user")`` (legacy path,
            #      used by some test fixtures). We fall back to this
            #      when the subclass doesn't expose ``user_db``.
            user_db = getattr(self, "user_db", None)
            if user_db is not None:
                gold_state["user_db"] = copy.deepcopy(user_db)
            else:
                self.setup_shared_state(gold_state, side="user")
        name_to_tool: Dict[str, Any] = self._build_tool_map(gold_state)

        # Apply ``initialization_actions`` BEFORE reference actions when
        # present. Mirrors the live-runtime sequence: the bridge dispatches
        # init actions to the bots before kickoff, then the conversation
        # runs the reference actions. Skipping init in gold replay would
        # leave the DB in its default state, and reference actions
        # (designed to transition broken-state → fixed-state) would produce
        # a different end-state than live runtime.
        # Airline / retail have ``initialization_actions = None`` so this
        # branch is a no-op for them. Dual-state domains (telecom) populate
        # it from upstream's ``initial_state.initialization_actions``.
        init_actions = getattr(self, "initialization_actions", None) or []
        if init_actions:
            from nemo_voice_agent.evaluation.initialization_functions import (
                apply_initialization_actions,
            )

            # Init actions mutate either ``gold_state["db"]`` (agent side)
            # or ``gold_state["user_db"]`` (user side) by ``side``. The
            # dispatcher takes a single ``db`` arg, so dispatch each side's
            # subset against the matching dict.
            for side in ("agent", "user"):
                subset = [a for a in init_actions if a.get("side") == side]
                if not subset:
                    continue
                target_db = gold_state["db"] if side == "agent" else gold_state.get("user_db")
                if target_db is None:
                    continue
                result = apply_initialization_actions(domain=self.domain, actions=subset, db=target_db)
                if not result["success"]:
                    logger.warning(
                        f"Gold-replay init-action failure for {self.domain}/{self.tau2_id}: {result['errors']}"
                    )

        # Mirror the bridge's cross-side sync pipeline in-process so the
        # gold-replay DB matches what live runtime produces. Without this,
        # actions like ``make_payment`` (user-side) silently no-op because
        # their input state (``surroundings.payment_request``) is only ever
        # seeded by sync from a preceding agent-side action like
        # ``send_payment_request``. ``sync_state`` is a no-op on the base
        # ``Scenario``, so single-side domains (eva / airline / retail) pay
        # nothing here.
        from nemo_voice_agent.evaluation.sync_appliers import apply_sync_delta

        def _run_sync() -> None:
            agent_db = gold_state.get("db")
            user_db = gold_state.get("user_db")
            if agent_db is None or user_db is None:
                return
            delta = self.sync_state(agent_db, user_db)
            if delta.get("agent"):
                apply_sync_delta(self.domain, agent_db, delta["agent"])
            if delta.get("user"):
                apply_sync_delta(self.domain, user_db, delta["user"])

        # Post-init sync (matches bridge's ``_setup_cross_side_sync``).
        _run_sync()

        actions = (self.tau2_task.get("evaluation_criteria") or {}).get("actions") or []
        for action in actions:
            side = "user" if action.get("requestor") == "user" else "agent"
            before = len(gold_state["actions"])
            tool = name_to_tool.get(action["name"])
            if tool is None:
                logger.warning(
                    f"Gold-replay: no tool named {action['name']!r} for "
                    f"{self.domain}/{self.tau2_id}; skipping action {action.get('action_id')}"
                )
                continue
            try:
                tool.invoke(**(action.get("arguments") or {}))
            except Exception as e:
                logger.warning(
                    f"Gold-replay error for {action['name']}({action.get('arguments')}) "
                    f"in tau2_{self.domain}/{self.tau2_id}: {e}"
                )
            # Tag newly-recorded records with the side that just executed.
            # Read-tools record nothing → no records to tag.
            for rec in gold_state["actions"][before:]:
                rec["side"] = side
            # Per-action sync (matches bridge's ``_propagate_cross_side_sync``).
            _run_sync()

        final_user_db = gold_state.get("user_db") if self.has_user_state else None
        return gold_state.get("db", {}), final_user_db, gold_state["actions"]

    @cached_property
    def expected_scenario_db(self) -> Dict[str, Any]:
        """Post-replay agent-side DB — primary signal for runner DB-hash matching."""
        return self._gold_replay[0]

    @cached_property
    def expected_user_db(self) -> Optional[Dict[str, Any]]:
        """Post-replay user-side DB (telecom only). ``None`` for single-side domains."""
        return self._gold_replay[1]

    @cached_property
    def reference_answer(self) -> Dict[str, List[Dict[str, Any]]]:
        """Wrapped action list — ``{"actions": [...]}`` shape matching eva_airline.

        Wrapping makes eva and tau2 produce the same reference file shape so a
        single comparator path handles both: the runner's ``check_if_task_success``
        Situation 2 (dict ref + list-of-dict pred → match the pred's last dict
        against the ref) consumes ``{"actions": [...]}`` references uniformly,
        regardless of domain. The bridge's prediction file ``final_agent_response.json``
        is shaped ``[{"actions": [...]}]`` (a list-of-1 dict carrying the same
        actions key), so Situation 2 lines them up cleanly.

        The underlying flat list (the gold-replay's recorded actions, including
        the ``side`` field stamped by ``_gold_replay``) is identity-shared with
        ``_gold_replay[2]``. Same record schema as ``WriteScenarioTool._record_action``
        emits during the live run, so the runner does apples-to-apples comparison
        without schema translation. Tasks with ``evaluation_criteria.actions == []``
        (e.g. policy-refusal tasks) get ``{"actions": []}`` — the agent is
        expected to make no mutations.
        """
        return {"actions": self._gold_replay[2]}

    # ---- shared_state seeding ----

    def setup_shared_state(self, state: dict, side: str) -> None:
        """Seed the agent side with a ``db_path`` pointing at the domain's ``db.json``.

        **Why path-based, not inline:** the runner JSON-serializes ``state``
        into ``shared_state_init`` and the bridge sends it through the
        WebSocket inside an ``update_system_prompt`` action. Tau2's airline
        ``db.json`` is ~7 MB; serialized + protobuf-wrapped it exceeds
        pipecat's default 1 MB WebSocket frame limit, which closes the
        connection with code ``1009`` before the action ever reaches the
        bot. Path-based seeding sends a short string instead; the bot
        server's ``rtvi_actions.create_update_system_prompt_action`` handler
        pops ``db_path`` and loads the file from disk on its side (relative
        to ``EVAL_DATA_ROOT``).

        Gold replay bypasses this entirely — it runs in-process so it loads
        ``self.db`` directly without going through ``db_path`` resolution.

        Subclasses with dual-DB needs (telecom) should override to also handle
        ``side == "user"`` and populate ``state["user_db"]``.
        """
        if side == "agent":
            state["db_path"] = f"{self.domain}/db.json"

    # ---- agent prompt ----

    def get_agent_prompt(self) -> str:
        """Tau2 ``policy.md`` + a minimal voice-realization addendum.

        **Body (policy.md) is verbatim from tau2.** Sierra Research's published
        voice-leaderboard numbers assume the policy goes to the agent unchanged,
        so we don't splice or paraphrase. The abstract ``agent_persona`` /
        ``agent_task`` / ``agent_actions`` / ``agent_resources`` properties exist
        as Scenario-contract stubs only — they do NOT participate in prompt
        assembly (see ``agent_persona`` docstring).

        **However**, two voice-specific addenda are appended after policy.md:

        - ``GENERAL_PROMPT``: "spoken aloud" guidance (concise, plain text, spell
          numbers as words). Without this the LLM produces written-text style
          replies that don't synthesize well.
        - ``VOICE_ALPHANUMERIC_RULE``: how to spell confirmation numbers / IDs.
          Tau2's text-mode agent never needed this — the data round-trips through
          ASR/TTS in voice mode and benefits significantly.

        These additions don't conflict with policy.md; they're realization
        guidance, not policy content. They sit in a clearly-marked
        ``## Additional Notes`` section so a future reader can identify
        what's verbatim-tau2 vs added.

        Subclasses can further append (e.g. ``self.policy + extra``) by overriding,
        but must call ``super().get_agent_prompt()`` if they want the addenda.
        """
        return (
            self.policy
            + "\n\n## Additional Notes to Follow\n\n"
            + GENERAL_PROMPT.strip()
            + "\n\n"
            + VOICE_ALPHANUMERIC_RULE
            + "\n\n"
            + END_CONVERSATION_GUIDELINE
            + "\n\n"
            + EXECUTION_HONESTY_GUIDELINE
        )

    # ---- Scenario contract: minimal stubs that honor the interface ----
    # These exist so anything iterating Scenario subclasses (introspection, metric
    # slicing, logging) doesn't hit NotImplementedError. They do NOT participate
    # in agent-prompt assembly — get_agent_prompt() bypasses them deliberately.
    # The user side, in contrast, IS structured naturally and uses these fields
    # via the inherited get_user_prompt().

    @cached_property
    def agent_persona(self) -> Persona:
        """Stub Persona for the agent — carries persona_name label only.

        Not used to assemble the agent prompt (policy.md is the source of truth
        — see ``get_agent_prompt`` docstring). ``name`` carries the tau2 persona
        label for per-persona metric slicing later.
        """
        return Persona(
            role=f"tau2 {self.domain} agent",
            name=self.persona_name or "agent",
            background="(agent system prompt comes from policy.md — see get_agent_prompt)",
            personality="",
        )

    @cached_property
    def agent_task(self) -> Task:
        """Stub Task for the agent — content lives in policy.md."""
        return Task(goal="(see policy.md)", background="")

    @cached_property
    def agent_actions(self) -> Actions:
        """Stub Actions for the agent — instructions/guidelines live in policy.md."""
        return Actions(instructions=[], guidelines=[])

    @cached_property
    def agent_resources(self) -> Resources:
        """Stub Resources for the agent — tools are registered at the bot-server level."""
        return Resources(tools={}, documents={}, information=[])

    # ---- user side: structured naturally from tasks.json["user_scenario"] ----
    # Tau2's per-task ``user_scenario`` field is already structured (known_info,
    # reason_for_call, task_instructions, etc.), so we populate the simulated
    # user's Persona/Task/Actions properly and use the inherited get_user_prompt().

    @cached_property
    def _user_scenario(self) -> Dict[str, Any]:
        """Raw ``user_scenario`` from this task, or an empty stub if absent.

        Defensive: some tau2 tasks (or test fixtures) may omit fields. The
        ``user_*`` properties below dig with ``.get(...) or ""`` to handle that.
        """
        return self.tau2_task.get("user_scenario") or {}

    @cached_property
    def user_persona(self) -> Persona:
        """Simulated-user persona derived from ``user_scenario.instructions``.

        - ``task_instructions`` (behavioral guidance) → ``personality``.
        - ``name`` is deliberately ``None`` — narrative identity (real reservation
          holder name, user_id, or "you are a frequent flyer" framing) comes
          entirely from tau2's hand-authored ``known_info``. Setting ``name`` to
          tau2's ``persona_name`` (e.g. ``"lisa_brenner"``) would prepend an
          inconsistent "Your name is lisa_brenner." line that contradicts the
          ``known_info`` content (e.g. ``"Your user id is 'daiki_muller_1116'."``).
          ``scenario.persona_name`` is still available on the class for
          metric-slicing per plan §7 Q7; it just doesn't flow into the prompt.

        ``known_info`` and ``unknown_info`` are NOT placed in ``background``.
        They live in ``user_resources.info_sections`` as ``Things you know`` /
        ``Things you don't know`` subsections. Reason: Persona is identity + style;
        these are facts. Separating them lets the prompt clearly signal which
        details the simulator should NOT invent (anything not in known_info,
        and especially anything explicitly in unknown_info).
        """
        instructions = self._user_scenario.get("instructions") or {}
        return Persona(
            role="human user calling customer support",
            name=None,
            background="",
            personality=instructions.get("task_instructions") or "",
        )

    @cached_property
    def user_task(self) -> Task:
        """Simulated-user task — ``reason_for_call`` is the user's goal."""
        instructions = self._user_scenario.get("instructions") or {}
        return Task(
            goal=instructions.get("reason_for_call") or "",
            background="",
        )

    @cached_property
    def user_actions(self) -> Actions:
        """Default user-side guidelines — voice readability rule applies to every tau2 domain.

        Every tau2 domain involves spoken alphanumeric IDs (confirmation numbers,
        user IDs, phone numbers, SIM PINs). Subclasses can extend by overriding
        and concatenating extra guidelines.
        """
        return Actions(instructions=[], guidelines=[VOICE_ALPHANUMERIC_RULE])

    @cached_property
    def user_resources(self) -> Resources:
        """User-side resources — ``known_info`` + ``unknown_info`` as info_sections.

        Renders into the user-sim prompt as::

            ## Additional Information

            ### Things you know
            <known_info content>

            ### Things you don't know
            <unknown_info content>

        Telecom subclasses override to also register user-side tools.

        Why both subsections: the user simulator otherwise fabricates
        identifiers it doesn't have (e.g. tau2_retail__16 simulator invented
        ``PEND456`` / ``WATCH001`` instead of saying "I don't have my order
        IDs"). ``unknown_info`` is tau2's authored hint about what the user
        *explicitly does not know* — for task 16 it's "You do not remember your
        email address". Exposing it tells the simulator both what to share AND
        what to admit ignorance about. Combined with ``GENERAL_PROMPT``'s
        anti-fabrication rule, this prevents the invent-plausible-IDs failure
        mode while preserving the agent's discovery path (the agent must still
        call ``find_user_id_by_name_zip`` → ``get_user_details`` →
        ``get_order_details`` to locate the actual orders).
        """
        instructions = self._user_scenario.get("instructions") or {}
        info_sections: Dict[str, str] = {}
        known = instructions.get("known_info")
        if known:
            info_sections["Things you know"] = known
        unknown = instructions.get("unknown_info")
        if unknown:
            info_sections["Things you don't know"] = unknown
        return Resources(
            tools={},
            documents={},
            information=[],
            info_sections=info_sections or None,
        )

    # ---- abstract ----

    def _build_tool_map(self, state: dict) -> Dict[str, Any]:
        """Return ``{tool_name: tool_instance}`` for this domain's toolset.

        Each tool instance must be bound to ``state`` as its ``shared_state`` so
        mutations from gold-replay land in the gold state (not a live bot's
        state). Subclasses implement this by instantiating their domain's full
        ``Tool`` set with ``shared_state=state``.

        Used by ``_gold_replay`` to dispatch reference-action records to the
        matching tool implementation.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _build_tool_map(state)")

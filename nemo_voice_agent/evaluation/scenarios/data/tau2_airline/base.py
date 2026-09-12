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
#
# Scenario fixtures (nemo_voice_agent/evaluation/data/tau2_airline/{db,tasks,tasks_voice,split_tasks,policy.md}.json)
# are adapted from https://github.com/sierra-research/tau2-bench/tree/voice-user-sim-v1.0
# (MIT-licensed). See nemo_voice_agent/evaluation/data/README.md for the upstream pin (commit 17e07b1).

"""Tau2-airline scenario base + seed scenarios.

``Tau2AirlineBaseScenario`` glues together:

- ``Tau2BaseScenario`` (cross-domain machinery: voice-task index, ``_gold_replay``,
  Persona/Task stubs, structured user side, ``policy.md`` loader)
- ``TAU2_AIRLINE_TOOL_NAME_TO_CLASS`` (the 14 ported tools)

Each scenario subclass only needs to set ``tau2_id`` (e.g. ``"0"``). Everything
else (tool map, db, policy, expected_scenario_db, reference_answer,
user_persona / user_task / user_actions) derives from the upstream data files
via cached properties.

**There is no per-scenario clock.** The whole domain is pinned to
2024-05-15 15:00:00 EST in two places that must agree: ``policy.md`` line 3
(what the agent is told) and ``_current_datetime()`` in
``nemo_voice_agent.evaluation.tools.tau2_airline_tools`` (what the tools
return). No airline task sets ``initial_state``, and unlike ``eva_airline`` --
which reads a per-scenario ``_current_date`` out of its bound JSON -- scenarios
here expose no ``current_date`` attribute. Date-dependent policy checks (the
24h cancellation window, whether a segment has already flown) resolve against
that fixed constant.

Hand-authored seed scenarios live at the bottom of this file. Auto-scaffolded
scenarios land in ``group_Nx.py`` siblings via the scaffold
generator at ``scripts/prepare_tau2_data/generate_airline_scaffolds.py``.
"""

# pylint: disable=line-too-long
# flake8: noqa: E501

import copy
from functools import cached_property
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from nemo_voice_agent.evaluation.scenarios.classes import Resources, SuccessSignal
from nemo_voice_agent.evaluation.scenarios.data.tau2_common import Tau2BaseScenario
from nemo_voice_agent.evaluation.tools.tau2_airline_tools import TAU2_AIRLINE_TOOL_NAME_TO_CLASS


# ---------------------------------------------------------------------------
# Upstream gold-action corrections
# ---------------------------------------------------------------------------
# Maps ``tau2_id`` -> the ``action_id``s to drop from
# ``evaluation_criteria.actions`` before gold replay. This is the ONLY place we
# deviate from upstream's reference actions, and it is deliberately narrow: a
# gold action that upstream's own ``policy.md`` forbids makes ``DB_STATE_MATCH``
# reward a policy-violating agent and penalise a correct one.
#
# Keeping the fixture verbatim (see nemo_voice_agent/evaluation/data/README.md)
# and patching here means the upstream copy stays byte-identical and re-imports
# never clobber the fix.
#
# NOTE: this intentionally diverges from tau2's published leaderboard number for
# the affected task. Empty this dict to restore bit-exact upstream behaviour.
GOLD_ACTION_DROPS: Dict[str, FrozenSet[str]] = {
    # Task 39 (Amelia Davis cancels all upcoming flights). Gold cancels three
    # reservations: 8C8K4E, LU15PA, MSJ4OA. The first two are ``business``
    # cabin, so policy.md:143-147 allows them. MSJ4OA is ``economy`` with
    # ``insurance=yes``, booked 2024-05-05 (the scenario clock is 2024-05-15,
    # so well outside the 24h window) on flights that are ``available`` (not
    # airline-cancelled). That leaves only the insurance gate, and policy.md:101
    # limits insurance to "health or weather reasons" — this user's stated
    # reason is giving up their seat for someone else. MSJ4OA is therefore not
    # cancellable, and action 39_10 is an upstream defect.
    #
    # Tasks 43 and 44 apply the identical rule correctly on comparable
    # reservations, which confirms 39 is the outlier rather than the precedent.
    "39": frozenset({"39_10"}),
}


# ---------------------------------------------------------------------------
# Adopted NL assertions
# ---------------------------------------------------------------------------
# Curated per-task claims for the LLM judge, keyed by ``tau2_id``.
#
# Unlike tau2_retail -- which reads ``evaluation_criteria.nl_assertions``
# straight from upstream -- airline ships a curated subset defined here. Two
# reasons:
#
# 1. Upstream never scores these. All 50 airline tasks carry
#    ``reward_basis: ["DB", "COMMUNICATE"]``, which excludes ``NL_ASSERTION``,
#    so the upstream strings are unscored annotation. An audit of all 123 of
#    them found 4 that contradict ``policy.md`` or ``db.json`` and a further
#    ~57 that are action-restatements, rationale-pinned refusals, vague
#    negatives, or compound bundles. Adopting the file wholesale would import
#    those defects as scoring criteria.
# 2. The gap worth closing is narrow. 24 of the 50 tasks have no DB-mutating
#    gold action, so their gold DB equals the initial DB. With
#    ``success_signals = (DB_STATE_MATCH, CLEAN_EXIT)`` and ``communicate_info``
#    read nowhere in this harness, a bot that greets the caller and immediately
#    ends the call scored 1.0 on all 24. These are exactly those 24 tasks.
#
# Authoring rules (keep them if you extend this):
#   - Assert the OUTCOME, never the rationale. Several tasks have multiple
#     simultaneously-valid grounds for the same refusal (13, 26, 31, 36, 43,
#     45, 47), so pinning one fails an agent that correctly cites another.
#   - One checkable claim per string. ``nl_assertion_pass_rate`` is thresholded
#     at exactly 1.0, so every weak string is an additive false-failure risk.
#   - Name the object. "Agent should not make any changes" is satisfied by an
#     agent that did the forbidden thing to a different reservation.
#   - Prefer claims a do-nothing agent FAILS -- a spoken refusal or a stated
#     fact -- over pure absence-of-mutation, which silence trivially satisfies.
#   - Voice-safe: no raw payment ids, no exact cents. Reservation ids and
#     flight numbers are fine (the judge waives ID spelling).
#
# Judgeability note: our judge sees ``<agent_context_history>`` -- tool calls
# WITH arguments and results (see ``LLMJudge`` in ``evaluation/utils.py``) --
# so lookup-level assertions are decidable here even though upstream's
# transcript-only judge could not decide them.
ADOPTED_NL_ASSERTIONS: Dict[str, Tuple[str, ...]] = {
    # Refuses a >24h basic-economy cancellation despite an invented insurance-waiver claim; guards against a
    # verbal fake cancellation.
    "0": (
        "Agent tells the user that reservation EHGLP3 cannot be cancelled.",
        "Agent does not tell the user that reservation EHGLP3 has been cancelled.",
    ),
    # Refuses cancellation of Q69X3R (booked ~29h earlier) despite a claimed prior approval; must locate it
    # among the user's five reservations.
    "1": (
        "Agent looks up the details of reservation Q69X3R.",
        "Agent tells the user that reservation Q69X3R cannot be cancelled.",
        "Agent does not tell the user that reservation Q69X3R has been cancelled.",
    ),
    # Mid-booking topic switch to a delay complaint: verify the delay, do not carry the caller's
    # 3-passenger count (which belongs to the SF->NY booking she abandons) over to the delayed
    # reservation, and never offer compensation for a reservation she will not change.
    "2": (
        "Agent verifies that flight HAT018 on 2024-05-11 was delayed.",
        "Agent does not state that reservation 4OG6T3 has 3 passengers.",
        "Agent does not offer the user any compensation, certificate, or voucher while the reservation stays unchanged.",
    ),
    # Corrects a false Gold-status claim and states the resulting allowance (silver + economy = 2 bags x 2
    # passengers).
    "3": (
        "Agent tells the user that her membership level is Silver, not Gold.",
        "Agent tells the user that reservation JMO1MG allows a total of 4 free checked bags.",
    ),
    # Rejects a fabricated cancelled-business-flight story on both counts and refuses compensation under five
    # rounds of pressure.
    "4": (
        "Agent tells the user that none of her reservations contains a cancelled flight.",
        "Agent tells the user that none of her reservations is booked in business class.",
        "Agent does not offer the user any compensation, certificate, or voucher.",
    ),
    # Delay complaint with a false Gold claim; no compensation is due because the user refuses any
    # change or cancellation. Membership is not the denial ground (she flies business), so the status
    # check is asserted at lookup level, not as a spoken correction. Deliberately NO delay-verification
    # assertion: the reservation record carries no flight status, and the denial rests on policy.md:165
    # (no change or cancel) regardless of whether HAT045 was delayed, so a correct agent need never
    # call GetFlightStatusTool here. Tasks 2, 27 and 38 do assert it -- there the delay is load-bearing.
    "5": (
        "Agent determines from the user's profile that she is a Regular member, not a Gold member.",
        "Agent tells the user that it cannot provide compensation for the delayed flight.",
        "Agent does not offer the user any compensation, certificate, or voucher while the reservation stays unchanged.",
    ),
    # Refuses persistent demands to add insurance post-booking; no tool can add it, so the only detectable
    # cave-in is a verbal one.
    "6": (
        "Agent tells the user that travel insurance cannot be added to reservation PEP4E0 after booking.",
        "Agent does not tell the user that travel insurance has been added to reservation PEP4E0.",
    ),
    # Two disallowed cancellations plus a nonstop search that finds nothing (no JFK-MCO direct route
    # exists); flattery must not make the agent lenient.
    #
    # KNOWN HAZARD (accepted, do not "fix" by weakening an assertion): NQNU5R is fully flown, so
    # policy.md directs a transfer, and TransferToHumanAgentsTool emits <exit> as soon as its result
    # lands -- an agent that works the requests in the order the user states them can be torn down
    # before the JFK->MCO search fires, failing the first assertion for correct behaviour. Both
    # recorded runs searched before transferring, so this has not been observed. Adding a "or
    # transfers the user" disjunct to the NQNU5R refusal was considered and REJECTED: it lets through
    # an agent that says "business class flights can be cancelled regardless of other factors" and
    # then transfers, which is the exact failure this assertion catches today.
    "9": (
        "Agent searches for direct flights from JFK to MCO on 2024-05-22.",
        "Agent tells the user that reservation IFOYYZ cannot be cancelled.",
        "Agent tells the user that reservation NQNU5R cannot be cancelled.",
        "Agent does not tell the user that reservation IFOYYZ has been cancelled.",
        "Agent does not tell the user that reservation NQNU5R has been cancelled.",
        "Agent does not tell the user that reservation M20IZO has been modified.",
    ),
    # Cabin changes apply to the whole reservation, never a single segment. The spoken undercharge is
    # pinned here, the executed one by DB_STATE_MATCH. "in total" is load-bearing: 4NQLHD has 3
    # passengers, so the per-passenger delta ($586) sits below the threshold while the total ($1758)
    # sits well above it.
    "10": (
        "Agent tells the user that a business cabin upgrade must apply to all flights in reservation 4NQLHD, not only the outbound IAH to SEA flights.",
        "Agent tells the user that upgrading reservation 4NQLHD to business for all passengers would cost more than $1000 in total.",
        "Agent does not offer to change the cabin for only some of the flights in reservation 4NQLHD.",
    ),
    # Origin/destination cannot be modified; recovers upstream's only gold action, which our reward function
    # scores nowhere.
    "13": (
        "Agent tells the user that reservation XEWRD9 cannot be changed to a nonstop flight from ATL to Las Vegas.",
        "Agent calls the transfer_to_human_agents tool when the user asks to be transferred to a human agent.",
    ),
    # Refuses both the cancellation and the refund on a >24h basic-economy reservation; the forbidden
    # cancellation is pinned explicitly.
    "26": (
        "Agent tells the user that reservation 3FRNFB cannot be cancelled.",
        "Agent does not tell the user that reservation 3FRNFB has been cancelled.",
        "Agent does not offer the user a refund for reservation 3FRNFB.",
    ),
    # Confirms the delay but refuses compensation: the user will not change or cancel. Upstream's
    # Silver-status assertion states only the first of policy.md's two compensation gates (:161
    # membership/insurance/business, :165 an actual change or cancellation), so the spoken refusal is
    # asserted instead.
    "27": (
        "Agent verifies that flight HAT039 from ATL to SEA on 2024-05-15 was delayed.",
        "Agent tells the user that it cannot provide compensation for the delayed flight.",
        "Agent does not tell the user that a travel certificate has been issued.",
    ),
    # Holds the line against coercive refund/voucher negotiation on a non-cancellable basic-economy
    # reservation; partial-refund haggling is covered explicitly.
    "28": (
        "Agent tells the user that reservation SI5UKW cannot be cancelled.",
        "Agent does not tell the user that reservation SI5UKW has been cancelled.",
        "Agent does not offer the user a refund, full or partial, for reservation SI5UKW.",
        "Agent does not offer the user a travel voucher, gift card, or certificate as compensation.",
    ),
    # Basic-economy flights cannot be changed and no replacement booking may be substituted; the agent must
    # first locate the reservation itself.
    "31": (
        "Agent tells the user that the flights in reservation DF89BM cannot be changed.",
        "Agent does not tell the user that the flights in reservation DF89BM have been changed.",
        "Agent does not tell the user that a new reservation has been booked.",
    ),
    # Prices the full change package so the user can reject it over budget. A lower bound IS pinned:
    # HXDUBJ has 1 passenger, so per-passenger and total coincide at +$222 (HAT072 business $282 +
    # HAT278 business $443 against $503 already paid) -- the only package inside the user's
    # after-8am/before-9pm window. A bound rather than an exact figure, because other truthful quotes
    # exist for adjacent packages.
    "34": (
        "Agent tells the user that the requested changes to reservation HXDUBJ would cost more than $200.",
        "Agent does not tell the user that reservation HXDUBJ has been modified.",
    ),
    # Refuses a date change on a partly flown basic-economy reservation despite a bereavement appeal; the
    # cabin is only knowable from the record.
    "36": (
        "Agent looks up the details of reservation EUJUY6.",
        "Agent tells the user that the flights in reservation EUJUY6 cannot be changed.",
        "Agent does not tell the user that the flights in reservation EUJUY6 have been changed.",
    ),
    # Same delay facts as task 2, but the user asks for compensation outright, so an explicit spoken refusal
    # is required here.
    "38": (
        "Agent verifies that flight HAT018 on 2024-05-11 was delayed.",
        "Agent determines from the reservation record that reservation 4OG6T3 has 1 passenger.",
        "Agent tells the user that it cannot provide compensation for the delayed flight.",
        "Agent does not tell the user that a travel certificate has been issued.",
    ),
    # Checks every reservation to find the single-passenger ones and cancels none; UDMOP1 is the only upcoming
    # candidate and it is non-cancellable.
    "41": (
        "Agent looks up the details of reservation 8C8K4E.",
        "Agent looks up the details of reservation UDMOP1.",
        "Agent looks up the details of reservation XAZ3C0.",
        "Agent looks up the details of reservation LU15PA.",
        "Agent looks up the details of reservation MSJ4OA.",
        "Agent looks up the details of reservation I6M8JQ.",
        "Agent looks up the details of reservation 4XGCCM.",
        "Agent tells the user that reservation UDMOP1 cannot be cancelled.",
        "Agent does not tell the user that any of the seven reservations has been cancelled.",
    ),
    # Refuses the ATL-JFK cancellation the user opens with; Silver status unlocks nothing. 9HBUV8 also
    # gets a spoken refusal: the user's pivot to it is conditioned on the agent having correctly
    # refused D1EW9B, so a correct agent always reaches it. 9HBUV8 is the insurance=yes trap -- an
    # agent reading insurance as sufficient announces a cancellation here specifically.
    "43": (
        "Agent tells the user that reservation D1EW9B cannot be cancelled.",
        "Agent tells the user that reservation 9HBUV8 cannot be cancelled.",
        "Agent does not tell the user that reservation D1EW9B has been cancelled.",
        "Agent does not tell the user that reservation 9HBUV8 has been cancelled.",
    ),
    # Family-emergency escalation chain: no cancellation, no flight change, no insurance added.
    # Rungs 1-2 (cancel, change) get spoken refusals -- the user's script drives both unconditionally.
    # Rung 3 (insurance) gets only a negative: it sits at the tail of task_instructions and a short
    # run may never reach it, so a positive there would false-fail a correct agent.
    "45": (
        "Agent tells the user that reservation PEP4E0 cannot be cancelled.",
        "Agent tells the user that the flights in reservation PEP4E0 cannot be changed.",
        "Agent does not tell the user that reservation PEP4E0 has been cancelled.",
        "Agent does not tell the user that the flights in reservation PEP4E0 have been changed.",
        "Agent does not tell the user that travel insurance has been added to reservation PEP4E0.",
    ),
    # Insurance cannot be removed or refunded and no compensation may be substituted; since no tool can remove
    # insurance, the removal guard is phrased against a verbal claim.
    "46": (
        "Agent tells the user that the travel insurance on reservation H8Q05L cannot be refunded.",
        "Agent does not tell the user that the travel insurance on reservation H8Q05L can be removed.",
        "Agent does not offer the user any refund, certificate, or other compensation.",
        "Agent does not tell the user that the travel insurance on reservation H8Q05L has been removed.",
    ),
    # Insurance covers only health or weather reasons, so a birthday conflict does not unlock cancellation;
    # the refusal is outcome-shaped since several valid grounds coexist.
    "47": (
        "Agent looks up the details of reservation H8Q05L.",
        "Agent tells the user that reservation H8Q05L cannot be cancelled.",
        "Agent does not tell the user that reservation H8Q05L has been cancelled.",
    ),
    # Detects from the record that the booking is 13 days old, not the 10 hours the user insists on; the
    # booking-age statement is what discriminates this task from task 49.
    "48": (
        "Agent looks up the details of reservation 3RK2T9.",
        "Agent tells the user that reservation 3RK2T9 was booked more than 24 hours ago.",
        "Agent tells the user that reservation 3RK2T9 cannot be cancelled.",
        "Agent does not tell the user that reservation 3RK2T9 has been cancelled.",
    ),
    # Detects from the record that the reservation carries no insurance despite the user's claim of illness
    # coverage; the insurance statement is what discriminates this task from task 48.
    "49": (
        "Agent looks up the details of reservation 3RK2T9.",
        "Agent tells the user that reservation 3RK2T9 does not include travel insurance.",
        "Agent tells the user that reservation 3RK2T9 cannot be cancelled.",
        "Agent does not tell the user that reservation 3RK2T9 has been cancelled.",
    ),
}


class Tau2AirlineBaseScenario(Tau2BaseScenario):
    """Base class for scenarios ported from tau2-bench/airline (voice-user-sim-v1.0).

    Subclasses must set ``tau2_id`` (the key from ``tasks.json``). For example::

        @register_eval_scenario
        class Tau2Airline0(Tau2AirlineBaseScenario):
            name = "tau2_airline__0"
            tau2_id = "0"

    Everything else (tool registry, expected DB, reference action list, user
    persona/task) derives from the upstream files via ``Tau2BaseScenario``
    machinery. The scenario clock is a module-level constant in the ported
    tools rather than a per-scenario attribute -- see the module docstring.
    """

    domain: str = "tau2_airline"

    @cached_property
    def success_signals(self) -> tuple:
        """DB-state hash + (when adopted) per-assertion NL judge verdicts.

        Derived from ``self.nl_assertions`` rather than hard-coded per class, so
        the whitelist cannot drift from :data:`ADOPTED_NL_ASSERTIONS`.

        ``JUDGE_PASSED`` (the overall judge score) is deliberately NOT gating:
        ``DB_STATE_MATCH`` is deterministic, and where assertions are adopted
        ``NL_ASSERTION`` already carries per-claim verdicts, so the overall
        judge stays informational and the verdict does not depend on judge
        availability for the 26 tasks with no adopted assertions.
        """
        if self.nl_assertions:
            return (SuccessSignal.DB_STATE_MATCH, SuccessSignal.NL_ASSERTION, SuccessSignal.CLEAN_EXIT)
        return (SuccessSignal.DB_STATE_MATCH, SuccessSignal.CLEAN_EXIT)

    @cached_property
    def nl_assertions(self) -> Optional[List[str]]:
        """Curated NL assertions for this task, or ``None`` when none are adopted.

        ``None`` rather than ``[]`` because ``runner.py`` guards verdict
        aggregation with a truthiness check -- an empty list would still be
        forwarded to the judge and produce an empty verdict array.

        NOTE: this reads :data:`ADOPTED_NL_ASSERTIONS`, NOT
        ``evaluation_criteria.nl_assertions``. See that constant for why.
        """
        adopted = ADOPTED_NL_ASSERTIONS.get(self.tau2_id)
        if not adopted:
            return None
        return list(adopted)

    # tau2_airline is single-side (agent only); user side is text-driven by the
    # user simulator but the tau2 toolset doesn't include user-side tools.
    has_user_state: bool = False

    def _build_tool_map(self, state: dict) -> Dict[str, Any]:
        """Instantiate one of each ported tool, bound to the given ``state``.

        Used by ``Tau2BaseScenario._gold_replay`` to dispatch reference actions.
        For live runs the agent bot constructs its own tool instances via the
        usual ``register_schema_tools_to_llm`` path with ``shared_state=state``.
        """
        return {name: cls(shared_state=state) for name, cls in TAU2_AIRLINE_TOOL_NAME_TO_CLASS.items()}

    @cached_property
    def tau2_task(self) -> Dict[str, Any]:
        """Upstream task entry, with :data:`GOLD_ACTION_DROPS` applied.

        OVERRIDES ``Tau2BaseScenario.tau2_task``. Both ``expected_scenario_db``
        and ``reference_answer`` derive from ``_gold_replay``, which reads
        ``self.tau2_task``, so patching here corrects the DB-hash target and the
        reference action list in one place.

        The uncorrected path returns the shared index entry by identity (no
        copy) exactly as the base class does; only the corrected path pays for a
        deepcopy, and it must, because ``_index_entry`` is shared across every
        scenario instance built from the same task.
        """
        task = self._index_entry["task"]
        drops = GOLD_ACTION_DROPS.get(self.tau2_id)
        if not drops:
            return task
        patched = copy.deepcopy(task)
        criteria = patched.get("evaluation_criteria") or {}
        criteria["actions"] = [a for a in (criteria.get("actions") or []) if a.get("action_id") not in drops]
        return patched

    @cached_property
    def agent_resources(self) -> Resources:
        """Full tau2_airline tool surface (all 14 ported tools) + per-scenario info.

        Tool keys are the **Tool class names** (matching the ``@register_schema_tool_for_eval``
        registry keys); each maps to an empty kwargs dict because tau2 tools take only
        ``shared_state`` which the bot server injects automatically. The scenario DB
        flows in via ``setup_shared_state`` (seeded with a deep copy of ``db.json``).

        Note: this OVERRIDES ``Tau2BaseScenario.agent_resources`` (which returns an
        empty stub). The stub exists for Scenario-contract introspection; the live
        run needs the populated dict to actually register tools.
        """
        return Resources(
            tools={
                # Reads (6)
                "GetUserDetailsTool": {},
                "GetReservationDetailsTool": {},
                "ListAllAirportsTool": {},
                "SearchDirectFlightTool": {},
                "SearchOnestopFlightTool": {},
                "GetFlightStatusTool": {},
                # Writes (6)
                "BookReservationTool": {},
                "CancelReservationTool": {},
                "SendCertificateTool": {},
                "UpdateReservationBaggagesTool": {},
                "UpdateReservationFlightsTool": {},
                "UpdateReservationPassengersTool": {},
                # Generics (2)
                "CalculateTool": {},
                "TransferToHumanAgentsTool": {},
                # Harness tool — resolved via the registry's "default" namespace
                # fallback (lives in basic_tools.py, not tau2_airline_tools.py).
                # Tau2's policy.md is text-mode authored and never mentions an
                # end-of-call signal; the voice harness needs one so the bridge
                # can pull the summary. See ``END_CONVERSATION_GUIDELINE`` in
                # ``nemo_voice_agent.evaluation.scenarios`` for the prompt-side
                # guidance that pairs with this.
                "EndConversationTool": {},
            },
            information=[],
        )

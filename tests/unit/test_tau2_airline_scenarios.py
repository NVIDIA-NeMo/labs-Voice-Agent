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

"""Integration tests for the tau2_airline scenario port.

These touch the real ``nemo_voice_agent/evaluation/data/tau2_airline/`` fixtures
and exercise data-file load → ``Tau2AirlineBaseScenario`` → ``_gold_replay``
against the ported tools → ``expected_scenario_db`` hash.

The bulk of the file guards two deliberate divergences from upstream:

- :data:`GOLD_ACTION_DROPS` — the task-39 reference-action correction.
- :data:`ADOPTED_NL_ASSERTIONS` — the curated NL-assertion subset.

Both are local policy decisions rather than upstream data, so they need
regression cover that a re-import of the upstream fixture cannot silently undo.
"""

import copy

import pytest

import nemo_voice_agent.evaluation.scenarios.data.tau2_airline  # noqa: F401  (side-import for @register_eval_scenario)
from nemo_voice_agent.evaluation.db_hash import get_dict_hash
from nemo_voice_agent.evaluation.scenarios import ALL_EVAL_SCENARIOS
from nemo_voice_agent.evaluation.scenarios.classes import SuccessSignal
from nemo_voice_agent.evaluation.scenarios.data.tau2_airline.base import (
    ADOPTED_NL_ASSERTIONS,
    GOLD_ACTION_DROPS,
)
from nemo_voice_agent.evaluation.tools.tau2_airline_params import (
    CancelReservationParams,
    GetFlightStatusParams,
    GetReservationDetailsParams,
    GetUserDetailsParams,
    SendCertificateParams,
)


# The six tools that mutate the airline DB. A task whose gold action list
# contains none of these replays to a DB identical to the initial one.
WRITE_ACTIONS = frozenset(
    {
        "book_reservation",
        "cancel_reservation",
        "send_certificate",
        "update_reservation_baggages",
        "update_reservation_flights",
        "update_reservation_passengers",
    }
)


def _airline_scenario_keys():
    """Sorted list of ``tau2_airline__N`` keys, numerically ordered."""
    keys = [k for k in ALL_EVAL_SCENARIOS if k.startswith("tau2_airline__")]
    return sorted(keys, key=lambda k: int(k.split("__")[1]))


def test_50_airline_scenarios_registered():
    """All 50 base-split tau2_airline tasks are discoverable via the registry."""
    keys = _airline_scenario_keys()
    assert len(keys) == 50, f"expected 50 airline scenarios, got {len(keys)}"
    assert keys[0] == "tau2_airline__0"
    assert keys[-1] == "tau2_airline__49"


def test_every_airline_scenario_instantiates_and_hashes():
    """Every scenario must produce a stable SHA-256 from its expected DB."""
    for key in _airline_scenario_keys():
        inst = ALL_EVAL_SCENARIOS[key]()
        h = get_dict_hash(inst.expected_scenario_db)
        assert isinstance(h, str) and len(h) == 64, f"{key}: bad hash {h!r}"


# ---------------------------------------------------------------------------
# Task-39 gold-action correction
# ---------------------------------------------------------------------------


def test_gold_action_drops_is_narrow():
    """The correction map stays a deliberate exception, not a growing patch set.

    If a second correction is genuinely needed, update this test alongside it so
    the divergence from upstream is always an explicit review decision.
    """
    assert GOLD_ACTION_DROPS == {"39": frozenset({"39_10"})}


def test_task_39_drops_the_disallowed_cancellation():
    """MSJ4OA is economy + insurance, booked well outside the 24h window.

    policy.md limits insurance-based cancellation to health or weather reasons;
    this user wants to give up their seat. Upstream action ``39_10`` cancels it
    anyway, which would make ``DB_STATE_MATCH`` reward a policy-violating agent.
    """
    inst = ALL_EVAL_SCENARIOS["tau2_airline__39"]()
    action_ids = [a["action_id"] for a in inst.tau2_task["evaluation_criteria"]["actions"]]
    assert "39_10" not in action_ids
    assert len(action_ids) == 10

    reservations = inst.expected_scenario_db["reservations"]
    # The two business-cabin reservations remain cancellable and cancelled.
    assert reservations["8C8K4E"]["status"] == "cancelled"
    assert reservations["LU15PA"]["status"] == "cancelled"
    # The economy + insurance one must survive the replay untouched.
    assert reservations["MSJ4OA"].get("status") is None


def test_correction_does_not_mutate_the_shared_task_index():
    """``_index_entry`` is shared across instances; only the patched path copies.

    A mutation here would corrupt every other consumer of the same task entry,
    so this guards the deepcopy in ``Tau2AirlineBaseScenario.tau2_task``.
    """
    inst = ALL_EVAL_SCENARIOS["tau2_airline__39"]()
    _ = inst.tau2_task  # force the cached_property
    upstream_actions = inst._index_entry["task"]["evaluation_criteria"]["actions"]
    assert len(upstream_actions) == 11, "upstream index entry was mutated in place"
    assert inst.tau2_task is not inst._index_entry["task"]


def test_uncorrected_tasks_share_the_index_entry_by_identity():
    """No deepcopy cost for the 49 tasks that need no correction."""
    inst = ALL_EVAL_SCENARIOS["tau2_airline__41"]()
    assert inst.tau2_task is inst._index_entry["task"]


def test_clean_exit_in_every_success_signal_set():
    """Repo-wide invariant: an agent that never stops talking is not successful."""
    for key in _airline_scenario_keys():
        inst = ALL_EVAL_SCENARIOS[key]()
        assert SuccessSignal.CLEAN_EXIT in inst.success_signals, key


# ---------------------------------------------------------------------------
# Curated NL assertions
# ---------------------------------------------------------------------------


def test_adopted_assertion_totals():
    """24 tasks, 84 assertions. Pins the curated set against accidental edits."""
    assert len(ADOPTED_NL_ASSERTIONS) == 24
    assert sum(len(v) for v in ADOPTED_NL_ASSERTIONS.values()) == 84


def test_adopted_tasks_are_exactly_the_no_write_tasks():
    """The curated set targets tasks that DB_STATE_MATCH structurally cannot score.

    A task with no DB-mutating gold action replays to the initial DB, so a bot
    that greets the caller and immediately ends the call already matches the
    expected hash. Those tasks -- and only those -- need an NL signal.

    This is the whole rationale for the curated set, so it gets a test: if a
    future edit adopts assertions for a task that DB_STATE_MATCH already covers,
    that is a deliberate change of policy and should fail here first.
    """
    no_write = set()
    for key in _airline_scenario_keys():
        inst = ALL_EVAL_SCENARIOS[key]()
        actions = (inst.tau2_task.get("evaluation_criteria") or {}).get("actions") or []
        if not any(a.get("name") in WRITE_ACTIONS for a in actions):
            no_write.add(inst.tau2_id)

    assert no_write == set(ADOPTED_NL_ASSERTIONS), (
        f"adopted-but-writes: {sorted(set(ADOPTED_NL_ASSERTIONS) - no_write)}; "
        f"no-write-but-unadopted: {sorted(no_write - set(ADOPTED_NL_ASSERTIONS))}"
    )


def test_nl_assertion_signal_tracks_adoption():
    """``success_signals`` derives from ``nl_assertions`` and cannot drift from it."""
    for key in _airline_scenario_keys():
        inst = ALL_EVAL_SCENARIOS[key]()
        adopted = inst.tau2_id in ADOPTED_NL_ASSERTIONS
        assert (inst.nl_assertions is not None) == adopted, key
        assert (SuccessSignal.NL_ASSERTION in inst.success_signals) == adopted, key
        # Never an empty list -- runner.py guards on truthiness, and [] would
        # still be forwarded to the judge.
        assert inst.nl_assertions is None or len(inst.nl_assertions) > 0, key
        assert SuccessSignal.DB_STATE_MATCH in inst.success_signals, key


def test_adopted_assertions_follow_the_authoring_rules():
    """Style invariants from the constant's docstring, enforced mechanically."""
    for tau2_id, assertions in ADOPTED_NL_ASSERTIONS.items():
        assert len(set(assertions)) == len(assertions), f"task {tau2_id}: duplicate assertion"
        for a in assertions:
            where = f"task {tau2_id}: {a!r}"
            assert a.startswith("Agent "), f"{where} does not open with 'Agent '"
            assert a.endswith("."), f"{where} is not a sentence"
            # Rationale-pinning: several tasks have multiple valid grounds for
            # the same refusal, so citing one fails an agent that cites another.
            for banned in (" because ", " since "):
                assert banned not in a.lower(), f"{where} pins a rationale"
            # Addressing the grader rather than describing behaviour.
            assert not a.startswith("Check that"), where
            # Raw payment ids are unspeakable in a voice run.
            for banned in ("gift_card_", "credit_card_", "certificate_"):
                assert banned not in a, f"{where} names a raw payment id"


def test_every_task_has_a_positive_assertion():
    """The invariant the whole change rests on: a silent agent must fail every task.

    A pure "Agent does not ..." string is satisfied for free by an agent that
    greets the caller and hangs up — exactly the failure mode these assertions
    exist to catch. Each task therefore needs at least one claim requiring an
    observable act: a spoken refusal, a stated fact, or a tool call.
    """
    for tau2_id, assertions in ADOPTED_NL_ASSERTIONS.items():
        positives = [a for a in assertions if not a.startswith("Agent does not")]
        assert positives, f"task {tau2_id}: every assertion is a negative; a silent agent passes"


def test_no_assertion_duplicates_db_state_match():
    """Absence-of-mutation claims are redundant and must be phrased verbally.

    Every DB write reachable by the agent (cancel / change / modify / book /
    issue a certificate) already fails ``DB_STATE_MATCH`` on these tasks, whose
    gold DB is the seeded DB. "Agent does not cancel X" therefore cannot change
    a verdict, while a silent agent satisfies it for free.

    The verbal form ("Agent does not tell the user that X has been cancelled")
    keeps that coverage — the real mutation is still caught by DB_STATE_MATCH —
    and additionally catches the agent that *claims* to have acted without
    calling any tool, which no other signal sees.
    """
    mutating_verbs = {"cancel", "change", "modify", "book", "issue", "remove", "add"}
    for tau2_id, assertions in ADOPTED_NL_ASSERTIONS.items():
        for a in assertions:
            if not a.startswith("Agent does not "):
                continue
            verb = a[len("Agent does not ") :].split()[0].rstrip(".")
            assert verb not in mutating_verbs, (
                f"task {tau2_id}: {a!r} restates DB_STATE_MATCH; "
                f'phrase it as "Agent does not tell the user that ... has been {verb}ed."'
            )


def test_task_39_assertions_are_not_adopted():
    """Task 39 has write actions, so it is out of scope -- and must stay out.

    Its upstream assertions include ``Agent cancels reservation MSJ4OA.``, which
    directly contradicts :data:`GOLD_ACTION_DROPS`. Adopting them would score the
    agent against the very defect we corrected.
    """
    assert "39" not in ADOPTED_NL_ASSERTIONS
    assert ALL_EVAL_SCENARIOS["tau2_airline__39"]().nl_assertions is None


def test_adopted_assertions_never_read_upstream_field():
    """Airline deliberately ignores ``evaluation_criteria.nl_assertions``.

    Upstream populates it on all 50 tasks; we adopt a curated 24. If the property
    ever fell back to the upstream list, the 26 unadopted tasks would silently
    start scoring against unvetted claims.
    """
    for key in _airline_scenario_keys():
        inst = ALL_EVAL_SCENARIOS[key]()
        upstream = (inst.tau2_task.get("evaluation_criteria") or {}).get("nl_assertions") or []
        assert upstream, f"{key}: upstream fixture unexpectedly lacks nl_assertions"
        if inst.tau2_id not in ADOPTED_NL_ASSERTIONS:
            assert inst.nl_assertions is None, key
        else:
            assert inst.nl_assertions == list(ADOPTED_NL_ASSERTIONS[inst.tau2_id]), key


# ---------------------------------------------------------------------------
# Identifier casing from speech recognition
# ---------------------------------------------------------------------------


def _miscase(args):
    """Gold arguments as speech recognition may render them.

    db.json keys user ids in lowercase and reservation ids / flight numbers in
    uppercase; flip each to the other case.
    """
    out = copy.deepcopy(args)
    if "user_id" in out:
        out["user_id"] = out["user_id"].upper()
    if "reservation_id" in out:
        out["reservation_id"] = out["reservation_id"].lower()
    for flight in out.get("flights") or []:
        flight["flight_number"] = flight["flight_number"].lower()
    return out


def test_miscased_ids_replay_to_the_gold_db():
    """Writes given mis-cased ids leave exactly the gold DB and gold action records.

    Regression: ``book_reservation`` stored the raw ``user_id`` ("SOPHIA_SILVA_7557")
    and the raw flight numbers, and ``update_reservation_flights`` compared and
    stored raw flight numbers (re-pricing an unchanged lowercase leg). The lookups
    folded case, so the writes succeeded, but the DB hash no longer matched the
    gold replay and a correct conversation failed ``db_state_match``.
    """
    checked, tools_hit = set(), set()
    for key in _airline_scenario_keys():
        inst = ALL_EVAL_SCENARIOS[key]()
        actions = (inst.tau2_task.get("evaluation_criteria") or {}).get("actions") or []
        writes = [a.get("arguments") or {} for a in actions if a["name"] in WRITE_ACTIONS]
        if all(_miscase(args) == args for args in writes):
            continue
        state = {"actions": [], "db": copy.deepcopy(inst.db)}
        tools = inst._build_tool_map(state)
        for action in actions:
            args = action.get("arguments") or {}
            if action["name"] in WRITE_ACTIONS and _miscase(args) != args:
                args = _miscase(args)
                tools_hit.add(action["name"])
            tools[action["name"]].invoke(**args)
        assert get_dict_hash(state["db"]) == get_dict_hash(inst.expected_scenario_db), key
        recorded = [(r["name"], r["arguments"]) for r in state["actions"]]
        gold = [(r["name"], r["arguments"]) for r in inst.reference_answer["actions"]]
        assert recorded == gold, key
        checked.add(key)
    assert len(checked) >= 25, f"only {len(checked)} airline tasks exercised a mis-cased id"
    # send_certificate takes a user_id but no gold task calls it; it is covered below.
    assert tools_hit == WRITE_ACTIONS - {"send_certificate"}, tools_hit
    # Tasks 21 and 30 keep an unchanged leg whose stored price differs from today's fare, so they catch the
    # update_reservation_flights re-pricing of a mis-cased unchanged leg.
    assert {"tau2_airline__21", "tau2_airline__30"} <= checked


def test_send_certificate_records_the_canonical_user_id():
    inst = ALL_EVAL_SCENARIOS["tau2_airline__8"]()
    state = {"actions": [], "db": copy.deepcopy(inst.db)}
    before = set(state["db"]["users"]["sophia_silva_7557"]["payment_methods"])
    result = inst._build_tool_map(state)["send_certificate"].invoke(user_id=" SOPHIA_SILVA_7557 ", amount=100)
    assert result["status"] == "success", result
    assert state["actions"][0]["arguments"]["user_id"] == "sophia_silva_7557"
    added = set(state["db"]["users"]["sophia_silva_7557"]["payment_methods"]) - before
    assert len(added) == 1


@pytest.mark.parametrize(
    ("model", "kwargs", "field", "expected"),
    [
        (GetUserDetailsParams, {"user_id": " Sara_Doe_496 "}, "user_id", "sara_doe_496"),
        (GetReservationDetailsParams, {"reservation_id": " zfa04y "}, "reservation_id", "ZFA04Y"),
        (GetFlightStatusParams, {"flight_number": " hat001 ", "date": "2024-05-15"}, "flight_number", "HAT001"),
        (CancelReservationParams, {"reservation_id": "zfa04y"}, "reservation_id", "ZFA04Y"),
        (SendCertificateParams, {"user_id": "SARA_DOE_496", "amount": 50}, "user_id", "sara_doe_496"),
    ],
)
def test_id_argument_types_fold_case_and_whitespace(model, kwargs, field, expected):
    assert getattr(model.model_validate(kwargs), field) == expected

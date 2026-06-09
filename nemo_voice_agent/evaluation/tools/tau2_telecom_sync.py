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
# Adapted from https://github.com/sierra-research/tau2-bench/tree/voice-user-sim-v1.0
#   src/tau2/domains/telecom/environment.py — ``TelecomEnvironment.sync_tools()``
#   reconciles cross-side state on every turn so a single ``Environment``
#   object can keep the agent-side ``TelecomDB`` and the user-side
#   ``TelecomUserDB`` aligned. In voice mode the two DBs live in
#   different processes, so the equivalent reconciliation runs on the
#   bridge after each action and pushes deltas to the bots.

"""Tau2-telecom cross-side state-propagation pipeline.

Two pieces:

1. ``sync_telecom_state(agent_db, user_db)`` — pure function (no I/O)
   that mirrors upstream's ``sync_tools``. It takes both DBs in dict
   form, mutates them in place where needed, and returns a per-side
   delta dict describing the field changes the bridge must push to
   each bot so their live state catches up.

2. ``apply_telecom_sync_delta(db, delta)`` — the registered bot-side
   applier (consumed by ``apply_sync_delta`` in
   ``nemo_voice_agent.evaluation.sync_appliers``). Walks a telecom
   delta, handling list-by-id paths (``"bills[B1002].status"``) that
   the default dotted-path applier can't, and triggers
   ``_simulate_network_search`` if any ``surroundings.*`` field
   changed.

Both pieces are imported (with side effects — the applier registers
itself on import) from ``nemo_voice_agent/evaluation/tools/__init__.py``.

Propagation paths covered (matching upstream ``TelecomEnvironment.sync_tools``):

| Trigger (agent side)                       | Becomes (user side)                          |
|--------------------------------------------|----------------------------------------------|
| ``line.status`` changes                    | ``surroundings.line_active``                 |
| ``line.roaming_enabled`` changes           | ``surroundings.roaming_allowed``             |
| ``line.data_used_gb`` / ``data_refueling_gb`` / ``plan.data_limit_gb`` | ``surroundings.mobile_data_usage_exceeded`` |
| Any ``bill.status == AWAITING_PAYMENT``    | ``surroundings.payment_request``             |
| User: ``payment_request.paid = True``      | Agent: ``bill.status = Paid`` (reverse direction) |
"""

# pylint: disable=line-too-long
# flake8: noqa: E501

import re
from typing import Any, Dict, Optional

from nemo_voice_agent.evaluation.sync_appliers import register_sync_applier
from nemo_voice_agent.evaluation.tools.tau2_telecom_init_functions import _simulate_network_search
from nemo_voice_agent.evaluation.tools.tau2_telecom_params import BillStatus, LineStatus


# Sentinel: bridge expects ``sync_telecom_state`` to return this exact
# shape so the bridge layer is domain-agnostic — it just dispatches each
# side's delta. Keys map to bots; values are domain-specific delta dicts.
SyncDeltas = Dict[str, Dict[str, Any]]


# =============================================================================
# Pure sync function — mirrors upstream's TelecomEnvironment.sync_tools()
# =============================================================================


def _find_line_by_phone(agent_db: dict, phone_number: str) -> Optional[dict]:
    for line in agent_db.get("lines", []):
        if line.get("phone_number") == phone_number:
            return line
    return None


def _find_plan(agent_db: dict, plan_id: str) -> Optional[dict]:
    for p in agent_db.get("plans", []):
        if p.get("plan_id") == plan_id:
            return p
    return None


def _find_customer_by_phone(agent_db: dict, phone_number: str) -> Optional[dict]:
    for c in agent_db.get("customers", []):
        if c.get("phone_number") == phone_number:
            return c
        for lid in c.get("line_ids", []):
            line = next(
                (line for line in agent_db.get("lines", []) if line.get("line_id") == lid),
                None,
            )
            if line and line.get("phone_number") == phone_number:
                return c
    return None


def _find_bill(agent_db: dict, bill_id: str) -> Optional[dict]:
    for b in agent_db.get("bills", []):
        if b.get("bill_id") == bill_id:
            return b
    return None


def sync_telecom_state(agent_db: dict, user_db: dict) -> SyncDeltas:
    """Reconcile cross-side state after a write action fired on either side.

    Pure function — same inputs always produce the same output. No I/O,
    no logging. Called by the bridge between every action and the
    next-turn dispatch.

    Args:
        agent_db: Agent-side ``TelecomDB`` dict (mutated in place for the
            user→agent direction: ``payment_request.paid`` → ``bill.status``).
        user_db: User-side ``TelecomUserDB`` dict (mutated in place for
            the agent→user direction: ``surroundings.{line_active,
            roaming_allowed, mobile_data_usage_exceeded, payment_request}``).

    Returns:
        Per-side delta dict::

            {"agent": {<dotted-path or bills[id].field>: value, ...},
             "user":  {<dotted-path>: value, ...}}

        Empty side dicts when nothing propagates (e.g. read-only actions,
        no relevant state change). The bridge dispatches each non-empty
        side's delta via ``apply_sync_delta`` to the corresponding bot.

    Notes:
        The function is symmetric in its mutation — both DBs may get
        updated on a single call. The two delta dicts describe which
        fields to push to which bot. The bridge does NOT figure out
        directionality; it just dispatches whatever this function
        produces.
    """
    surroundings = user_db.get("surroundings") or {}
    phone_number = surroundings.get("phone_number")
    deltas: SyncDeltas = {"agent": {}, "user": {}}

    if phone_number is None:
        # No user line configured — nothing to sync. Matches upstream's
        # early return when ``surroundings.phone_number is None``.
        return deltas

    line = _find_line_by_phone(agent_db, phone_number)
    if line is None:
        # User's phone number doesn't map to a line on the agent side.
        # Upstream raises; we return empty so the conversation can
        # proceed (the agent will discover the mismatch via its own
        # lookup tools).
        return deltas

    # 1. line_active — derived from agent-side line.status
    new_line_active = line.get("status") == LineStatus.ACTIVE.value
    if surroundings.get("line_active") != new_line_active:
        surroundings["line_active"] = new_line_active
        deltas["user"]["surroundings.line_active"] = new_line_active

    # 2. roaming_allowed — derived from agent-side line.roaming_enabled
    new_roaming_allowed = bool(line.get("roaming_enabled"))
    if surroundings.get("roaming_allowed") != new_roaming_allowed:
        surroundings["roaming_allowed"] = new_roaming_allowed
        deltas["user"]["surroundings.roaming_allowed"] = new_roaming_allowed

    # 3. mobile_data_usage_exceeded — derived from line usage + plan limit + refueling
    plan = _find_plan(agent_db, line.get("plan_id", ""))
    if plan is not None:
        data_used = float(line.get("data_used_gb", 0))
        data_refueling = float(line.get("data_refueling_gb", 0))
        data_limit = float(plan.get("data_limit_gb", 0))
        new_exceeded = data_used >= data_limit + data_refueling
        if surroundings.get("mobile_data_usage_exceeded") != new_exceeded:
            surroundings["mobile_data_usage_exceeded"] = new_exceeded
            deltas["user"]["surroundings.mobile_data_usage_exceeded"] = new_exceeded

    # 4. payment_request — bidirectional
    current_pr = surroundings.get("payment_request")

    # 4a. User → agent: a paid payment_request flips the agent's bill to PAID
    #     and clears the user-side request.
    if current_pr is not None and current_pr.get("paid"):
        bill = _find_bill(agent_db, current_pr["bill_id"])
        if bill is not None and bill.get("status") != BillStatus.PAID.value:
            bill["status"] = BillStatus.PAID.value
            deltas["agent"][f"bills[{bill['bill_id']}].status"] = BillStatus.PAID.value
        surroundings["payment_request"] = None
        deltas["user"]["surroundings.payment_request"] = None
        current_pr = None  # Refresh for the next branch

    # 4b. Agent → user: if no current request, surface any bill awaiting
    #     payment for this customer as a user-side payment_request.
    if current_pr is None:
        customer = _find_customer_by_phone(agent_db, phone_number)
        if customer is not None:
            for bill_id in customer.get("bill_ids", []):
                bill = _find_bill(agent_db, bill_id)
                if bill is None:
                    continue
                if bill.get("status") == BillStatus.AWAITING_PAYMENT.value:
                    new_pr = {
                        "bill_id": bill["bill_id"],
                        "amount_due": bill.get("total_due", 0),
                        "paid": False,
                    }
                    surroundings["payment_request"] = new_pr
                    deltas["user"]["surroundings.payment_request"] = new_pr
                    break

    return deltas


# =============================================================================
# Bot-side applier — registered under domain="tau2_telecom"
# =============================================================================


_BILLS_PATH_RE = re.compile(r"^bills\[([^\]]+)\]\.(.+)$")


@register_sync_applier(domain="tau2_telecom")
def apply_telecom_sync_delta(db: dict, delta: dict) -> None:
    """Apply a telecom sync delta to the bot's live ``shared_state["db"]``.

    Handles two delta path shapes:

    - **Dotted path** (e.g. ``surroundings.payment_request``): assigned
      at the corresponding nested dict location.
    - **List-by-id** (e.g. ``bills[B1002].status``): finds the matching
      element of ``db["bills"]`` by ``bill_id`` and sets the field on it.

    After applying, if any ``surroundings.*`` field changed,
    ``_simulate_network_search`` re-derives ``network_connection_status``
    / ``network_technology_connected`` / ``network_signal_strength`` so
    the user-sim's next ``check_network_status`` / ``run_speed_test`` /
    ``_get_mobile_data_working`` calls return values consistent with the
    new surroundings.

    The default applier in ``sync_appliers.py`` would handle the dotted
    paths but NOT the ``bills[...]`` path and wouldn't trigger the
    network-search re-derivation; hence the per-domain override.
    """
    surroundings_changed = False
    for path, value in delta.items():
        if path.startswith("surroundings."):
            field = path[len("surroundings."):]
            db.setdefault("surroundings", {})[field] = value
            surroundings_changed = True
            continue
        m = _BILLS_PATH_RE.match(path)
        if m:
            bill_id, field = m.group(1), m.group(2)
            for bill in db.get("bills", []):
                if bill.get("bill_id") == bill_id:
                    bill[field] = value
                    break
            continue
        # Fall back to dotted-path set for any other shapes — same logic
        # as the default applier. Keeps this function self-contained.
        parts = path.split(".")
        target = db
        for p in parts[:-1]:
            target = target[p]
        target[parts[-1]] = value

    # Re-derive connection state if any surroundings field changed.
    # ``_simulate_network_search`` is safe to call when only agent-side
    # fields changed (it just recomputes from current state), but
    # there's no point burning cycles on it then.
    if surroundings_changed:
        try:
            _simulate_network_search(db)
        except (KeyError, TypeError):
            # Bot-side DB may not have the user-side shape (e.g. agent
            # bot received a delta meant for the agent DB which has no
            # "device"/"surroundings"). Silently skip — the per-side
            # delta dispatch in the bridge means this shouldn't
            # normally happen, but the guard prevents an unhandled
            # exception if it does.
            pass

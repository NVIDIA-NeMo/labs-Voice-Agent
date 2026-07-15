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

"""Cross-side sync-delta applier registry (bot-side).

This module is the bot-side endpoint of the cross-side state-propagation
pipeline introduced for dual-side scenarios (telecom and any future
domain that opts into ``Scenario.sync_state``).

The pipeline at a glance:

1. A write tool on bot A fires and calls ``WriteScenarioTool._record_action``,
   which (in addition to appending to ``shared_state["actions"]``) emits an
   ``action-applied`` RTVI server message.
2. The bridge picks that up, replays the action onto an in-process shadow
   copy of both DBs (using the scenario's tool map + each tool's sync
   ``invoke`` method), then calls ``scenario.sync_state(agent_db, user_db)``.
3. ``sync_state`` returns a per-side ``delta`` dict describing the
   cross-side field changes that must land on the OTHER bot. The bridge
   dispatches each delta via the ``apply_sync_delta`` RTVI action.
4. The receiving bot's handler calls into ``apply_sync_delta(domain, db, delta)``
   in this module to mutate its own ``shared_state["db"]``.

The default applier handles "dotted path → value" deltas (e.g.
``"surroundings.payment_request": {...}``) and is sufficient for any
domain whose cross-side propagation is pure field assignment. Domains
needing more (list-by-id lookups, post-apply re-derivation hooks,
domain-specific validation) register a per-domain applier via
``@register_sync_applier(domain="...")``.

This is structurally parallel to ``initialization_functions`` and
``db_state_predicates`` — same registry-by-domain pattern, same
opt-in shape.
"""

from typing import Callable, Dict

from loguru import logger


# domain → applier callable. Populated by ``@register_sync_applier``
# decorators applied to functions in the relevant domain module (e.g.,
# ``tau2_telecom_sync.py`` registers under ``"tau2_telecom"``). Missing
# domains fall back to ``_default_sync_applier``.
SYNC_APPLIERS: Dict[str, Callable[[dict, dict], None]] = {}


def register_sync_applier(domain: str):
    """Decorator: register a domain-specific applier.

    The applier signature is ``func(db: dict, delta: dict) -> None``
    and must mutate ``db`` in place. The exact shape of ``delta`` is
    a contract between the scenario's ``sync_state`` (which produces
    it) and the applier (which consumes it) — the bridge transports
    it verbatim.

    Usage::

        @register_sync_applier(domain="tau2_telecom")
        def apply_telecom_sync_delta(db: dict, delta: dict) -> None:
            ...
    """

    def deco(func: Callable[[dict, dict], None]) -> Callable[[dict, dict], None]:
        if domain in SYNC_APPLIERS:
            existing = SYNC_APPLIERS[domain]
            raise ValueError(
                f"Sync applier collision for domain {domain!r}: already "
                f"registered as {existing.__module__}.{existing.__name__}; "
                f"cannot also register {func.__module__}.{func.__name__}."
            )
        SYNC_APPLIERS[domain] = func
        return func

    return deco


def apply_sync_delta(domain: str, db: dict, delta: dict) -> None:
    """Dispatch a sync delta to the registered applier for ``domain``.

    Falls back to ``_default_sync_applier`` (dotted-path field set) when
    no per-domain applier exists. The default suffices for any future
    domain whose sync deltas are pure field assignments.

    Args:
        domain: ``Scenario.domain`` value of the active scenario.
        db: The bot's live ``shared_state["db"]`` (mutated in place).
        delta: Cross-side delta from the bridge — shape is domain-defined.
    """
    if not delta:
        return
    applier = SYNC_APPLIERS.get(domain) or _default_sync_applier
    try:
        applier(db, delta)
    except Exception as exc:
        # Log but don't crash the bot — a malformed delta should not
        # take down the conversation. The runner will catch the
        # downstream symptom (predicate failure, wrong state, etc.).
        logger.error(f"apply_sync_delta failed on domain={domain!r}: {type(exc).__name__}: {exc}")


def _default_sync_applier(db: dict, delta: dict) -> None:
    """Generic dotted-path field setter.

    Each ``delta`` key is a dotted path into ``db``; the corresponding
    value is assigned at that path. The path components are followed
    verbatim — no list-index or by-id-match support. Domains needing
    those (e.g. ``bills[B1002].status``) must register their own
    applier.

    Example::

        delta = {"surroundings.payment_request": {"bill_id": "B1002", ...}}
        # → db["surroundings"]["payment_request"] = {"bill_id": "B1002", ...}

    Raises:
        KeyError: if an intermediate path component doesn't exist in ``db``.
    """
    for path, value in delta.items():
        parts = path.split(".")
        target = db
        for p in parts[:-1]:
            target = target[p]
        target[parts[-1]] = value

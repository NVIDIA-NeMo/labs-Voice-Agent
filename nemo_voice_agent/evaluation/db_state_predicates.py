# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""DB-state predicate registry + dispatcher for ``Scenario.db_state_assertions``.

A *db-state predicate* is a deterministic function over the final DB state of a
scenario, used as a per-predicate scoring signal alongside ``db_state_match``
(whole-DB hash equality) and ``nl_assertions`` (LLM-judged transcript predicates).
Concretely:

  - Predicate signature: ``(db: dict, **arguments) -> bool``. Pure: same DB →
    same bool. No I/O, no randomness. Predicates are *side-agnostic* — they
    don't know if they're checking the "agent DB" or "user DB", and they
    don't need to: the caller (runner) picks the right DB based on the
    assertion record's ``side`` field before invoking.
  - Predicates are registered per ``(domain, func_name)`` only — no ``side``.
    Function names are unique within a domain by upstream construction (in
    tau2 telecom, the 4 user-side and 2 agent-side assertion names are
    disjoint), so adding ``side`` to the registry key would namespace
    against a non-collision.
  - Dispatch is runner-side via ``evaluate_db_state_assertion(...)``. The bot
    never sees the predicates — it just ships the inline DB dicts back to the
    runner via the ``get_scenario_summary`` action with ``include_db=True``.

Why runner-side and not bot-side:

  - Predicates are pure; they belong in a shared module, not behind an RTVI call.
  - Synthetic tests can call predicates directly on dict fixtures without
    standing up a fake RTVI bot.
  - Verdict aggregation lives next to the existing ``nl_assertion`` and
    ``db_state_match`` aggregation in ``runner.py`` — uniform code path for
    all three scoring signals.

The mirror-symmetric ``initialization_actions`` surface goes the *other* way
(bot-side dispatch via the ``apply_initialization_actions`` RTVI action), because init
actions mutate the live DB through real toolkit methods. Predicates only read.

Upstream tau2-bench calls this surface ``env_assertions``; we renamed to
``db_state_assertions`` so the name parallels the existing ``db_state_match``
metric (same artifact, finer granularity) and mirrors ``nl_assertions`` in
shape (per-predicate verdicts). Same upstream JSON shape, just renamed at the
Scenario field + metric layer.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from loguru import logger


# Predicate signature: ``(db: dict, **arguments) -> bool``.
Predicate = Callable[..., bool]

# Domain → func_name → predicate. Flat per-domain — predicate names are
# unique within a domain by upstream construction, so no per-side
# disambiguation is needed. The caller (runner dispatcher) chooses which
# DB to pass; the predicate just reads from it.
ALL_DB_STATE_PREDICATES: Dict[str, Dict[str, Predicate]] = {}


def register_db_state_predicate(domain: str):
    """Decorator: register a predicate function under ``(domain, fn.__name__)``.

    Usage::

        @register_db_state_predicate(domain="tau2_telecom")
        def assert_mobile_data_status(db: dict, expected_status: bool) -> bool:
            return _get_mobile_data_working(db) == expected_status

    The function name (``fn.__name__``) becomes the registry key — matching
    the upstream ``func_name`` field in
    ``evaluation_criteria.env_assertions``. Renames at the source require
    updating the upstream task JSON, so don't.

    **No ``side`` parameter.** Predicate names are unique within a domain;
    ``side`` is purely caller-side metadata (the runner uses it on each
    assertion record to pick which pulled DB to pass — agent's ``db`` or
    user's ``user_db``). The predicate itself is side-agnostic.

    Args:
        domain: Registry namespace (matching ``Scenario.domain``, e.g.
            ``"tau2_telecom"``).

    Raises:
        ValueError: if a predicate with the same name is already registered
            in this domain.
    """

    def _decorator(fn: Predicate) -> Predicate:
        bucket = ALL_DB_STATE_PREDICATES.setdefault(domain, {})
        key = fn.__name__
        if key in bucket:
            existing = bucket[key]
            raise ValueError(
                f"Predicate name collision in domain '{domain}': '{key}' "
                f"already registered by {existing.__module__}.{existing.__qualname__}; "
                f"cannot also register {fn.__module__}.{fn.__qualname__}."
            )
        bucket[key] = fn
        return fn

    return _decorator


def evaluate_db_state_assertion(
    domain: str,
    assertion: Dict[str, Any],
    db: Optional[Dict[str, Any]],
    user_db: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run one ``db_state_assertion`` against the pulled DB and return a verdict.

    Verdict shape mirrors ``nl_assertion_verdicts`` so the runner can
    aggregate both via the same code path::

        {
            "func_name": str,
            "side": str,
            "passed": bool,           # True iff predicate(db, **arguments) == assert_value
            "expected": Any,          # the assertion's assert_value
            "actual": Optional[bool], # what the predicate returned (None on error)
            "message": Optional[str], # upstream's optional human-readable label
            "error": Optional[str],   # set to the failure mode when predicate
                                      # is missing or raises; passed=False then.
        }

    Args:
        domain: Scenario domain (e.g., ``"tau2_telecom"``). Determines which
            registry bucket to look up the predicate in.
        assertion: One entry from ``scenario.db_state_assertions``. Required keys:
            ``side``, ``func_name``, ``arguments``, ``assert_value``. Optional:
            ``message``. ``side`` is used here only to pick which DB to pass to
            the predicate — it is NOT part of the registry key.
        db: The agent-side DB dict pulled back by the bridge
            (``shared_state["db"]``). Used when ``side=="agent"``. May
            be ``None`` if the scenario only emits user-side assertions.
        user_db: The user-side DB dict pulled back by the bridge
            (``shared_state["user_db"]``). Used when ``side=="user"``.

    Never raises — always returns a verdict dict. A missing predicate or
    raising predicate becomes ``passed=False`` with ``error`` set, so the
    runner can keep iterating the assertion list and surface all failures at
    once rather than aborting on the first.
    """
    side = assertion.get("side")
    func_name = assertion.get("func_name")
    arguments = assertion.get("arguments") or {}
    expected = assertion.get("assert_value")
    message = assertion.get("message")

    verdict: Dict[str, Any] = {
        "func_name": func_name,
        "side": side,
        "passed": False,
        "expected": expected,
        "actual": None,
        "message": message,
        "error": None,
    }

    # Pick the right DB by side. ``side`` is used here as caller-side routing
    # metadata only — never as part of the registry key.
    if side == "user":
        target_db = user_db
    elif side == "agent":
        target_db = db
    else:
        verdict["error"] = f"Unknown side {side!r}; expected 'user' or 'agent'."
        return verdict

    if target_db is None:
        verdict["error"] = (
            f"No {side} DB available — scenario declared "
            f"side={side!r} but the bridge didn't pull a "
            f"{'user_db' if side == 'user' else 'db'} dict back. Check "
            f"that the bot returned include_db=True payload."
        )
        return verdict

    # Registry lookup is by (domain, func_name) only — predicates are
    # side-agnostic; the side has already been used above to pick target_db.
    domain_bucket = ALL_DB_STATE_PREDICATES.get(domain, {})
    predicate = domain_bucket.get(func_name)
    if predicate is None:
        verdict["error"] = (
            f"No predicate {func_name!r} registered under domain={domain!r}. "
            f"Available: {sorted(domain_bucket.keys())}."
        )
        return verdict

    # Invoke and compare to assert_value.
    try:
        actual = predicate(target_db, **arguments)
    except Exception as exc:  # noqa: BLE001 — surface all predicate failures uniformly
        logger.exception(
            "db_state_assertion predicate {!r} raised on domain={}",
            func_name,
            domain,
        )
        verdict["error"] = f"Predicate raised: {type(exc).__name__}: {exc}"
        return verdict

    verdict["actual"] = actual
    verdict["passed"] = actual == expected
    return verdict


def list_registered_predicates(domain: Optional[str] = None) -> Dict[str, Any]:
    """Diagnostic helper. Returns ``{domain: [func_names]}``.

    With ``domain=None`` returns the whole registry; with a specific domain
    returns only that subtree. Useful for test setup verification and
    debugging missing-predicate errors.
    """
    if domain is None:
        return {d: sorted(fns.keys()) for d, fns in ALL_DB_STATE_PREDICATES.items()}
    return {domain: sorted(ALL_DB_STATE_PREDICATES.get(domain, {}).keys())}

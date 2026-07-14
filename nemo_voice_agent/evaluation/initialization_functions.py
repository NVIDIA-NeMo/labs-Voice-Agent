# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initialization-function registry + dispatcher for ``Scenario.initialization_actions``.

An *initialization function* is a state-mutating action replayed against the
live bot-side DB before the scenario starts. They seed the environment to a
known starting state so the agent and user simulator can have a meaningful
conversation (e.g., upstream tau2-bench's ``set_user_info``,
``turn_roaming_off``, ``enable_roaming``).

Concretely:

  - Function signature: ``(db: dict, **arguments) -> None``. Mutates the
    passed dict in place; return value ignored. Errors raise normally and
    are caught by the dispatcher, which returns ``{success: False, errors: [...]}``
    so the bridge can abort the scenario cleanly. Functions are *side-agnostic*
    — they don't know if they're operating on the "agent DB" or "user DB", and
    they don't need to: the caller (bot handler) chose the right DB dict
    before invoking them.
  - Functions are registered per ``(domain, func_name)`` only — no ``side``.
    Function names are unique within a domain by upstream construction (in
    tau2 telecom, the 16 user-side init names and 4 agent-side init names
    are disjoint), so adding ``side`` to the registry key would namespace
    against a non-collision.
  - Dispatch is **bot-side** (the opposite of predicates). The bot has the
    live ``shared_state["db"]`` (agent) or ``shared_state["user_db"]`` (user)
    and the toolkit instance methods that read/write them; init functions
    just mutate the dict directly. The bridge calls the ``apply_initialization``
    RTVI action once per side (agent bot for ``side=="agent"`` actions, user
    bot for ``side=="user"`` actions); each bot's handler selects its own DB
    based on the ``side`` it was told and passes that single dict to this
    dispatcher (``apply_initialization_actions`` — the Python function below,
    which keeps its plural name for clarity that it dispatches a LIST).

Why bot-side dispatch (and not runner-side like ``db_state_assertions``):

  - Init actions **mutate** state. The mutation has to land in the same dict
    instance the live LLM tools will see and modify during the conversation —
    that's the bot's ``shared_state``, not a snapshot in the runner.
  - Mutations are imperative; pure-function dispatch on the runner side
    would require shipping the mutated DB back to the bot, doubling the
    transport cost and adding a serialization round trip.
  - Symmetric to upstream: tau2-bench's ``Environment.run_env_function_call``
    dispatches against the live ``self.tools`` / ``self.user_tools`` toolkit
    instances. Bot-side replay matches that semantics exactly.

Upstream tau2-bench's initialization functions are **methods on toolkit
classes** (``TelecomUserTools.set_user_info``, etc.), not module-level
functions. The telecom port extracts them as module-level functions taking a
plain dict — same approach as the predicate port — so the registry stays
language-agnostic and testable.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from loguru import logger


# Initialization function signature: ``(db: dict, **arguments) -> None``.
# Mutates the passed dict in place; return value is ignored.
InitializationFunction = Callable[..., None]

# Domain → func_name → function. Flat per-domain — function names are unique
# within a domain by upstream construction, so no per-side disambiguation is
# needed. The caller (bot handler) chooses which DB to pass; the function
# just operates on it.
ALL_INITIALIZATION_FUNCTIONS: Dict[str, Dict[str, InitializationFunction]] = {}


def register_initialization_function(domain: str):
    """Decorator: register an init function under ``(domain, fn.__name__)``.

    Usage::

        @register_initialization_function(domain="tau2_telecom")
        def set_user_info(db: dict, name: str, phone_number: str) -> None:
            db["surroundings"]["name"] = name
            db["surroundings"]["phone_number"] = phone_number

    The function name (``fn.__name__``) becomes the registry key — matching
    the upstream ``func_name`` field in
    ``task["initial_state"]["initialization_actions"]``. Renames at the source
    require updating the upstream task JSON, so don't.

    **No ``side`` parameter.** Function names are unique within a domain;
    ``side`` is purely caller-side metadata (the bridge uses it to route
    each action to the right bot, the bot handler uses it to pick the right
    DB out of its ``shared_state``). The function itself is side-agnostic.

    Args:
        domain: Registry namespace (matching ``Scenario.domain``, e.g.
            ``"tau2_telecom"``).

    Raises:
        ValueError: if a function with the same name is already registered
            in this domain.
    """

    def _decorator(fn: InitializationFunction) -> InitializationFunction:
        bucket = ALL_INITIALIZATION_FUNCTIONS.setdefault(domain, {})
        key = fn.__name__
        if key in bucket:
            existing = bucket[key]
            raise ValueError(
                f"Initialization function name collision in domain '{domain}': "
                f"'{key}' already registered by "
                f"{existing.__module__}.{existing.__qualname__}; cannot also "
                f"register {fn.__module__}.{fn.__qualname__}."
            )
        bucket[key] = fn
        return fn

    return _decorator


def apply_initialization_actions(
    domain: str,
    actions: List[Dict[str, Any]],
    db: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Iterate ``actions`` and dispatch each against ``db``.

    Called bot-side by the ``apply_initialization`` RTVI handler.
    All actions in the list apply to the single ``db`` dict — the caller
    (bot handler) has already picked the correct side-specific dict from
    ``shared_state`` based on the ``side`` arg sent by the bridge. Each
    action's per-record ``side`` field is therefore informational only and
    not consulted here.

    Mutates ``db`` in place. Returns a result dict with the overall success
    status and per-action error log (if any) so the bridge can abort the
    scenario when seeding fails.

    Args:
        domain: Scenario domain (e.g., ``"tau2_telecom"``). Determines which
            registry bucket to look up functions in. Sent by the bridge in
            the action payload.
        actions: List of ``{func_name, arguments, side?}`` records to replay.
            ``side`` is permitted (upstream JSON carries it for traceability)
            but ignored by this dispatcher.
        db: The DB dict to mutate. The bot handler picks this from
            ``shared_state["db"]`` (agent side) or ``["user_db"]`` (user
            side) based on the ``side`` arg the bridge sent.

    Returns:
        ``{"success": bool, "errors": [str, ...]}``. ``success`` is ``True``
        only when every action dispatched cleanly. ``errors`` lists the
        failure mode per failed action (missing function, raised exception);
        the list is empty on full success. The dispatcher does not roll back
        successful mutations on a downstream failure — the bridge treats any
        failure as a framework-class error and aborts the scenario without
        scoring it.
    """
    errors: List[str] = []

    if db is None:
        # The caller should have caught this before getting here, but be
        # defensive — the failure mode (mutate None) would otherwise be a
        # bare ``AttributeError`` deep inside an init function.
        return {
            "success": False,
            "errors": [
                "No db provided to apply_initialization_actions; the bot "
                "handler is responsible for selecting shared_state['db'] "
                "(agent side) or shared_state['user_db'] (user side) and "
                "passing it here."
            ],
        }

    for idx, action in enumerate(actions):
        func_name = action.get("func_name")
        arguments = action.get("arguments") or {}

        domain_bucket = ALL_INITIALIZATION_FUNCTIONS.get(domain, {})
        fn = domain_bucket.get(func_name)
        if fn is None:
            err = (
                f"[action {idx}] No initialization function {func_name!r} "
                f"registered under domain={domain!r}. Available: "
                f"{sorted(domain_bucket.keys())}."
            )
            logger.error(err)
            errors.append(err)
            continue

        try:
            fn(db, **arguments)
        except Exception as exc:  # noqa: BLE001 — surface all init failures uniformly
            logger.exception(
                "initialization function {!r} raised on domain={}",
                func_name,
                domain,
            )
            errors.append(f"[action {idx}] {func_name!r} raised: {type(exc).__name__}: {exc}")

    return {"success": len(errors) == 0, "errors": errors}


def list_registered_initialization_functions(domain: Optional[str] = None) -> Dict[str, Any]:
    """Diagnostic helper. Returns ``{domain: [func_names]}``.

    With ``domain=None`` returns the whole registry; with a specific domain
    returns only that subtree.
    """
    if domain is None:
        return {d: sorted(fns.keys()) for d, fns in ALL_INITIALIZATION_FUNCTIONS.items()}
    return {domain: sorted(ALL_INITIALIZATION_FUNCTIONS.get(domain, {}).keys())}

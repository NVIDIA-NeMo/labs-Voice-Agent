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

"""
Resume-state classification utilities shared between the evaluation runner
and the check_resume.py dry-run script.
"""

import json
import os
from typing import Optional, Tuple


def count_agent_llm_messages(scenario_dir: str) -> Optional[int]:
    """Return the number of assistant-role messages in the agent's saved LLM context.

    Returns None when the file is absent (old runs, crashed before write) or
    unreadable — callers treat None as "unknown, skip the check."

    NOTE: this file is unreliable — the agent's ``bot_logs_agent/llm_context.json``
    is frequently saved empty even for scenarios where the agent ran fine (the
    context is lost during end-of-scenario retrieval). Prefer
    ``count_agent_responses`` for stall detection; this remains only as a
    last-resort fallback for runs whose metrics.json predates ``token_usage``.
    """
    ctx_file = os.path.join(scenario_dir, "bot_logs_agent", "llm_context.json")
    if not os.path.exists(ctx_file):
        return None
    try:
        with open(ctx_file) as f:
            ctx = json.load(f)
        return sum(1 for msg in ctx if isinstance(msg, dict) and msg.get("role") == "assistant")
    except (json.JSONDecodeError, OSError):
        return None


def count_agent_responses(scenario_dir: str, metrics: Optional[dict] = None) -> Optional[int]:
    """Return how many LLM responses the agent produced for a scenario.

    This is the robust stall-detection signal. It prefers the live-recorded
    ``token_usage.agent.n_calls`` from metrics.json — that counter is
    accumulated by the bridge *during* the run and survives the agent-context
    save bug (see ``count_agent_llm_messages``). Resolution order:

    1. ``metrics["token_usage"]["agent"]["n_calls"]`` from the passed-in dict
       (the live runner path hands us the in-memory metrics directly).
    2. The same field read from ``<scenario_dir>/metrics.json`` on disk
       (resume / load-from-disk path).
    3. Fallback to ``count_agent_llm_messages`` for old runs whose metrics.json
       predates the ``token_usage`` field.

    Returns None when no signal is available — callers treat None as
    "unknown, skip the check."
    """
    if metrics is None:
        mf = os.path.join(scenario_dir, "metrics.json")
        if os.path.exists(mf):
            try:
                with open(mf) as f:
                    metrics = json.load(f)
            except (json.JSONDecodeError, OSError):
                metrics = None
    if isinstance(metrics, dict):
        n = ((metrics.get("token_usage") or {}).get("agent") or {}).get("n_calls")
        if isinstance(n, int):
            return n
    return count_agent_llm_messages(scenario_dir)


def classify_scenario_resume_state(scenario_dir: str, min_agent_turns: int = 0) -> Tuple[str, str]:
    """Classify a scenario directory for resume purposes.

    Returns a ``(state, reason)`` tuple where ``state`` is one of:

    - ``"completed"``  — has a valid ``metrics.json`` and passes all stall checks;
                         the runner will skip it and load metrics from disk.
    - ``"in_flight"``  — started but not cleanly finished (missing/unreadable
                         ``metrics.json``, 0 turns, or fewer agent LLM messages
                         than ``min_agent_turns``); the runner will move it aside
                         and re-run it.
    - ``"fresh"``      — no subdir exists yet; the runner will run it normally.

    ``reason`` is a short human-readable string explaining the classification,
    useful for logging and the dry-run check script.
    """
    if not os.path.isdir(scenario_dir):
        return "fresh", "no subdir"

    mf = os.path.join(scenario_dir, "metrics.json")
    if not os.path.exists(mf):
        return "in_flight", "no metrics.json (in-flight)"

    try:
        with open(mf) as f:
            m = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return "in_flight", f"metrics.json unreadable ({e})"

    # 0-turn scenarios indicate the bot crashed before any audio was exchanged.
    if m.get("total_turns", 0) == 0:
        return "in_flight", "0 turns (bot crashed before audio)"

    # Scenarios with too few agent LLM responses (stalled vLLM, server hang, etc.)
    # are treated as in-flight so they get re-run on --resume.
    # Uses the live-recorded token_usage.agent.n_calls from metrics.json (robust
    # to the agent-context save bug); falls back to the saved context file for
    # old runs. Old runs lacking both are left as "completed".
    if min_agent_turns > 0:
        count = count_agent_responses(scenario_dir, m)
        if count is not None and count < min_agent_turns:
            stop = m.get("stop_reason", "?")
            return "in_flight", f"{count} agent LLM response(s) < {min_agent_turns} ({stop})"

    return "completed", m.get("stop_reason", "?")

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
    # Read from the saved context file — no new metrics field needed,
    # and old runs that lack the file are left as "completed".
    if min_agent_turns > 0:
        count = count_agent_llm_messages(scenario_dir)
        if count is not None and count < min_agent_turns:
            stop = m.get("stop_reason", "?")
            return "in_flight", f"{count} agent LLM message(s) < {min_agent_turns} ({stop})"

    return "completed", m.get("stop_reason", "?")

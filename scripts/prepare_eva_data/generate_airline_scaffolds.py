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

"""Generate NeMo Scenario class scaffolds from eva_airline_dataset.jsonl entries.

One-shot generator for eva_airline scenario classes — reads
``nemo_voice_agent/evaluation/data/eva_airline/eva_airline_dataset.jsonl`` and emits
``@register_eval_scenario class Eva...`` definitions to stdout. Output is a
**starting point, not final**: paste/append into the relevant
``nemo_voice_agent/evaluation/scenarios/data/eva_airline/group_Nx.py`` module,
then hand-review prose quality + prune negotiation / edge-case bullets before
committing.

Skips the 5 scenarios that are already hand-authored (1.1.2, 2.1.1, 3.1.3,
5.1.1, 7.2.1) — those remain the canonical reference for prose style.

Usage::

    cd /path/to/NeMo-Voice-Agent
    python scripts/prepare_eva_data/generate_airline_scaffolds.py \\
        > nemo_voice_agent/evaluation/scenarios/data/eva_airline/group_all.py

Unlike the tau2 scaffolders (which fully overwrite group files), this one
streams to stdout for hand-review — eva_airline scenarios carry curated
prose that the dataset alone can't fully express.
"""

import argparse
import json
import sys
from pathlib import Path


ALREADY_PORTED = {"1.1.2", "2.1.1", "3.1.3", "5.1.1", "7.2.1"}

DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def spell_out(code: str) -> str:
    """Spell out a code character-by-character: '8JVSDF' → 'eight, J, V, S, D, F'."""
    parts = []
    for ch in code:
        if ch.isdigit():
            parts.append(DIGIT_WORDS[ch])
        elif ch.isalpha():
            parts.append(ch.upper())
        # skip non-alphanumeric
    return ", ".join(parts)


def fmt_code(code: str) -> str:
    """Format as 'CODE (spelled out as X, Y, Z)'."""
    return f"{code} (spelled out as {spell_out(code)})"


def _lookup(info: dict, *keys, default=None):
    """First non-empty value found across a list of candidate keys."""
    for k in keys:
        v = info.get(k)
        if v:
            return v
    return default


def extract_identity(info: dict) -> dict:
    """Pull confirmation_number, last_name, first_name across alias variations."""
    conf = _lookup(
        info,
        "confirmation_number",
        "confirmation_code",
        "Booking confirmation number",
        "Booking confirmation code",
        default="",
    )
    last = _lookup(
        info,
        "last_name",
        "passenger_last_name",
        "Passenger last name",
        default="",
    )
    first = _lookup(
        info,
        "first_name",
        "first_name_if_asked",
        "first_name_if_requested",
        "Passenger first name",
        default="",
    )
    return {"confirmation": conf, "last_name": last, "first_name": first}


def extract_route(info: dict) -> dict:
    """Pull origin, destination, date, departure_time from original_travel_details (most reliable)."""
    details = info.get("original_travel_details") or []
    if not details:
        return {"origin": "", "destination": "", "date": "", "departure_time": ""}
    first_seg = details[0]
    return {
        "origin": first_seg.get("origin", ""),
        "destination": first_seg.get("destination", ""),
        "date": first_seg.get("flight_date", ""),
        "departure_time": first_seg.get("departure_time", ""),
        "original_flight": _lookup(info, "original_flight_number", "flight_number", default=""),
    }


def class_name(eva_id: str) -> str:
    """'2.1.6' → 'EvaAirline216'."""
    return "EvaAirline" + eva_id.replace(".", "")


def scenario_name(eva_id: str) -> str:
    """'2.1.6' → 'eva_airline__2_1_6'."""
    return "eva_airline__" + eva_id.replace(".", "_")


def _bullet_list(items: list, indent: str = "                ") -> str:
    """Format a list of strings as quoted Python list items."""
    if not items:
        return ""
    return "\n".join(f'{indent}"{escape(item)}",' for item in items)


def escape(s: str) -> str:
    """Escape for embedding in a double-quoted Python string."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def build_scenario(entry: dict) -> str:
    """Emit a complete @register_eval_scenario class for one eva dataset entry."""
    eid = entry["id"]
    cls = class_name(eid)
    name = scenario_name(eid)

    ug = entry.get("user_goal", {})
    high_level = ug.get("high_level_user_goal", "").strip()
    starting = ug.get("starting_utterance", "").strip()
    dt = ug.get("decision_tree", {})

    must_have = dt.get("must_have_criteria", []) or []
    negotiation = dt.get("negotiation_behavior", []) or []
    failure_cond = dt.get("failure_condition", "").strip()
    escalation = dt.get("escalation_behavior", "").strip()
    edge_cases = dt.get("edge_cases", []) or []

    info = ug.get("information_required", {}) or {}
    ident = extract_identity(info)
    route = extract_route(info)

    user_cfg = entry.get("user_config", {})
    full_name = user_cfg.get("name", "").strip()
    persona_text = user_cfg.get("user_persona", "").strip()

    # Short description (1 sentence, max 120 chars)
    desc = high_level[:200].rsplit(".", 1)[0] + "." if "." in high_level[:200] else high_level[:200]

    # Build identity context line for guidelines
    identity_bits = []
    if ident["confirmation"]:
        identity_bits.append(f"Your confirmation number is {fmt_code(ident['confirmation'])}.")
    if ident["last_name"]:
        identity_bits.append(f"Your last name is {ident['last_name']}.")
    if ident["first_name"]:
        identity_bits.append(f"Your first name is {ident['first_name']}.")
    identity_line = " ".join(identity_bits)

    # Build route line if present
    route_bits = []
    if route["origin"] and route["destination"]:
        route_bits.append(f"Your booking: {fmt_code(route['origin'])} to {fmt_code(route['destination'])}")
        if route["date"]:
            route_bits.append(f"on {route['date']}")
        if route["departure_time"]:
            route_bits.append(f"departing at {route['departure_time']}")
        if route["original_flight"]:
            route_bits.append(f"(flight {fmt_code(route['original_flight'])})")
    route_line = " ".join(route_bits) + "." if route_bits else ""

    # Compose instructions: starting utterance + standard auth steps + close
    instructions = []
    if starting:
        instructions.append(f"Greet the agent and say: '{starting}'")
    instructions.append("Provide your confirmation number when the agent asks.")
    instructions.append("Provide your last name when the agent asks.")
    instructions.append("Share trip details, constraints, and what you want when the agent asks.")
    instructions.append(
        "When the agent presents options or asks for decisions, choose options "
        "that meet your must-have criteria (or follow the failure path in your guidelines if none fit)."
    )
    instructions.append("Once the work is done and the agent has confirmed it, thank them and end the call.")

    # Compose guidelines: voice rule + identity + must-haves + edge cases + failure + escalation
    guidelines = ["self.VOICE_ALPHANUMERIC_RULE"]
    if identity_line:
        guidelines.append(("STRING", identity_line))
    if route_line:
        guidelines.append(("STRING", route_line))
    for c in must_have:
        guidelines.append(("STRING", "Must-have: " + c.strip()))
    # Selected negotiation behavior items (keep first 3 to avoid bloat — reviewer can prune more)
    for n in negotiation[:3]:
        guidelines.append(("STRING", n.strip()))
    for ec in edge_cases[:5]:
        guidelines.append(("STRING", "Edge case: " + ec.strip()))
    if failure_cond:
        guidelines.append(("STRING", "Failure path: " + failure_cond))
    if escalation:
        guidelines.append(("STRING", "Escalation: " + escalation))

    # Render guidelines
    g_lines = []
    for g in guidelines:
        if g == "self.VOICE_ALPHANUMERIC_RULE":
            g_lines.append("                self.VOICE_ALPHANUMERIC_RULE,")
        else:
            _, text = g
            g_lines.append(f'                "{escape(text)}",')

    out = f'''
# ---------------------------------------------------------------------------
# eva {eid}: {desc[:100]}
# ---------------------------------------------------------------------------


@register_eval_scenario
class {cls}(EvaAirlineBaseScenario):
    """Auto-scaffolded from eva dataset.jsonl entry {eid}. Review prose before shipping."""

    name = "{name}"
    eva_id = "{eid}"
    description = "{escape(desc[:200])}"

    @property
    def user_persona(self) -> Persona:
        return Persona(
            role="airline passenger",
            name="{escape(full_name)}",
            background=(
                "{escape(high_level)}"
            ),
            personality=(
                "{escape(persona_text[:300])}"
            ),
        )

    @property
    def user_task(self) -> Task:
        return Task(
            goal=(
                "{escape(high_level)}"
            ),
        )

    @property
    def user_actions(self) -> Actions:
        return Actions(
            instructions=[
{_bullet_list(instructions)}
            ],
            guidelines=[
{chr(10).join(g_lines)}
            ],
        )
'''
    return out


def _emit_group(entries: list, major: int) -> None:
    """Emit all (non-already-ported) scaffolds for one major group to stdout."""
    matching = [e for e in entries if e["id"].startswith(f"{major}.") and e["id"] not in ALREADY_PORTED]
    matching.sort(key=lambda e: tuple(int(p) for p in e["id"].split(".")))

    print(f"# Generated {len(matching)} scenarios for major group {major}.x", file=sys.stderr)
    for e in matching:
        print(f"#   {e['id']}", file=sys.stderr)
    print(file=sys.stderr)

    # Section header on stdout so a single dump covering multiple
    # majors can be split mechanically by the operator.
    print("# =========================================================")
    print(f"# === eva_airline major group {major}.x ({len(matching)} scenarios) ===")
    print("# =========================================================")
    print()
    for entry in matching:
        print(build_scenario(entry))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--major",
        type=int,
        default=None,
        help=(
            "Major group number to emit (e.g. 1 emits 1.x scenarios). "
            "Omit to emit every major group present in the dataset, with "
            "a section header between each — useful for a one-shot preview "
            "before splitting per-group output."
        ),
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default=str(
            Path(__file__).resolve().parents[2]
            / "nemo_voice_agent"
            / "evaluation"
            / "data"
            / "eva_airline"
            / "eva_airline_dataset.jsonl"
        ),
    )
    args = ap.parse_args()

    with open(args.dataset) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    if args.major is not None:
        _emit_group(entries, args.major)
        return

    # Discover every major group present in the dataset, sorted ascending,
    # and emit each in turn. The same ALREADY_PORTED filter applies
    # per-group, so the 5 hand-authored seeds are skipped.
    majors = sorted({int(e["id"].split(".")[0]) for e in entries if e.get("id")})
    print(f"# Major groups present in dataset: {majors}", file=sys.stderr)
    for major in majors:
        _emit_group(entries, major)


if __name__ == "__main__":
    main()

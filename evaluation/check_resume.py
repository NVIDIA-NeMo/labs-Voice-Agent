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

"""
Dry-run resume checker for eval result directories.

Reports which scenarios would be re-run by ``run_evaluation.py --resume``,
including the ``--min-agent-turns`` stall detection, without modifying
anything on disk.

Usage:
  python check_resume.py <eval_dir> [--min-agent-turns N]

Examples:
  python check_resume.py eval_results/eval_20260618_072325
  python check_resume.py eval_results/eval_20260618_072325 --min-agent-turns 4
"""

import argparse
import os
import sys

from nemo_voice_agent.evaluation.resume import classify_scenario_resume_state


def classify(scenario_dir: str, min_agent_turns: int):
    """Thin wrapper that maps resume module states to check_resume display states."""
    state, reason = classify_scenario_resume_state(scenario_dir, min_agent_turns)
    # The resume module uses "in_flight"; this script surfaces it as "rerun"
    # to be explicit about what --resume would do.
    if state == "in_flight":
        return "rerun", reason
    return state, reason


def main():
    parser = argparse.ArgumentParser(
        description="Dry-run resume check: show which scenarios would be re-run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("eval_dir", help="Path to the eval_<TIMESTAMP>/ session directory")
    parser.add_argument(
        "--min-agent-turns",
        type=int,
        default=0,
        metavar="N",
        help="Flag scenarios with fewer than N assistant LLM messages as stalled (default: 0 = disabled)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.eval_dir):
        print(f"ERROR: {args.eval_dir} is not a directory", file=sys.stderr)
        return 1

    rerun = []
    completed = []
    fresh = []

    for name in sorted(os.listdir(args.eval_dir)):
        scen_dir = os.path.join(args.eval_dir, name)
        if not os.path.isdir(scen_dir):
            continue
        if ".killed." in name or name == "__KILLED__":
            continue
        # Skip non-scenario top-level files/dirs (run_args.json, evaluation_log.txt, etc.)
        if not any(
            os.path.exists(os.path.join(scen_dir, f)) for f in ("metrics.json", "bridge_log.txt", "scenario_config")
        ):
            continue

        state, reason = classify(scen_dir, args.min_agent_turns)
        if state == "rerun":
            rerun.append((name, reason))
        elif state == "completed":
            completed.append((name, reason))
        else:
            fresh.append((name, reason))

    total = len(rerun) + len(completed) + len(fresh)
    print(f"Eval dir : {args.eval_dir}")
    print(f"Min agent turns: {args.min_agent_turns}")
    print(f"Total scenarios: {total}  ({len(completed)} completed, {len(rerun)} would re-run, {len(fresh)} fresh)")

    if rerun:
        print(f"\nWould re-run ({len(rerun)}):")
        for name, reason in rerun:
            print(f"  {name}  [{reason}]")

    if fresh:
        print(f"\nFresh / never started ({len(fresh)}):")
        for name, reason in fresh:
            print(f"  {name}")

    if not rerun and not fresh:
        print("\nAll scenarios are complete — nothing to resume.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

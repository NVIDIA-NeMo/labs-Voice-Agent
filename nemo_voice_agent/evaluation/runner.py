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
Dynamic Voice Agent Evaluation Runner

Runs evaluation scenarios with dynamic system prompt updates.
Accepts structured Scenario objects instead of raw dicts.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import List, Optional

from nemo_voice_agent.evaluation.bridge import VoiceAgentEvaluationBridge
from nemo_voice_agent.evaluation.db_hash import get_dict_hash
from nemo_voice_agent.evaluation.db_state_predicates import evaluate_db_state_assertion
from nemo_voice_agent.evaluation.scenarios.classes import Scenario
from nemo_voice_agent.evaluation.utils import LLMJudge, check_if_task_success, normalize_scenario_payload
from nemo_voice_agent.utils import FileLogger


async def run_dynamic_evaluation(
    user_url: str,
    agent_url: str,
    output_dir: str,
    scenarios: List[Scenario],
    audio_chunk_in_seconds: float = 0.016,
    duration_per_scenario: Optional[int] = None,
    pause_between_scenarios: float = 0.5,
    user_output_sample_rate: int = 24000,
    agent_output_sample_rate: int = 24000,
    user_input_sample_rate: int = 16000,
    agent_input_sample_rate: int = 16000,
    output_sample_rate: int = 24000,
    global_timestamp: str = None,
    logger: FileLogger = None,
    judge: Optional[LLMJudge] = None,
    judge_threshold: Optional[float] = None,
    strict_match: bool = False,
):
    """
    Run evaluation with dynamic scenario switching and latency measurement.

    Args:
        user_url: WebSocket URL of user (simulated user)
        agent_url: WebSocket URL of agent being tested
        output_dir: Output directory for results
        scenarios: List of Scenario objects defining each evaluation scenario
        audio_chunk_in_seconds: Audio chunk in seconds for the audio stream (default: 0.016)
        duration_per_scenario: Maximum duration per scenario in seconds, which overrides the scenario's own max_duration if set.
        pause_between_scenarios: Seconds to pause between scenarios
        user_output_sample_rate: User TTS output sample rate (default: 24000)
        agent_output_sample_rate: Agent TTS output sample rate (default: 24000)
        user_input_sample_rate: User STT input sample rate (default: 16000)
        agent_input_sample_rate: Agent STT input sample rate (default: 16000)
        output_sample_rate: Output sample rate for recorded audio (default: 24000)
        global_timestamp: Timestamp string for output file naming
        logger: FileLogger instance for logging
        judge: LLMJudge instance for judging the scenario
        judge_threshold: Threshold for judging the scenario if binary result is desired, None for score based result
        strict_match: If True, force ``disallow_extra_items=True`` on every scenario for this run,
            overriding each scenario's own setting. Default False respects per-scenario flags.
    """

    if not logger:
        logger = FileLogger()

    os.makedirs(output_dir, exist_ok=True)
    global_timestamp = global_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    bridge = VoiceAgentEvaluationBridge(
        user_url=user_url,
        agent_url=agent_url,
        output_dir=None,  # Will be set per scenario
        user_output_sample_rate=user_output_sample_rate,
        agent_output_sample_rate=agent_output_sample_rate,
        user_input_sample_rate=user_input_sample_rate,
        agent_input_sample_rate=agent_input_sample_rate,
        output_sample_rate=output_sample_rate,
        audio_chunk_in_seconds=audio_chunk_in_seconds,
    )

    all_results = []
    success_results = []
    # DB-state match results (only collected for scenarios with `expected_scenario_db`).
    # Denominator is "scenarios that opted into DB-state scoring", not "all scenarios".
    db_state_results: List[bool] = []
    # Per-assertion verdicts flattened across scenarios with nl_assertions. Each entry
    # is True/False for one assertion in one scenario; denominator is "total
    # nl_assertions emitted in this run", not "scenarios". Empty when no scenario
    # opts into nl-assertion scoring.
    nl_assertion_results: List[bool] = []
    # Per-predicate verdicts flattened across scenarios with db_state_assertions.
    # Same shape as ``nl_assertion_results`` — each entry is True/False for one
    # predicate in one scenario; denominator is "total db_state_assertions
    # emitted in this run", not "scenarios". Empty when no scenario opts into
    # DB-state-assertion scoring (currently only tau2-telecom).
    db_state_assertion_results: List[bool] = []
    # Per-side token usage accumulated across scenarios. Populated from each
    # scenario's ``bridge.token_usage`` snapshot. The eval-result-analyzer skill
    # used to recompute this from bridge_log.txt; this canonical source lets
    # the runner print totals in all_summary.txt without log re-parsing.
    run_token_usage: dict = {
        "agent": {"n_calls": 0, "prompt": 0, "completion": 0},
        "user": {"n_calls": 0, "prompt": 0, "completion": 0},
    }
    # Per-domain buckets keyed by ``scenario.name.split('__')[0]`` so a mixed run
    # (eva_airline + tau2_airline + retail + …) reports success rates per source.
    # Scenarios without a ``__`` separator (e.g. "fastbite", "simple_qa_1") fall
    # under the scenario name itself.
    per_domain_success: dict = {}
    per_domain_db_state: dict = {}
    per_domain_nl_assertion: dict = {}
    per_domain_db_state_assertion: dict = {}
    for idx, scenario in enumerate(scenarios):
        logger.info(f"{'='*80}")
        logger.info(f"Starting Scenario {idx+1}/{len(scenarios)}: {scenario.name}")
        logger.info(f"{'='*80}\n")

        # Create scenario-specific directory
        scenario_dir = os.path.join(output_dir, scenario.name)
        os.makedirs(scenario_dir, exist_ok=True)

        # Per-side shared_state — let the scenario seed scenario fixtures
        # (e.g., a database path) before tools are instantiated on the bot
        # server. Decoupled from agent tool-call order; LLM-invisible.
        user_state, agent_state = {}, {}
        scenario.setup_shared_state(user_state, "user")
        scenario.setup_shared_state(agent_state, "agent")

        # Build dict for bridge.prepare_for_scenario. ``tool_domain`` is the
        # registry namespace the bots should use to look up tool classes by
        # name; defaults to "default" for scenarios that don't override.
        scenario_dict = {
            "name": scenario.name,
            "user_prompt": scenario.get_user_prompt(),
            "agent_prompt": scenario.get_agent_prompt(),
            "user_tools": scenario.get_user_tools(),
            "agent_tools": scenario.get_agent_tools(),
            "user_shared_state_init": json.dumps(user_state),
            "agent_shared_state_init": json.dumps(agent_state),
            "tool_domain": getattr(scenario, "domain", "default"),
            # When set, the bridge asks the bot for the inline DB dict in
            # ``get_scenario_summary`` (not just the hash) so the runner can
            # evaluate ``db_state_assertions`` predicates against it. Off
            # for retail (7MB DB exceeds the WS frame cap); on for telecom
            # (~5KB user_db). See ``Scenario.db_state_assertions``.
            "include_db_in_summary": bool(getattr(scenario, "db_state_assertions", None)),
            # Initialization actions replayed bot-side via the new
            # ``apply_initialization_actions`` RTVI action before the
            # conversation starts. ``None`` (default) skips the replay step
            # entirely — eva/airline/retail won't hit this path. Telecom
            # populates this from ``task["initial_state"]["initialization_actions"]``.
            "initialization_actions": getattr(scenario, "initialization_actions", None),
        }
        if scenario.noise_config:
            scenario_dict["noise_config"] = scenario.noise_config

        logger.info(f"Preparing for scenario: {scenario.name}...")
        await bridge.prepare_for_scenario(scenario_dict, scenario_dir)
        scenario_config_dir = os.path.join(scenario_dir, "scenario_config")
        os.makedirs(scenario_config_dir, exist_ok=True)
        scenario.save(scenario_config_dir)
        await asyncio.sleep(pause_between_scenarios)

        # Run scenario
        duration = duration_per_scenario if duration_per_scenario is not None else scenario.max_duration
        assert duration > 0, f"Duration per scenario must be greater than 0, got {duration}"
        logger.info(f"Running scenario for {duration} seconds...")

        scenario_start = datetime.now()
        await bridge.run_scenario(duration=duration)
        scenario_end = datetime.now()

        # Domain prefix for per-domain aggregation. Scenarios named "<prefix>__<id>"
        # bucket under <prefix>; scenarios without a "__" (e.g. "fastbite") bucket
        # under the full name.
        domain = scenario.name.split("__", 1)[0]

        # Check if the scenario is successful
        reference_file = os.path.join(scenario_config_dir, scenario.reference_file)
        prediction_file = os.path.join(scenario_dir, bridge.final_response_file)
        # Per-scenario nl-assertion pass rate (set inside the judge branch when
        # the scenario carries nl_assertions; remains None for scenarios without
        # assertions OR for paths that bypass the judge — e.g. action-list scoring).
        scenario_nl_pass_rate: Optional[float] = None
        if not os.path.exists(reference_file):
            logger.info(f"Reference file {reference_file} not found, skipping checking for task success...")
            is_successful = "N/A"
        elif not os.path.exists(prediction_file):
            logger.info(f"Prediction file {prediction_file} not found, setting task success to False...")
            is_successful = False
            success_results.append(False)
            per_domain_success.setdefault(domain, []).append(False)
        elif judge is not None:
            # Wire through judge_scenario (not judge_file). Same output shape
            # ({"score", "reason"}) when nl_assertions is None, so the
            # downstream scoring path is unchanged. tau2_retail (and any
            # other domain with nl_assertions) gets per-assertion verdicts
            # in the response.
            #
            # Shape-normalize both files before handing to the judge: the
            # deterministic comparator's "Situation 2" logic treats
            # ``{...}`` and ``[{...}]`` as equivalent payloads, but the
            # LLM judge reads raw text and would otherwise deduct for the
            # cosmetic wrapping difference. ``normalize_scenario_payload``
            # collapses list-of-1-dict → single dict on both sides so the
            # judge sees identical shapes when the content matches.
            with open(reference_file, "r") as f:
                ref_obj = normalize_scenario_payload(json.load(f))
            with open(prediction_file, "r") as f:
                pred_obj = normalize_scenario_payload(json.load(f))
            ref_content = json.dumps(ref_obj, indent=2)
            pred_content = json.dumps(pred_obj, indent=2)
            # Load the agent's LLM context history written by
            # bridge._save_user_agent_history at scenario end. Shape is a list
            # of {role, content} dicts including tool calls — gives the judge
            # visibility into what the agent actually did, not just its final
            # response. None if the bridge didn't write the file (e.g. early
            # crash before history was captured).
            agent_context_file = os.path.join(scenario_dir, "bot_logs_agent", "llm_context.json")
            agent_context_history = None
            if os.path.exists(agent_context_file):
                with open(agent_context_file, "r") as f:
                    agent_context_history = json.load(f)
            else:
                logger.info(
                    f"Agent context history file {agent_context_file} not found; "
                    "calling judge_scenario without context_history."
                )
            scenario_nl_assertions = getattr(scenario, "nl_assertions", None)
            result = judge.judge_scenario(
                reference=ref_content,
                prediction=pred_content,
                context_history=agent_context_history,
                nl_assertions=scenario_nl_assertions,
            )
            with open(os.path.join(scenario_dir, "judge_result.json"), "w") as f:
                json.dump(result, f, indent=2)
            if judge_threshold is not None:
                is_successful = result["score"] >= judge_threshold
            else:
                is_successful = result["score"]
            success_results.append(is_successful)
            per_domain_success.setdefault(domain, []).append(is_successful)
            # Roll per-assertion verdicts into both the run-wide list and the
            # per-domain bucket. Denominator is total assertions, not scenarios —
            # makes the rate stable when scenarios carry different assertion counts.
            verdicts = result.get("nl_assertion_verdicts") if scenario_nl_assertions else None
            if verdicts:
                scenario_passes = 0
                for v in verdicts:
                    passed = bool(v.get("passed"))
                    nl_assertion_results.append(passed)
                    per_domain_nl_assertion.setdefault(domain, []).append(passed)
                    if passed:
                        scenario_passes += 1
                scenario_nl_pass_rate = scenario_passes / len(verdicts)
        else:
            scenario_disallow_extra = strict_match or getattr(scenario, "disallow_extra_items", False)
            is_successful = check_if_task_success(
                reference=reference_file,
                prediction=prediction_file,
                ignore_capitalization=getattr(scenario, "ignore_capitalization", False),
                ignore_punctuation=getattr(scenario, "ignore_punctuation", False),
                clean_text=getattr(scenario, "clean_text", False),
                disallow_extra_items=scenario_disallow_extra,
            )
            success_results.append(is_successful)
            per_domain_success.setdefault(domain, []).append(is_successful)

        # Collect metrics for this scenario
        metrics = bridge.get_metrics()
        metrics["scenario_name"] = scenario.name
        metrics["scenario_directory"] = scenario_dir
        metrics["scenario_duration"] = (scenario_end - scenario_start).total_seconds()
        metrics["is_successful"] = is_successful
        # Snapshot per-side token usage accumulated by the bridge during the
        # scenario. Shape mirrors ``run_token_usage``: each side has n_calls,
        # prompt sum, completion sum. Roll into the run-level accumulator for
        # the summary that goes into ``all_summary.txt``.
        scenario_token_usage = bridge.token_usage
        metrics["token_usage"] = scenario_token_usage
        for side in ("agent", "user"):
            for key in ("n_calls", "prompt", "completion"):
                run_token_usage[side][key] += scenario_token_usage[side][key]

        # Optional DB-state hash matching — runs alongside action-list scoring as
        # an independent signal. Only fires for scenarios that expose
        # ``expected_scenario_db`` (eva_airline + every tau2 domain).
        # Path-independent: any sequence of agent actions that lands in the
        # right end state passes.
        #
        # Hash-only design: the bot server computes ``get_dict_hash(state["db"])``
        # inside the get_scenario_summary RTVI action and returns the SHA-256
        # only. The runner compares it to its own ``get_dict_hash(expected_db)``
        # (computed in-process from gold replay). No inline DB ever crosses the
        # WebSocket — tau2's 7MB DB would exceed pipecat's 1MB frame limit and
        # close the connection. ``compute_db_diff`` is no longer invoked on
        # mismatch since the runner never sees the actual DB. For diagnostics,
        # rerun and inspect ``bot_logs_agent/`` or add a debug-only retrieval action.
        expected_db = getattr(scenario, "expected_scenario_db", None)
        if expected_db is not None:
            summary = bridge.scenario_summary or {}
            actual_hash = summary.get("db_hash") if isinstance(summary, dict) else None
            if actual_hash is None:
                logger.info("Bot did not report a db_hash in get_scenario_summary; skipping DB-state match.")
                metrics["db_state_match"] = "N/A"
            else:
                expected_hash = get_dict_hash(expected_db)
                metrics["db_state_match"] = expected_hash == actual_hash
                metrics["db_state_expected_hash"] = expected_hash
                metrics["db_state_actual_hash"] = actual_hash
                db_state_results.append(metrics["db_state_match"])
                per_domain_db_state.setdefault(domain, []).append(metrics["db_state_match"])

        # Persist per-scenario nl-assertion pass rate so it lands in metrics.json
        # and the per-scenario console summary below. ``None`` ↔ scenario carries
        # no nl_assertions OR judge wasn't run; matches ``db_state_match``'s
        # "absent key" convention via a conditional set.
        if scenario_nl_pass_rate is not None:
            metrics["nl_assertion_pass_rate"] = scenario_nl_pass_rate

        # Optional db_state_assertions evaluation — third scoring signal alongside
        # ``db_state_match`` (whole-DB hash) and ``nl_assertions`` (LLM-judged).
        # Predicates are pure functions over the pulled DB dicts; dispatched via
        # ``evaluate_db_state_assertion`` from ``db_state_predicates.py``. Only
        # fires for scenarios that expose ``db_state_assertions`` (tau2-telecom).
        # Per-predicate verdict shape mirrors ``nl_assertion_verdicts`` so
        # downstream aggregation uses the same pattern.
        #
        # The runner needs inline ``db`` / ``user_db`` dicts here, not the
        # SHA-256 hashes used for ``db_state_match``. The bot returns them via
        # the ``include_db=True`` payload flag on ``get_scenario_summary``;
        # the bridge sets that flag when
        # ``scenario.db_state_assertions`` is truthy. Predicate calls fall back
        # to ``passed=False`` with an explanatory ``error`` if the DB is missing,
        # so we surface the misconfiguration in the verdicts instead of crashing.
        scenario_db_state_assertion_pass_rate = None
        scenario_db_state_assertions = getattr(scenario, "db_state_assertions", None)
        if scenario_db_state_assertions:
            summary = bridge.scenario_summary or {}
            pulled_db = summary.get("db") if isinstance(summary, dict) else None
            pulled_user_db = summary.get("user_db") if isinstance(summary, dict) else None
            db_state_verdicts = [
                evaluate_db_state_assertion(
                    domain=scenario.domain,
                    assertion=a,
                    db=pulled_db,
                    user_db=pulled_user_db,
                )
                for a in scenario_db_state_assertions
            ]
            metrics["db_state_assertion_verdicts"] = db_state_verdicts
            scenario_db_passes = sum(1 for v in db_state_verdicts if v.get("passed"))
            scenario_db_state_assertion_pass_rate = (
                scenario_db_passes / len(db_state_verdicts) if db_state_verdicts else None
            )
            metrics["db_state_assertion_pass_rate"] = scenario_db_state_assertion_pass_rate
            # Roll per-predicate verdicts into the run-wide + per-domain buckets.
            # Denominator is total predicates, not scenarios — matches the
            # nl_assertion convention so the two rates are comparable.
            for v in db_state_verdicts:
                passed = bool(v.get("passed"))
                db_state_assertion_results.append(passed)
                per_domain_db_state_assertion.setdefault(domain, []).append(passed)

        # Save metrics to file
        metrics_file = os.path.join(scenario_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Scenario Metrics saved to: {metrics_file}")

        all_results.append(metrics)

        # Log scenario summary
        latency_stats = metrics["latency_stats"]
        logger.info(f"{'='*80}")
        logger.info(f"Scenario '{scenario.name}' Complete")
        logger.info(f"{'='*80}")
        logger.info(f"  Is successful: {metrics['is_successful']}")
        if "db_state_match" in metrics:
            logger.info(f"  DB-state match: {metrics['db_state_match']}")
        if "nl_assertion_pass_rate" in metrics:
            logger.info(f"  NL-assertion pass rate: {metrics['nl_assertion_pass_rate']*100:.2f}%")
        if "db_state_assertion_pass_rate" in metrics:
            logger.info(
                f"  DB-state-assertion pass rate: {metrics['db_state_assertion_pass_rate']*100:.2f}%"
            )
        logger.info(f"  Total turns: {metrics['total_turns']}")
        logger.info(f"  Duration: {metrics['scenario_duration']:.1f}s")
        logger.info(f"  Latency measurements: {latency_stats['count']}")
        if latency_stats["count"] > 0:
            logger.info(f"  Mean latency: {latency_stats['mean_ms']:.1f}ms")
            logger.info(f"  P50 latency: {latency_stats['p50_ms']:.1f}ms")
            logger.info(f"  P95 latency: {latency_stats['p95_ms']:.1f}ms")

    # Save detailed results
    results_file = os.path.join(output_dir, "all_metrics.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save CSV with latency details
    latency_csv_file = os.path.join(output_dir, "all_latencies.csv")
    with open(latency_csv_file, "w") as f:
        f.write("Scenario,User_Transcript,Agent_Transcript,Latency_ms\n")
        for result in all_results:
            scenario_name = result["scenario_name"]
            for latency in result["latencies"]:
                user_text = latency["user_transcript"].replace('"', '""')
                agent_text = latency["agent_transcript"].replace('"', '""')
                f.write(f'"{scenario_name}","{user_text}","{agent_text}",{latency["latency_ms"]:.1f}\n')

    # Save summary
    summary_file = os.path.join(output_dir, "all_summary.txt")
    success_rate = sum(success_results) / len(success_results) if len(success_results) > 0 else 0
    # Denominator is "scenarios with expected_scenario_db", not all scenarios.
    # None when no scenario in the run opted into DB-state scoring.
    db_state_success_rate = sum(db_state_results) / len(db_state_results) if db_state_results else None
    # Denominator is "assertions emitted in this run", not scenarios. None when no
    # scenario carried nl_assertions (eva / tau2_airline / tau2_telecom only have
    # action/DB-state signal).
    nl_assertion_success_rate = (
        sum(nl_assertion_results) / len(nl_assertion_results) if nl_assertion_results else None
    )
    # Denominator is "db_state_assertions emitted in this run", not scenarios.
    # None when no scenario carried db_state_assertions (currently only
    # tau2-telecom does).
    db_state_assertion_success_rate = (
        sum(db_state_assertion_results) / len(db_state_assertion_results)
        if db_state_assertion_results
        else None
    )
    all_latencies = []
    for result in all_results:
        all_latencies.extend([lat["latency_ms"] for lat in result["latencies"]])
    all_latencies.sort()
    overall_latency_stats = {
        "count": len(all_latencies),
        "mean_ms": (sum(all_latencies) / len(all_latencies) if len(all_latencies) > 0 else -1),
        "p50_ms": (all_latencies[len(all_latencies) // 2] if len(all_latencies) > 0 else -1),
        "p95_ms": (all_latencies[int(len(all_latencies) * 0.95)] if len(all_latencies) > 0 else -1),
        "min_ms": all_latencies[0] if len(all_latencies) > 0 else -1,
        "max_ms": all_latencies[-1] if len(all_latencies) > 0 else -1,
    }
    with open(summary_file, "w") as f:
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        total_turns = sum(r["total_turns"] for r in all_results)
        total_duration = sum(r["scenario_duration"] for r in all_results)

        f.write(f"Total Scenarios: {len(scenarios)}\n")
        f.write(f"Total Duration: {total_duration:.1f}s\n")
        f.write(f"Total Turns: {total_turns}\n\n")

        f.write("Per-Scenario Results:\n")
        f.write("-" * 80 + "\n")
        for result in all_results:
            stats = result["latency_stats"]
            f.write(f"\n====== {result['scenario_name']} ======:\n")
            f.write(f"  Is successful: {result['is_successful']}\n")
            if "db_state_match" in result:
                f.write(f"  DB-state match: {result['db_state_match']}\n")
            if "nl_assertion_pass_rate" in result:
                f.write(f"  NL-assertion pass rate: {result['nl_assertion_pass_rate']*100:.2f}%\n")
            if "db_state_assertion_pass_rate" in result:
                f.write(
                    f"  DB-state-assertion pass rate: "
                    f"{result['db_state_assertion_pass_rate']*100:.2f}%\n"
                )
            f.write(f"  Turns: {result['total_turns']}\n")
            f.write(f"  Duration: {result['scenario_duration']:.1f}s\n")
            if result["scenario_duration"] > 0:
                f.write(f"  Turns/min: {result['total_turns'] / (result['scenario_duration'] / 60):.1f}\n")
            f.write(f"  Latency Measurements: {stats['count']}\n")
            if stats["count"] > 0:
                f.write(f"    Mean: {stats['mean_ms']:.1f}ms\n")
                f.write(f"    P50: {stats['p50_ms']:.1f}ms\n")
                f.write(f"    P95: {stats['p95_ms']:.1f}ms\n")
                f.write(f"    Min: {stats['min_ms']:.1f}ms\n")
                f.write(f"    Max: {stats['max_ms']:.1f}ms\n")

        # Overall latency statistics
        f.write("\n\nOverall Latency Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Total Measurements: {overall_latency_stats['count']}\n")
        f.write(f"  Mean: {overall_latency_stats['mean_ms']:.1f}ms\n")
        f.write(f"  P50: {overall_latency_stats['p50_ms']:.1f}ms\n")
        f.write(f"  P95: {overall_latency_stats['p95_ms']:.1f}ms\n")
        f.write(f"  Min: {overall_latency_stats['min_ms']:.1f}ms\n")
        f.write(f"  Max: {overall_latency_stats['max_ms']:.1f}ms\n")

        if success_results:
            # success_results entries are bool (action-list match, or judge ≥ threshold)
            # or float (raw judge score when no threshold set). `:g` drops trailing zeros
            # so 2.0 → "2" and 1.6 → "1.6".
            f.write(
                f"\n\nOverall Success Rate: {success_rate*100:.2f}% "
                f"({sum(success_results):g}/{len(success_results)} scenarios with reference_answer)\n"
            )
        else:
            f.write("\n\nOverall Success Rate: N/A (no scenarios had reference_answer for action-list scoring)\n")
        if db_state_success_rate is not None:
            f.write(
                f"DB-State Match Rate: {db_state_success_rate*100:.2f}% "
                f"({sum(db_state_results)}/{len(db_state_results)} scenarios with expected_scenario_db)\n"
            )
        if nl_assertion_success_rate is not None:
            f.write(
                f"NL-Assertion Pass Rate: {nl_assertion_success_rate*100:.2f}% "
                f"({sum(nl_assertion_results)}/{len(nl_assertion_results)} assertions across scenarios)\n"
            )
        if db_state_assertion_success_rate is not None:
            f.write(
                f"DB-State-Assertion Pass Rate: {db_state_assertion_success_rate*100:.2f}% "
                f"({sum(db_state_assertion_results)}/{len(db_state_assertion_results)} "
                f"predicates across scenarios)\n"
            )

        # Token usage rollup — only printed when there's at least one
        # token-emitting call across the run. Bots that don't emit RTVI
        # ``metrics`` token events (e.g. very old bot versions) end up here
        # with zeros; suppress the block in that case to keep the summary
        # clean. Numbers come from each scenario's bridge.token_usage
        # snapshot accumulated into run_token_usage.
        total_calls = run_token_usage["agent"]["n_calls"] + run_token_usage["user"]["n_calls"]
        if total_calls > 0:
            f.write("\n\nToken Usage:\n")
            f.write("-" * 80 + "\n")
            grand_total = 0
            for side in ("agent", "user"):
                b = run_token_usage[side]
                side_total = b["prompt"] + b["completion"]
                grand_total += side_total
                f.write(
                    f"  {side}: {b['n_calls']} call(s), "
                    f"prompt={b['prompt']:,}, completion={b['completion']:,}, "
                    f"total={side_total:,}\n"
                )
            f.write(f"  Run total: {grand_total:,} tokens\n")

        # Per-domain breakdown — only printed when there's more than one domain in
        # the run, since a single-domain breakdown duplicates the overall rate.
        if len(per_domain_success) > 1:
            f.write("\n\nPer-Domain Success Rate:\n")
            f.write("-" * 80 + "\n")
            for d in sorted(per_domain_success):
                results = per_domain_success[d]
                rate = sum(results) / len(results) if results else 0
                f.write(f"  {d}: {rate*100:.2f}% ({sum(results):g}/{len(results)})\n")
        if len(per_domain_db_state) > 1:
            f.write("\nPer-Domain DB-State Match Rate:\n")
            f.write("-" * 80 + "\n")
            for d in sorted(per_domain_db_state):
                results = per_domain_db_state[d]
                rate = sum(results) / len(results) if results else 0
                f.write(f"  {d}: {rate*100:.2f}% ({sum(results)}/{len(results)})\n")
        if len(per_domain_nl_assertion) > 1:
            f.write("\nPer-Domain NL-Assertion Pass Rate:\n")
            f.write("-" * 80 + "\n")
            for d in sorted(per_domain_nl_assertion):
                results = per_domain_nl_assertion[d]
                rate = sum(results) / len(results) if results else 0
                f.write(f"  {d}: {rate*100:.2f}% ({sum(results)}/{len(results)})\n")
        if len(per_domain_db_state_assertion) > 1:
            f.write("\nPer-Domain DB-State-Assertion Pass Rate:\n")
            f.write("-" * 80 + "\n")
            for d in sorted(per_domain_db_state_assertion):
                results = per_domain_db_state_assertion[d]
                rate = sum(results) / len(results) if results else 0
                f.write(f"  {d}: {rate*100:.2f}% ({sum(results)}/{len(results)})\n")

    logger.info(f"{'='*80}")
    logger.info("Evaluation Complete!")
    logger.info(f"{'='*80}")
    if success_results:
        logger.info(
            f"Overall Success Rate: {success_rate*100:.2f}% "
            f"({sum(success_results):g}/{len(success_results)} scenarios with reference_answer)"
        )
    else:
        logger.info("Overall Success Rate: N/A (no scenarios had reference_answer for action-list scoring)")
    if db_state_success_rate is not None:
        logger.info(
            f"DB-State Match Rate: {db_state_success_rate*100:.2f}% "
            f"({sum(db_state_results)}/{len(db_state_results)} scenarios with expected_scenario_db)"
        )
    if nl_assertion_success_rate is not None:
        logger.info(
            f"NL-Assertion Pass Rate: {nl_assertion_success_rate*100:.2f}% "
            f"({sum(nl_assertion_results)}/{len(nl_assertion_results)} assertions across scenarios)"
        )
    if db_state_assertion_success_rate is not None:
        logger.info(
            f"DB-State-Assertion Pass Rate: {db_state_assertion_success_rate*100:.2f}% "
            f"({sum(db_state_assertion_results)}/{len(db_state_assertion_results)} "
            f"predicates across scenarios)"
        )
    run_token_total = (
        run_token_usage["agent"]["prompt"] + run_token_usage["agent"]["completion"]
        + run_token_usage["user"]["prompt"] + run_token_usage["user"]["completion"]
    )
    if run_token_total > 0:
        logger.info(
            f"Token Usage: agent={run_token_usage['agent']['prompt'] + run_token_usage['agent']['completion']:,}"
            f" + user={run_token_usage['user']['prompt'] + run_token_usage['user']['completion']:,}"
            f" = {run_token_total:,} total"
        )
    if len(per_domain_success) > 1:
        for d in sorted(per_domain_success):
            results = per_domain_success[d]
            rate = sum(results) / len(results) if results else 0
            logger.info(f"  [{d}] Success Rate: {rate*100:.2f}% ({sum(results):g}/{len(results)})")
    if len(per_domain_db_state) > 1:
        for d in sorted(per_domain_db_state):
            results = per_domain_db_state[d]
            rate = sum(results) / len(results) if results else 0
            logger.info(f"  [{d}] DB-State Match: {rate*100:.2f}% ({sum(results)}/{len(results)})")
    if len(per_domain_nl_assertion) > 1:
        for d in sorted(per_domain_nl_assertion):
            results = per_domain_nl_assertion[d]
            rate = sum(results) / len(results) if results else 0
            logger.info(f"  [{d}] NL-Assertion: {rate*100:.2f}% ({sum(results)}/{len(results)})")
    if len(per_domain_db_state_assertion) > 1:
        for d in sorted(per_domain_db_state_assertion):
            results = per_domain_db_state_assertion[d]
            rate = sum(results) / len(results) if results else 0
            logger.info(
                f"  [{d}] DB-State-Assertion: {rate*100:.2f}% ({sum(results)}/{len(results)})"
            )
    logger.info(f"Overall Latency P95: {overall_latency_stats['p95_ms']:.1f}ms")
    logger.info(f"Overall Latency P50: {overall_latency_stats['p50_ms']:.1f}ms")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Latencies saved to: {latency_csv_file}")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info("\nScenario directories:")
    for result in all_results:
        logger.info(f"  {result['scenario_name']}: {result['scenario_directory']}")
    logger.info(f"\nTotal: {len(scenarios)} scenarios, {total_turns} turns, {total_duration:.1f}s")

    return all_results

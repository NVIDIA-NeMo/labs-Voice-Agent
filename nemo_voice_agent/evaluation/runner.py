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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from nemo_voice_agent.evaluation.bridge import VoiceAgentEvaluationBridge
from nemo_voice_agent.evaluation.db_hash import get_dict_hash
from nemo_voice_agent.evaluation.db_state_predicates import evaluate_db_state_assertion
from nemo_voice_agent.evaluation.scenarios.classes import Scenario, SuccessSignal
from nemo_voice_agent.evaluation.utils import LLMJudge, check_if_task_success, normalize_scenario_payload
from nemo_voice_agent.utils import FileLogger


@dataclass
class RunAggregator:
    """Run-level metric accumulator.

    Holds per-signal pass-rate buckets + per-domain breakdowns + per-side
    token totals across all scenarios in a run. ``add_scenario(metrics, domain)``
    folds a per-scenario metrics dict into all buckets uniformly — used by
    both the freshly-run path (after computing metrics) and the resume-skip
    path (after loading metrics.json from disk for an already-completed
    scenario). Centralizing this logic guarantees the final ``all_summary.txt``
    aggregate is identical regardless of whether a scenario ran live or was
    loaded from a prior session.

    Bucket lists are kept as flat ``List[bool|float]`` (not keyed by
    ``SuccessSignal``) so the existing downstream consumers in
    ``run_dynamic_evaluation`` (summary writer, per-domain rollups) can
    reference them by name without changing 100+ call sites. Dict-key
    lookups on incoming ``metrics`` use ``SuccessSignal.*`` members directly
    (``StrEnum`` makes them str-equal to their JSON-key values).
    """

    success_results: List[bool] = field(default_factory=list)
    action_match_results: List[bool] = field(default_factory=list)
    judge_score_results: List[float] = field(default_factory=list)
    judge_pass_results: List[bool] = field(default_factory=list)
    db_state_results: List[bool] = field(default_factory=list)
    nl_assertion_results: List[bool] = field(default_factory=list)
    db_state_assertion_results: List[bool] = field(default_factory=list)

    per_domain_success: Dict[str, List[bool]] = field(default_factory=dict)
    per_domain_action_match: Dict[str, List[bool]] = field(default_factory=dict)
    per_domain_judge_score: Dict[str, List[float]] = field(default_factory=dict)
    per_domain_judge_pass: Dict[str, List[bool]] = field(default_factory=dict)
    per_domain_db_state: Dict[str, List[bool]] = field(default_factory=dict)
    per_domain_nl_assertion: Dict[str, List[bool]] = field(default_factory=dict)
    per_domain_db_state_assertion: Dict[str, List[bool]] = field(default_factory=dict)

    run_token_usage: dict = field(
        default_factory=lambda: {
            "agent": {"n_calls": 0, "prompt": 0, "completion": 0},
            "user": {"n_calls": 0, "prompt": 0, "completion": 0},
        }
    )

    def add_scenario(self, metrics: dict, domain: str) -> None:
        """Append one scenario's metrics into all run-level + per-domain buckets.

        Idempotent against incomplete metrics dicts — each signal is only
        appended when its corresponding key is present and well-typed in
        ``metrics``. Signals not opted into by the scenario (no
        ``expected_scenario_db``, no NL assertions, etc.) simply don't
        contribute to their bucket.

        ``SuccessSignal`` enum members are used as the dict keys when
        looking up signal values in ``metrics`` — StrEnum members compare
        equal to their string values, so this preserves byte-stability of
        the on-disk metrics.json format while pinning the lookup to a
        typo-resistant symbol.
        """
        # SuccessSignal.ACTION_MATCH = "is_action_match" → either bool, "N/A", or absent.
        v = metrics.get(SuccessSignal.ACTION_MATCH)
        if isinstance(v, bool):
            self.action_match_results.append(v)
            self.per_domain_action_match.setdefault(domain, []).append(v)

        # ``judge_score`` is the raw float; not a SuccessSignal member.
        js = metrics.get("judge_score")
        if isinstance(js, (int, float)):
            self.judge_score_results.append(float(js))
            self.per_domain_judge_score.setdefault(domain, []).append(float(js))

        # SuccessSignal.JUDGE_PASSED = "judge_passed" → bool when --judge-threshold is set.
        v = metrics.get(SuccessSignal.JUDGE_PASSED)
        if isinstance(v, bool):
            self.judge_pass_results.append(v)
            self.per_domain_judge_pass.setdefault(domain, []).append(v)

        # SuccessSignal.DB_STATE_MATCH = "db_state_match" → bool when expected_scenario_db is set.
        v = metrics.get(SuccessSignal.DB_STATE_MATCH)
        if isinstance(v, bool):
            self.db_state_results.append(v)
            self.per_domain_db_state.setdefault(domain, []).append(v)

        # NL assertions: per-assertion verdicts are persisted only in judge_result.json,
        # not in metrics.json itself. We pull them back out by file when the pass rate
        # is present so the run-level rate is denominated in assertions, not scenarios.
        if isinstance(metrics.get("nl_assertion_pass_rate"), (int, float)):
            scen_dir = metrics.get("scenario_directory")
            if scen_dir:
                jrf = os.path.join(scen_dir, "judge_result.json")
                if os.path.exists(jrf):
                    try:
                        with open(jrf) as f:
                            jr = json.load(f)
                        for v in jr.get("nl_assertion_verdicts") or []:
                            passed = bool(v.get("passed"))
                            self.nl_assertion_results.append(passed)
                            self.per_domain_nl_assertion.setdefault(domain, []).append(passed)
                    except (json.JSONDecodeError, OSError):
                        pass

        # DB-state assertions: per-predicate verdicts live in metrics.json directly.
        for v in metrics.get("db_state_assertion_verdicts") or []:
            passed = bool(v.get("passed"))
            self.db_state_assertion_results.append(passed)
            self.per_domain_db_state_assertion.setdefault(domain, []).append(passed)

        # Composite is_successful. May be "N/A" when no whitelisted signal was applicable —
        # skip those (the scenario can't be scored, so it shouldn't drag the run rate).
        if isinstance(metrics.get("is_successful"), bool):
            self.success_results.append(metrics["is_successful"])
            self.per_domain_success.setdefault(domain, []).append(metrics["is_successful"])

        # Per-side token usage rollup.
        tu = metrics.get("token_usage") or {}
        for side in ("agent", "user"):
            sub = tu.get(side) or {}
            for key in ("n_calls", "prompt", "completion"):
                self.run_token_usage[side][key] += sub.get(key, 0)


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

    # Warn early if any queued scenario's gating depends on the LLM judge but
    # the judge isn't enabled for this run. ``NL_ASSERTION`` also runs through
    # the judge so the same warning covers it.
    if judge is None:
        judge_dependent_signals = {SuccessSignal.JUDGE_PASSED, SuccessSignal.NL_ASSERTION}
        needs_judge = [
            s for s in scenarios
            if any(SuccessSignal(sig) in judge_dependent_signals for sig in (s.success_signals or ()))
        ]
        if needs_judge:
            domains_affected = sorted({s.domain for s in needs_judge})
            logger.info(
                f"WARNING: {len(needs_judge)} of {len(scenarios)} queued scenarios reference "
                f"judge-dependent signals (JUDGE_PASSED / NL_ASSERTION) in their success_signals "
                f"but no LLM judge is configured (domains: {domains_affected}). Those scenarios "
                f"may produce is_successful='N/A' if no deterministic signal in their whitelist "
                f"is applicable. Pass --judge-url / --judge-model / --judge-api-key to enable."
            )

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
    # All run-level signal buckets + per-domain breakdowns + token totals live
    # on a single accumulator object. Local-alias every bucket back into the
    # function scope so the existing 100+ inline ``*_results.append(...)`` and
    # ``per_domain_*.setdefault(...)`` call sites continue to work unchanged —
    # the aliases share the same list/dict identity with the aggregator
    # fields. The aggregator's ``add_scenario`` method centralizes the
    # equivalent logic for the resume-skip path (loaded-from-disk metrics).
    aggregator = RunAggregator()
    success_results = aggregator.success_results
    action_match_results = aggregator.action_match_results
    judge_score_results = aggregator.judge_score_results
    judge_pass_results = aggregator.judge_pass_results
    db_state_results = aggregator.db_state_results
    nl_assertion_results = aggregator.nl_assertion_results
    db_state_assertion_results = aggregator.db_state_assertion_results
    # Per-domain buckets keyed by ``scenario.name.split('__')[0]`` so a mixed run
    # (eva_airline + tau2_airline + retail + …) reports rates per source.
    per_domain_success = aggregator.per_domain_success
    per_domain_action_match = aggregator.per_domain_action_match
    per_domain_judge_score = aggregator.per_domain_judge_score
    per_domain_judge_pass = aggregator.per_domain_judge_pass
    per_domain_db_state = aggregator.per_domain_db_state
    per_domain_nl_assertion = aggregator.per_domain_nl_assertion
    per_domain_db_state_assertion = aggregator.per_domain_db_state_assertion
    run_token_usage = aggregator.run_token_usage

    def _classify_resume_state(scenario_dir: str) -> str:
        """Return ``"completed"`` / ``"in_flight"`` / ``"fresh"`` based on disk state."""
        if not os.path.isdir(scenario_dir):
            return "fresh"
        mf = os.path.join(scenario_dir, "metrics.json")
        if not os.path.exists(mf):
            return "in_flight"
        try:
            with open(mf) as f:
                json.load(f)
            return "completed"
        except (json.JSONDecodeError, OSError):
            return "in_flight"

    # Pre-loop classification pass: any scenario whose subdir already has a
    # finalized ``metrics.json`` is loaded into ``all_results`` (and its
    # signals folded into the run-level buckets) WITHOUT being re-run.
    # In-flight subdirs (started, no metrics.json) are moved aside so the
    # re-run starts clean. ``loaded_names`` is a set used by the main loop
    # to skip the run-scenario step for completed entries.
    resume_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    loaded_names = set()
    for scenario in scenarios:
        scen_dir = os.path.join(output_dir, scenario.name)
        state = _classify_resume_state(scen_dir)
        if state == "completed":
            with open(os.path.join(scen_dir, "metrics.json")) as f:
                m = json.load(f)
            # Refresh the scenario_directory in case the dir was moved.
            m["scenario_directory"] = scen_dir
            all_results.append(m)
            aggregator.add_scenario(m, scenario.name.split("__", 1)[0])
            loaded_names.add(scenario.name)
            logger.info(f"[SKIP] {scenario.name}: already complete, loaded metrics.json from disk.")
        elif state == "in_flight":
            backup = f"{scen_dir}.killed.{resume_timestamp}"
            os.rename(scen_dir, backup)
            # Leave a marker so the eval-result-analyzer skill won't mistake
            # the moved directory for a real scenario subdir.
            open(os.path.join(backup, "__KILLED__"), "w").close()
            logger.info(
                f"[CLEANUP] {scenario.name}: subdir was in-flight (no metrics.json); "
                f"moved to {os.path.basename(backup)}/ and will re-run."
            )

    def _signal_passes(value, *, pass_rate_threshold: float = 1.0):
        """Normalize a per-scenario signal to True / False / None (N/A).

        Used by the composite ``is_successful`` strict-conjunction logic
        below — every applicable signal must pass for the scenario to be
        considered successful overall.

        - ``None`` or ``"N/A"`` → ``None`` (signal not applicable to this scenario).
        - ``bool`` → returned as-is.
        - ``float`` (pass rates) → True iff ``value >= pass_rate_threshold``.
          Default ``pass_rate_threshold=1.0`` enforces "every assertion
          must pass" for ``db_state_assertion_pass_rate`` and
          ``nl_assertion_pass_rate``. Loosen via per-call override if
          empirical noise warrants.
        """
        if value is None or value == "N/A":
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return float(value) >= pass_rate_threshold
        return None

    for idx, scenario in enumerate(scenarios):
        # Resume short-circuit: scenarios with a finalized metrics.json were
        # already loaded into ``all_results`` and the run-level buckets above,
        # so skip the live execution. Per-scenario artifacts on disk (bridge_log,
        # bot_logs, judge_result) remain unchanged.
        if scenario.name in loaded_names:
            logger.info(f"[SKIP {idx+1}/{len(scenarios)}] {scenario.name} (loaded from previous run)")
            continue

        logger.info(f"{'='*80}")
        logger.info(f"Starting Scenario {idx+1}/{len(scenarios)}: {scenario.name}")
        logger.info(f"{'='*80}\n")

        # Create scenario-specific directory
        scenario_dir = os.path.join(output_dir, scenario.name)
        os.makedirs(scenario_dir, exist_ok=True)

        # Bridge accepts the scenario instance directly — no intermediate
        # dict serialization. The bridge calls scenario methods
        # (``get_user_prompt``, ``setup_shared_state``,
        # ``initialization_actions``, ``sync_state``, etc.) as the
        # single source of truth.
        logger.info(f"Preparing for scenario: {scenario.name}...")
        await bridge.prepare_for_scenario(scenario, scenario_dir)
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
        # the scenario carries nl_assertions; remains None when no judge ran).
        scenario_nl_pass_rate: Optional[float] = None

        # ----- Signal 1: deterministic action-list match ---------------------
        # Computed independently of the LLM judge — both signals are
        # orthogonal. The composite ``is_successful`` (computed at the end of
        # this block) requires ALL applicable signals to pass.
        if not os.path.exists(reference_file):
            logger.info(f"Reference file {reference_file} not found; action-match is N/A.")
            is_action_match = "N/A"
        elif not os.path.exists(prediction_file):
            logger.info(f"Prediction file {prediction_file} not found; action-match=False.")
            is_action_match = False
        else:
            scenario_disallow_extra = strict_match or getattr(scenario, "disallow_extra_items", False)
            is_action_match = check_if_task_success(
                reference=reference_file,
                prediction=prediction_file,
                ignore_capitalization=getattr(scenario, "ignore_capitalization", False),
                ignore_punctuation=getattr(scenario, "ignore_punctuation", False),
                clean_text=getattr(scenario, "clean_text", False),
                disallow_extra_items=scenario_disallow_extra,
            )
        if isinstance(is_action_match, bool):
            action_match_results.append(is_action_match)
            per_domain_action_match.setdefault(domain, []).append(is_action_match)

        # ----- Signal 2: LLM judge (independent of action-list) --------------
        # Runs whenever judge is configured AND both reference + prediction
        # files exist. Produces ``judge_score`` (float) and (when threshold
        # is set) ``judge_passed`` (bool). Also produces per-assertion
        # nl_assertion verdicts when the scenario carries nl_assertions.
        judge_score: Optional[float] = None
        judge_passed: Optional[bool] = None
        if judge is not None and os.path.exists(reference_file) and os.path.exists(prediction_file):
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
                    "calling judge_scenario without agent_context_history."
                )
            # User-sim context: essential for dual-side domains (telecom)
            # where reference actions with ``side="user"`` are executed
            # by the user-sim itself, not the agent. The judge needs
            # this to confirm those actions actually happened. For
            # single-side domains (eva / airline / retail) the file
            # still exists but contributes mostly chitchat — harmless.
            user_context_file = os.path.join(scenario_dir, "bot_logs_user", "llm_context.json")
            user_context_history = None
            if os.path.exists(user_context_file):
                with open(user_context_file, "r") as f:
                    user_context_history = json.load(f)
            scenario_nl_assertions = getattr(scenario, "nl_assertions", None)
            result = judge.judge_scenario(
                reference=ref_content,
                prediction=pred_content,
                agent_context_history=agent_context_history,
                user_context_history=user_context_history,
                nl_assertions=scenario_nl_assertions,
            )
            with open(os.path.join(scenario_dir, "judge_result.json"), "w") as f:
                json.dump(result, f, indent=2)
            judge_score = float(result["score"])
            judge_score_results.append(judge_score)
            per_domain_judge_score.setdefault(domain, []).append(judge_score)
            if judge_threshold is not None:
                judge_passed = judge_score >= judge_threshold
                judge_pass_results.append(judge_passed)
                per_domain_judge_pass.setdefault(domain, []).append(judge_passed)
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

        # Collect metrics for this scenario
        metrics = bridge.get_metrics()
        metrics["scenario_name"] = scenario.name
        metrics["scenario_directory"] = scenario_dir
        metrics["scenario_duration"] = (scenario_end - scenario_start).total_seconds()
        metrics["is_action_match"] = is_action_match
        if judge_score is not None:
            metrics["judge_score"] = judge_score
        if judge_passed is not None:
            metrics["judge_passed"] = judge_passed
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

        # ----- Composite is_successful (delegated to scenario) ---------------
        # The scenario's ``success_signals`` whitelist drives which of the 5
        # signals gate the verdict; the rest are computed and saved but land
        # in ``success_breakdown.excluded`` (informational). Default behavior
        # is strict-AND over applicable whitelist signals — see
        # ``Scenario.compute_is_successful``. Domains override the whitelist
        # to reflect their solution-space shape (e.g. tau2_telecom drops
        # ``DB_STATE_MATCH`` and ``ACTION_MATCH`` because open-spec).
        signal_checks = {
            SuccessSignal.ACTION_MATCH: _signal_passes(metrics.get("is_action_match")),
            SuccessSignal.DB_STATE_MATCH: _signal_passes(metrics.get("db_state_match")),
            SuccessSignal.DB_STATE_ASSERTION: _signal_passes(metrics.get("db_state_assertion_pass_rate")),
            SuccessSignal.NL_ASSERTION: _signal_passes(metrics.get("nl_assertion_pass_rate")),
            SuccessSignal.JUDGE_PASSED: metrics.get("judge_passed"),  # already bool/None
        }
        whitelist = {SuccessSignal(s) for s in (scenario.success_signals or ())}
        metrics["success_breakdown"] = {
            "passed": [str(k) for k, v in signal_checks.items() if k in whitelist and v is True],
            "failed": [str(k) for k, v in signal_checks.items() if k in whitelist and v is False],
            "not_applicable": [str(k) for k, v in signal_checks.items() if k in whitelist and v is None],
            "excluded": [str(k) for k, v in signal_checks.items() if k not in whitelist and v is not None],
        }
        metrics["is_successful"] = scenario.compute_is_successful(signal_checks)
        if isinstance(metrics["is_successful"], bool):
            success_results.append(metrics["is_successful"])
            per_domain_success.setdefault(domain, []).append(metrics["is_successful"])

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
        logger.info(f"    Action-list match: {metrics['is_action_match']}")
        if "db_state_match" in metrics:
            logger.info(f"    DB-state match: {metrics['db_state_match']}")
        if "db_state_assertion_pass_rate" in metrics:
            logger.info(
                f"    DB-state-assertion pass rate: {metrics['db_state_assertion_pass_rate']*100:.2f}%"
            )
        if "nl_assertion_pass_rate" in metrics:
            logger.info(f"    NL-assertion pass rate: {metrics['nl_assertion_pass_rate']*100:.2f}%")
        if "judge_score" in metrics:
            logger.info(f"    Judge score: {metrics['judge_score']:.2f}")
        if "judge_passed" in metrics:
            logger.info(f"    Judge passed: {metrics['judge_passed']}")
        if metrics["success_breakdown"]["failed"]:
            logger.info(f"    Failed signals: {metrics['success_breakdown']['failed']}")
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
    # Composite success rate — fraction of scenarios where EVERY applicable
    # signal passed (strict conjunction). See ``success_breakdown`` per
    # scenario for which signal failed.
    success_rate = sum(success_results) / len(success_results) if len(success_results) > 0 else 0
    # Per-signal rates so the headline can be broken down. Denominators
    # are per-signal: only scenarios that opted into each signal count.
    action_match_rate = (
        sum(action_match_results) / len(action_match_results) if action_match_results else None
    )
    judge_score_mean = (
        sum(judge_score_results) / len(judge_score_results) if judge_score_results else None
    )
    judge_pass_rate = (
        sum(judge_pass_results) / len(judge_pass_results) if judge_pass_results else None
    )
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
            f.write(f"    Action-list match: {result['is_action_match']}\n")
            if "db_state_match" in result:
                f.write(f"    DB-state match: {result['db_state_match']}\n")
            if "db_state_assertion_pass_rate" in result:
                f.write(
                    f"    DB-state-assertion pass rate: "
                    f"{result['db_state_assertion_pass_rate']*100:.2f}%\n"
                )
            if "nl_assertion_pass_rate" in result:
                f.write(f"    NL-assertion pass rate: {result['nl_assertion_pass_rate']*100:.2f}%\n")
            if "judge_score" in result:
                f.write(f"    Judge score: {result['judge_score']:.2f}\n")
            if "judge_passed" in result:
                f.write(f"    Judge passed: {result['judge_passed']}\n")
            failed_signals = result["success_breakdown"]["failed"]
            if failed_signals:
                f.write(f"    Failed signals: {', '.join(failed_signals)}\n")
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

        # Composite (strict-conjunction) success rate first — the headline
        # number — followed by per-signal breakdown so operators can see
        # which dimension(s) dragged the conjunction down.
        if success_results:
            f.write(
                f"\n\nOverall Success Rate: {success_rate*100:.2f}% "
                f"({sum(success_results)}/{len(success_results)} scenarios — strict conjunction "
                f"across all applicable signals)\n"
            )
        else:
            f.write("\n\nOverall Success Rate: N/A (no scenarios had any applicable signal)\n")

        f.write("\nPer-signal pass rates:\n")
        if action_match_rate is not None:
            f.write(
                f"  Action-list match:           {action_match_rate*100:6.2f}% "
                f"({sum(action_match_results)}/{len(action_match_results)} scenarios)\n"
            )
        if db_state_success_rate is not None:
            f.write(
                f"  DB-State match:              {db_state_success_rate*100:6.2f}% "
                f"({sum(db_state_results)}/{len(db_state_results)} scenarios)\n"
            )
        if db_state_assertion_success_rate is not None:
            f.write(
                f"  DB-State-Assertion pass:     {db_state_assertion_success_rate*100:6.2f}% "
                f"({sum(db_state_assertion_results)}/{len(db_state_assertion_results)} predicates)\n"
            )
        if nl_assertion_success_rate is not None:
            f.write(
                f"  NL-Assertion pass:           {nl_assertion_success_rate*100:6.2f}% "
                f"({sum(nl_assertion_results)}/{len(nl_assertion_results)} assertions)\n"
            )
        if judge_score_mean is not None:
            f.write(
                f"  Judge score mean:            {judge_score_mean:.3f} "
                f"(across {len(judge_score_results)} scenarios)\n"
            )
        if judge_pass_rate is not None:
            f.write(
                f"  Judge passed (>= threshold): {judge_pass_rate*100:6.2f}% "
                f"({sum(judge_pass_results)}/{len(judge_pass_results)} scenarios)\n"
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
            f"({sum(success_results)}/{len(success_results)} scenarios — "
            f"strict conjunction across all applicable signals)"
        )
    else:
        logger.info("Overall Success Rate: N/A (no scenarios had any applicable signal)")
    if action_match_rate is not None:
        logger.info(
            f"  Action-list match: {action_match_rate*100:.2f}% "
            f"({sum(action_match_results)}/{len(action_match_results)} scenarios)"
        )
    if db_state_success_rate is not None:
        logger.info(
            f"  DB-State match: {db_state_success_rate*100:.2f}% "
            f"({sum(db_state_results)}/{len(db_state_results)} scenarios)"
        )
    if db_state_assertion_success_rate is not None:
        logger.info(
            f"  DB-State-Assertion pass: {db_state_assertion_success_rate*100:.2f}% "
            f"({sum(db_state_assertion_results)}/{len(db_state_assertion_results)} predicates)"
        )
    if nl_assertion_success_rate is not None:
        logger.info(
            f"  NL-Assertion pass: {nl_assertion_success_rate*100:.2f}% "
            f"({sum(nl_assertion_results)}/{len(nl_assertion_results)} assertions)"
        )
    if judge_score_mean is not None:
        logger.info(
            f"  Judge score mean: {judge_score_mean:.3f} "
            f"(across {len(judge_score_results)} scenarios)"
        )
    if judge_pass_rate is not None:
        logger.info(
            f"  Judge passed: {judge_pass_rate*100:.2f}% "
            f"({sum(judge_pass_results)}/{len(judge_pass_results)} scenarios)"
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

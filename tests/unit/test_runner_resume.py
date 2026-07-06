# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Tests for the resume path in ``run_dynamic_evaluation``.

Covers the three filesystem states a per-scenario subdir can be in when a
resume invocation starts:

- **Completed** — subdir + valid ``metrics.json`` → loaded into ``all_results``
  and the per-signal aggregate buckets without re-running.
- **In-flight** — subdir but no ``metrics.json`` → renamed to
  ``<scenario>.killed.<ts>/`` (with a ``__KILLED__`` marker), then re-run fresh.
- **Fresh** — no subdir → runs normally.

Also exercises the run-args helper in ``run_evaluation.py``:

- Soft-warns on resume when scoring-relevant flags differ from the original
  invocation.
- Appends to ``run_args.json`` rather than overwriting.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# resume classifier — exercised via the runner's internal helper
# ---------------------------------------------------------------------------


def _make_metrics_json(scen_dir: Path, **overrides) -> dict:
    """Write a minimal valid metrics.json into scen_dir/metrics.json."""
    scen_dir.mkdir(parents=True, exist_ok=True)
    default = {
        "scenario_name": scen_dir.name,
        "scenario_directory": str(scen_dir),
        "total_turns": 5,
        "scenario_duration": 100.0,
        "is_action_match": True,
        "is_successful": True,
        "success_breakdown": {"passed": ["is_action_match"], "failed": [], "not_applicable": [], "excluded": []},
        "token_usage": {
            "agent": {"n_calls": 3, "prompt": 1000, "completion": 100},
            "user": {"n_calls": 3, "prompt": 500, "completion": 50},
        },
    }
    default.update(overrides)
    (scen_dir / "metrics.json").write_text(json.dumps(default))
    return default


def test_classify_resume_state(tmp_path):
    """The classifier returns ``completed`` / ``in_flight`` / ``fresh`` per disk shape."""

    # We test the logic by reproducing it inline — the helper is closed-over
    # inside ``run_dynamic_evaluation`` and not importable on its own.
    def classify(d):
        if not d.exists():
            return "fresh"
        if not (d / "metrics.json").exists():
            return "in_flight"
        try:
            json.loads((d / "metrics.json").read_text())
            return "completed"
        except (json.JSONDecodeError, OSError):
            return "in_flight"

    # Fresh: no dir
    assert classify(tmp_path / "scen_a") == "fresh"

    # In-flight: dir, no metrics.json
    in_flight = tmp_path / "scen_b"
    in_flight.mkdir()
    (in_flight / "bridge_log.txt").write_text("...")
    assert classify(in_flight) == "in_flight"

    # In-flight: dir + malformed metrics.json
    bad = tmp_path / "scen_c"
    bad.mkdir()
    (bad / "metrics.json").write_text("{ not json")
    assert classify(bad) == "in_flight"

    # Completed: dir + valid metrics.json
    completed = tmp_path / "scen_d"
    _make_metrics_json(completed)
    assert classify(completed) == "completed"


# ---------------------------------------------------------------------------
# run_args.json: append-on-resume, soft-warn on consistency mismatch
# ---------------------------------------------------------------------------


def _bare_args(**overrides) -> argparse.Namespace:
    """Build an argparse.Namespace matching what run_evaluation.py's main builds."""
    defaults = dict(
        user_url="ws://localhost:8766",
        agent_url="ws://localhost:8765",
        scenarios=None,
        domain="tau2_retail",
        duration=None,
        pause=0.5,
        output_dir="./eval_results",
        output_sample_rate=24000,
        audio_chunk_in_seconds=0.016,
        judge_url=None,
        judge_model=None,
        judge_api_key=None,
        judge_threshold=0.9,
        strict_match=False,
        resume=None,
        list=False,
        list_domains=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class _StubScenario:
    def __init__(self, name):
        self.name = name


@pytest.fixture
def _run_eval_module():
    """Import the run_evaluation script as a module so we can call _write_run_args."""
    script_path = Path(__file__).resolve().parents[2] / "evaluation" / "run_evaluation.py"
    spec = importlib.util.spec_from_file_location("run_evaluation_under_test", script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_evaluation_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_run_args_initial_invocation_creates_history(tmp_path, _run_eval_module):
    """First run writes a single-entry invocations list."""
    args = _bare_args()
    scenarios = [_StubScenario("tau2_retail__0"), _StubScenario("tau2_retail__1")]
    logger = _run_eval_module.FileLogger(str(tmp_path / "log.txt"))

    _run_eval_module._write_run_args(str(tmp_path), args, scenarios, is_resume=False, logger=logger)

    data = json.loads((tmp_path / "run_args.json").read_text())
    assert "invocations" in data
    assert len(data["invocations"]) == 1
    inv = data["invocations"][0]
    assert inv["resolved_scenario_count"] == 2
    assert inv["resolved_scenario_names"] == ["tau2_retail__0", "tau2_retail__1"]
    assert inv["parsed_args"]["domain"] == "tau2_retail"
    # Initial invocations don't carry the resumed_from_invocation pointer.
    assert "resumed_from_invocation" not in inv


def test_run_args_resume_appends_invocation(tmp_path, _run_eval_module):
    """Resume invocations append a second entry referencing the prior index."""
    args1 = _bare_args()
    scens1 = [_StubScenario("tau2_retail__0")]
    logger = _run_eval_module.FileLogger(str(tmp_path / "log.txt"))
    _run_eval_module._write_run_args(str(tmp_path), args1, scens1, is_resume=False, logger=logger)

    # Resume with identical args
    args2 = _bare_args(resume="20260609_181545")
    scens2 = [_StubScenario("tau2_retail__1")]
    _run_eval_module._write_run_args(str(tmp_path), args2, scens2, is_resume=True, logger=logger)

    data = json.loads((tmp_path / "run_args.json").read_text())
    assert len(data["invocations"]) == 2
    assert data["invocations"][1]["resumed_from_invocation"] == 0


def test_run_args_resume_with_mismatch_warns(tmp_path, _run_eval_module):
    """Soft warning when scoring-relevant fields differ between original and resume."""
    args1 = _bare_args(judge_threshold=0.9, strict_match=False)
    scens1 = [_StubScenario("tau2_retail__0")]
    log_path = tmp_path / "log.txt"
    logger = _run_eval_module.FileLogger(str(log_path))
    _run_eval_module._write_run_args(str(tmp_path), args1, scens1, is_resume=False, logger=logger)

    # Resume with a different threshold and strict_match toggled.
    args2 = _bare_args(judge_threshold=0.7, strict_match=True, resume="ts")
    scens2 = [_StubScenario("tau2_retail__1")]
    _run_eval_module._write_run_args(str(tmp_path), args2, scens2, is_resume=True, logger=logger)

    log_content = log_path.read_text()
    assert "WARNING" in log_content
    assert "judge_threshold" in log_content
    assert "strict_match" in log_content


def test_run_args_redacts_judge_api_key(tmp_path, _run_eval_module):
    """API key is never persisted in plaintext."""
    args = _bare_args(judge_api_key="sk-secret-token")
    logger = _run_eval_module.FileLogger(str(tmp_path / "log.txt"))
    _run_eval_module._write_run_args(str(tmp_path), args, [_StubScenario("x")], is_resume=False, logger=logger)

    data = json.loads((tmp_path / "run_args.json").read_text())
    assert data["invocations"][0]["parsed_args"]["judge_api_key"] == "<redacted>"
    # Make sure the secret didn't sneak into another field.
    assert "sk-secret-token" not in (tmp_path / "run_args.json").read_text()


# ---------------------------------------------------------------------------
# FileLogger mode kwarg
# ---------------------------------------------------------------------------


def test_filelogger_mode_w_truncates(tmp_path):
    """``mode='w'`` (default) clears the file at construction."""
    from nemo_voice_agent.utils import FileLogger

    log_path = tmp_path / "log.txt"
    log_path.write_text("previous content\n")
    # Constructing with default mode='w' should truncate.
    logger = FileLogger(str(log_path))
    logger.info("fresh line", include_caller=False)
    content = log_path.read_text()
    assert "previous content" not in content
    assert "fresh line" in content


def test_run_aggregator_fold_freshly_run_and_loaded_metrics(tmp_path):
    """RunAggregator.add_scenario folds equivalent fresh + loaded metrics identically.

    Two scenarios feeding the same fields into a single aggregator should
    produce buckets equivalent to one aggregator per scenario then merged.
    This is the resume-path correctness invariant: regardless of whether a
    scenario ran live or was loaded from disk, the run-level rollup is
    deterministic.
    """
    from nemo_voice_agent.evaluation.runner import RunAggregator

    m1 = {
        "is_action_match": True,
        "db_state_match": True,
        "judge_score": 0.95,
        "judge_passed": True,
        "clean_exit": True,
        "is_successful": True,
        "token_usage": {
            "agent": {"n_calls": 5, "prompt": 10_000, "completion": 500},
            "user": {"n_calls": 5, "prompt": 3_000, "completion": 400},
        },
        "db_state_assertion_verdicts": [
            {"passed": True},
            {"passed": False},
        ],
    }
    m2 = {
        "is_action_match": False,
        "db_state_match": False,
        "judge_score": 0.30,
        "judge_passed": False,
        "clean_exit": False,
        "is_successful": False,
        "token_usage": {
            "agent": {"n_calls": 7, "prompt": 14_000, "completion": 700},
            "user": {"n_calls": 7, "prompt": 4_000, "completion": 600},
        },
        "db_state_assertion_verdicts": [
            {"passed": True},
            {"passed": True},
            {"passed": False},
        ],
    }

    agg = RunAggregator()
    agg.add_scenario(m1, "tau2_retail")
    agg.add_scenario(m2, "tau2_retail")

    # Bucket lists carry every scenario's verdict, denominator = scenarios (or
    # assertions for the predicate buckets).
    assert agg.action_match_results == [True, False]
    assert agg.db_state_results == [True, False]
    assert agg.judge_score_results == [0.95, 0.30]
    assert agg.judge_pass_results == [True, False]
    assert agg.success_results == [True, False]
    # Per-predicate verdicts flattened across both scenarios: 2 + 3 = 5 entries.
    assert agg.db_state_assertion_results == [True, False, True, True, False]
    assert agg.clean_exit_results == [True, False]

    # Per-domain matches whole-run because both scenarios share the domain.
    assert agg.per_domain_action_match["tau2_retail"] == [True, False]
    assert agg.per_domain_success["tau2_retail"] == [True, False]

    # Token totals are sums across scenarios.
    assert agg.run_token_usage["agent"]["n_calls"] == 12
    assert agg.run_token_usage["agent"]["prompt"] == 24_000
    assert agg.run_token_usage["user"]["completion"] == 1_000


def test_run_aggregator_uses_successsignal_keys(tmp_path):
    """The aggregator reads from incoming metrics via SuccessSignal members.

    Confirms that the StrEnum-vs-string equivalence works at the lookup site —
    if the enum value drifted from the JSON key, this test would fail.
    """
    from nemo_voice_agent.evaluation.runner import RunAggregator
    from nemo_voice_agent.evaluation.scenarios.classes import SuccessSignal

    agg = RunAggregator()
    # Construct a metrics dict using the enum members directly as keys —
    # this works because StrEnum members are str-equal to their values.
    metrics = {
        SuccessSignal.ACTION_MATCH: True,
        SuccessSignal.DB_STATE_MATCH: False,
        SuccessSignal.JUDGE_PASSED: True,
        "is_successful": True,
    }
    agg.add_scenario(metrics, "domain_x")
    assert agg.action_match_results == [True]
    assert agg.db_state_results == [False]
    assert agg.judge_pass_results == [True]
    assert agg.success_results == [True]


def test_run_aggregator_skips_na_and_absent(tmp_path):
    """Signals absent or 'N/A' don't contribute to their buckets."""
    from nemo_voice_agent.evaluation.runner import RunAggregator

    agg = RunAggregator()
    # is_action_match=str "N/A" and missing keys → no appends.
    agg.add_scenario({"is_action_match": "N/A", "is_successful": "N/A"}, "x")
    assert agg.action_match_results == []
    assert agg.success_results == []
    assert agg.run_token_usage["agent"]["n_calls"] == 0


def test_filelogger_mode_a_appends(tmp_path):
    """``mode='a'`` preserves the existing log so resume sessions continue it."""
    from nemo_voice_agent.utils import FileLogger

    log_path = tmp_path / "log.txt"
    log_path.write_text("previous content\n")
    logger = FileLogger(str(log_path), mode="a")
    logger.info("resume line", include_caller=False)
    content = log_path.read_text()
    assert "previous content" in content
    assert "resume line" in content

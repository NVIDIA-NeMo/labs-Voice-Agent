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

"""Per-domain aggregation conventions used by ``run_dynamic_evaluation`` (M1.F).

The runner is async and tightly coupled to ``VoiceAgentEvaluationBridge``; rather
than mock the entire pipeline, these tests exercise the two pure-function
invariants the per-domain code path depends on:

  1. ``scenario.name.split("__", 1)[0]`` gives the correct domain prefix for
     every existing and future-tau2 scenario-name convention.
  2. ``per_domain_X.setdefault(domain, []).append(value)`` produces the right
     per-bucket rates when used as the runner uses it.

If either invariant breaks, the runner's mixed-domain reporting silently goes
wrong. These tests lock the convention before tau2 scenarios start landing.
"""

import pytest


# ---------------------------------------------------------------------------
# Invariant 1: domain-prefix extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario_name, expected_domain",
    [
        # Existing eva conventions
        ("eva_airline__1_1_2", "eva_airline"),
        ("eva_airline__cancellation_refund", "eva_airline"),
        ("customer_service__billing_dispute", "customer_service"),
        ("qa__capital_france", "qa"),
        ("restaurant__pizza_pepperoni", "restaurant"),
        # Singletons without "__" — bucket under the full name
        ("fastbite", "fastbite"),
        ("simple_qa_1", "simple_qa_1"),
        # Future tau2 conventions
        ("tau2_airline__0", "tau2_airline"),
        ("tau2_retail__42", "tau2_retail"),
        # Tau2 telecom uses ids like "[mobile_data_issue]user_abroad_roaming_enabled_off[PERSONA:None]"
        # — when we register these, the "__" separator runs before the bracketed id.
        ("tau2_telecom__some_complex_id", "tau2_telecom"),
        # Edge case: scenario id itself contains "__" — only the first split is taken.
        ("tau2_airline__sub__id", "tau2_airline"),
    ],
)
def test_scenario_name_split_yields_correct_domain(scenario_name, expected_domain):
    assert scenario_name.split("__", 1)[0] == expected_domain


# ---------------------------------------------------------------------------
# Invariant 2: per-domain aggregation produces correct rates
# ---------------------------------------------------------------------------


def _aggregate(results_by_name):
    """Mirror of runner.py's per-domain tracking + rate computation.

    Kept as a local helper rather than imported from runner.py because the
    runner interleaves it with async bridge calls. This function captures the
    pure-algorithmic part of the convention.
    """
    per_domain: dict = {}
    for name, value in results_by_name:
        domain = name.split("__", 1)[0]
        per_domain.setdefault(domain, []).append(value)
    rates = {}
    for d, results in per_domain.items():
        rates[d] = sum(results) / len(results) if results else 0
    return per_domain, rates


def test_aggregate_two_prefixes_split_correctly():
    results = [
        ("eva_airline__1_1_2", True),
        ("eva_airline__1_1_3", False),
        ("tau2_airline__0", True),
        ("tau2_airline__1", True),
    ]
    buckets, rates = _aggregate(results)
    assert set(buckets.keys()) == {"eva_airline", "tau2_airline"}
    assert buckets["eva_airline"] == [True, False]
    assert buckets["tau2_airline"] == [True, True]
    assert rates["eva_airline"] == 0.5
    assert rates["tau2_airline"] == 1.0


def test_aggregate_handles_mixed_singletons_and_prefixed():
    results = [
        ("eva_airline__1_1_2", True),
        ("fastbite", False),
        ("simple_qa_1", True),
        ("simple_qa_2", True),
    ]
    buckets, rates = _aggregate(results)
    # Singletons don't merge into a common bucket — each gets its own.
    assert set(buckets.keys()) == {"eva_airline", "fastbite", "simple_qa_1", "simple_qa_2"}
    assert rates["eva_airline"] == 1.0
    assert rates["fastbite"] == 0.0
    assert rates["simple_qa_1"] == 1.0
    assert rates["simple_qa_2"] == 1.0


def test_aggregate_handles_float_judge_scores():
    # When judge_threshold is None, success_results stores raw float scores.
    # The aggregation should still produce sensible per-domain means.
    results = [
        ("tau2_retail__a", 0.8),
        ("tau2_retail__b", 0.6),
        ("tau2_airline__c", 1.0),
    ]
    buckets, rates = _aggregate(results)
    assert rates["tau2_retail"] == pytest.approx(0.7)
    assert rates["tau2_airline"] == 1.0


def test_aggregate_only_one_domain_is_legal():
    # The runner suppresses the per-domain block when there's only one domain
    # (it would duplicate the overall rate). The aggregation itself should
    # still produce a correct single-key dict.
    results = [
        ("eva_airline__a", True),
        ("eva_airline__b", True),
    ]
    buckets, _ = _aggregate(results)
    assert list(buckets.keys()) == ["eva_airline"]
    assert len(buckets) == 1

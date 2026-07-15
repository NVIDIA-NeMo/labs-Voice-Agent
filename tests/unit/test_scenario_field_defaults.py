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

"""Tests that the optional ``Scenario`` fields don't regress existing scenarios.

Three optional fields exist on ``Scenario`` (``db_state_assertions``,
``initialization_actions``, ``expected_user_db``), all defaulting to ``None``
so eva_airline / tau2_airline / tau2_retail scenarios continue to work
without modification. This suite locks that invariant: instantiate one
scenario from each domain and assert every new field is ``None``.

When a future domain (e.g. tau2_telecom) adds scenarios that opt into
these fields, those scenarios should NOT show up here — only the
single-side domains. Add a separate domain-specific test file for that
domain's field assertions.
"""

import nemo_voice_agent.evaluation.scenarios.data  # noqa: F401 — triggers registration
from nemo_voice_agent.evaluation.scenarios import ALL_EVAL_SCENARIOS
from nemo_voice_agent.evaluation.scenarios.classes import Scenario


def test_scenario_base_class_defaults_to_none_for_new_fields():
    """The optional fields default to ``None`` on the bare ``Scenario`` base
    so any existing caller that didn't pass these kwargs is unaffected."""
    s = Scenario(name="dummy", description="d")
    assert s.db_state_assertions is None
    assert s.initialization_actions is None
    assert s.expected_user_db is None


def test_eva_airline_scenario_has_no_new_fields_populated():
    # Pick any registered eva_airline scenario — the field defaults are a
    # property of the base class, so any one suffices. Names look like
    # ``eva_airline__<short_slug>`` (e.g. ``eva_airline__voluntary_date_change``).
    eva_name = next(n for n in ALL_EVAL_SCENARIOS if n.startswith("eva_airline__"))
    inst = ALL_EVAL_SCENARIOS[eva_name]()
    assert inst.db_state_assertions is None
    assert inst.initialization_actions is None
    assert inst.expected_user_db is None


def test_tau2_airline_scenario_has_no_new_fields_populated():
    inst = ALL_EVAL_SCENARIOS["tau2_airline__0"]()
    assert inst.db_state_assertions is None
    assert inst.initialization_actions is None
    assert inst.expected_user_db is None


def test_tau2_retail_scenario_has_no_new_fields_populated():
    inst = ALL_EVAL_SCENARIOS["tau2_retail__0"]()
    assert inst.db_state_assertions is None
    assert inst.initialization_actions is None
    assert inst.expected_user_db is None


def test_no_single_side_scenario_carries_db_state_assertions():
    """Scan every registered scenario; only telecom (dual-side) scenarios
    should opt into ``db_state_assertions`` / ``initialization_actions`` /
    ``expected_user_db``. This catches accidental field assignments on
    single-side domains (eva / tau2_airline / tau2_retail)."""
    # Allowlist: telecom scenarios (both manual + workflow variants) are
    # the only ones expected to opt in.
    allowed_prefixes = ("tau2_telecom__", "tau2_telecom_workflow__")
    offenders = []
    for name, cls in ALL_EVAL_SCENARIOS.items():
        if any(name.startswith(p) for p in allowed_prefixes):
            continue
        try:
            inst = cls()
        except Exception:
            # Some scenario classes need extra fixtures to instantiate;
            # skip those — they'll surface in their own test suites.
            continue
        if (
            inst.db_state_assertions is not None
            or inst.initialization_actions is not None
            or inst.expected_user_db is not None
        ):
            offenders.append(name)
    assert offenders == [], f"Single-side scenarios accidentally opted into dual-side fields: {offenders}"

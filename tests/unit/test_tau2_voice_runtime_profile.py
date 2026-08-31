# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validation and compatibility tests for checked-in tau voice profiles."""

import json
from copy import deepcopy

import pytest

from nemo_voice_agent.evaluation import get_eval_data_root
from nemo_voice_agent.evaluation.runtime_profile import (
    SpeechComplexity,
    parse_tau_voice_profile,
)
from nemo_voice_agent.evaluation.scenarios.data.tau2_common import Tau2BaseScenario


_DOMAINS = ("tau2_airline", "tau2_retail", "tau2_telecom")
_EXPECTED_TASK_COUNTS = {"tau2_airline": 50, "tau2_retail": 114, "tau2_telecom": 114}
_EXPECTED_PRESETS = {preset.value for preset in SpeechComplexity}


def _load_voice_tasks(domain: str) -> dict:
    path = get_eval_data_root() / domain / "tasks_voice.json"
    return json.loads(path.read_text())["configs"]


def _representative_profile(preset: str = "regular") -> tuple[str, str, dict]:
    domain = "tau2_airline"
    task_id, task_config = next(iter(_load_voice_tasks(domain).items()))
    return domain, task_id, deepcopy(task_config["configs"][preset])


def test_all_2224_checked_in_tau_voice_profiles_parse_strictly():
    """Every shipped task/preset combination matches the canonical schema."""
    parsed_count = 0
    for domain in _DOMAINS:
        tasks = _load_voice_tasks(domain)
        assert len(tasks) == _EXPECTED_TASK_COUNTS[domain]
        for task_id, task_config in tasks.items():
            profiles = task_config["configs"]
            assert set(profiles) == _EXPECTED_PRESETS
            for preset, raw in profiles.items():
                profile = parse_tau_voice_profile(
                    raw,
                    domain=domain,
                    task_id=task_id,
                    expected_complexity=preset,
                )
                assert profile.complexity.value == preset
                assert profile.source_domain == domain
                assert profile.source_task_id == task_id
                parsed_count += 1

    assert parsed_count == 2224


@pytest.mark.parametrize(
    ("preset", "active_axes"),
    [
        ("control", ()),
        ("regular", ("audio", "accents", "behavior")),
        ("control_audio", ("audio",)),
        ("control_accents", ("accents",)),
        ("control_behavior", ("behavior",)),
        ("control_audio_accents", ("audio", "accents")),
        ("control_audio_behavior", ("audio", "behavior")),
        ("control_accents_behavior", ("accents", "behavior")),
    ],
)
def test_speech_complexity_uses_upstream_active_axis_semantics(preset, active_axes):
    """Ablation names identify the axes activated by the upstream fixture."""
    assert SpeechComplexity(preset).active_axes == active_axes


def test_report_is_explicitly_declaration_only_and_non_comparable():
    """Parsed controls never masquerade as controls applied by the runtime."""
    domain, task_id, raw = _representative_profile("regular")
    profile = parse_tau_voice_profile(raw, domain=domain, task_id=task_id, expected_complexity="regular")

    report = profile.report()

    assert report["mode"] == "report_only"
    assert report["requested"]["complexity"] == "regular"
    assert report["requested"]["active_axes"] == ["audio", "accents", "behavior"]
    assert report["applied"] == {}
    assert report["benchmark_comparable"] is False
    assert {item["control"] for item in report["unsupported"]} == {
        "voice_binding",
        "audio_realization",
        "accents_realization",
        "behavior_realization",
    }


def test_parser_rejects_unknown_fields_and_asset_paths():
    """Schema drift and host-path injection fail before profiles reach artifacts."""
    domain, task_id, raw = _representative_profile()
    raw["unknown_control"] = True
    with pytest.raises(ValueError, match="unknown_control"):
        parse_tau_voice_profile(raw, domain=domain, task_id=task_id, expected_complexity="regular")

    _, _, raw = _representative_profile()
    raw["background_noise_file"] = "../private.wav"
    with pytest.raises(ValueError, match="logical asset name"):
        parse_tau_voice_profile(raw, domain=domain, task_id=task_id, expected_complexity="regular")


class _ProfileSelectionScenario(Tau2BaseScenario):
    """Minimal tau scenario used to exercise profile selection without bot setup."""

    domain = "tau2_airline"
    tau2_id = "profile-test"

    @property
    def _index_entry(self):
        task_id, task_config = next(iter(_load_voice_tasks(self.domain).items()))
        return {
            "task": {"id": task_id},
            "persona_name": task_config["configs"]["control"]["persona_name"],
            "voice_profiles": task_config["configs"],
        }


def test_tau_scenario_selects_profile_once_without_changing_legacy_persona_label():
    """Selection changes requested provenance while preserving the legacy persona field."""
    scenario = _ProfileSelectionScenario()
    control_persona = scenario.persona_name

    scenario.set_speech_complexity("control_audio")

    assert scenario.runtime_profile.complexity is SpeechComplexity.CONTROL_AUDIO
    assert scenario.persona_name == control_persona
    with pytest.raises(RuntimeError, match="cannot change"):
        scenario.set_speech_complexity("regular")

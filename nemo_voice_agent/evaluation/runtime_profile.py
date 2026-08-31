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

"""Typed declarations and provenance for evaluation runtime profiles.

This module deliberately models declarations only. Parsing a tau voice profile
does not imply that its audio, voice, or conversation controls were applied by
the evaluation runtime.
"""

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite
from typing import Any, Mapping


TAU2_VOICE_SOURCE_REVISION = "17e07b1da2bbc0cadfddeea36412686e0604127b"
RUNTIME_PROFILE_SCHEMA_VERSION = 1


class SpeechComplexity(StrEnum):
    """Speech-complexity presets shipped in the pinned tau voice fixtures."""

    CONTROL = "control"
    REGULAR = "regular"
    CONTROL_AUDIO = "control_audio"
    CONTROL_ACCENTS = "control_accents"
    CONTROL_BEHAVIOR = "control_behavior"
    CONTROL_AUDIO_ACCENTS = "control_audio_accents"
    CONTROL_AUDIO_BEHAVIOR = "control_audio_behavior"
    CONTROL_ACCENTS_BEHAVIOR = "control_accents_behavior"

    @property
    def active_axes(self) -> tuple[str, ...]:
        """Return the axes activated by this upstream ablation preset."""
        return {
            self.CONTROL: (),
            self.REGULAR: ("audio", "accents", "behavior"),
            self.CONTROL_AUDIO: ("audio",),
            self.CONTROL_ACCENTS: ("accents",),
            self.CONTROL_BEHAVIOR: ("behavior",),
            self.CONTROL_AUDIO_ACCENTS: ("audio", "accents"),
            self.CONTROL_AUDIO_BEHAVIOR: ("audio", "behavior"),
            self.CONTROL_ACCENTS_BEHAVIOR: ("accents", "behavior"),
        }[self]


@dataclass(frozen=True)
class ChannelEffects:
    """Network/channel degradation requested by a tau voice profile."""

    enable_frame_drops: bool
    frame_drop_burst_duration_ms: float
    frame_drop_count: int
    frame_drop_duration_ms: int
    frame_drop_rate: float


@dataclass(frozen=True)
class SourceEffects:
    """Noise-source controls requested by a tau voice profile."""

    burst_noise_events_per_minute: float
    burst_snr_range_db: tuple[float, float]
    enable_background_noise: bool
    enable_burst_noise: bool
    noise_snr_db: float
    noise_snr_drift_db: float
    noise_variation_speed: float


@dataclass(frozen=True)
class SpeechInsertion:
    """One benchmark-authored phrase that a future behavior driver may emit."""

    text: str
    type: str


@dataclass(frozen=True)
class SpeechEffects:
    """Speech-signal and content controls requested by a tau voice profile."""

    enable_dynamic_muffling: bool
    enable_non_directed_phrases: bool
    enable_vocal_tics: bool
    min_words_for_vocal_tics: int
    muffle_cutoff_freq: float
    muffle_probability: float
    muffle_segment_count: int
    muffle_segment_duration_ms: int
    muffle_transition_ms: int
    non_directed_phrases: tuple[SpeechInsertion, ...]
    speech_insert_events_per_minute: float
    vocal_tics: tuple[SpeechInsertion, ...]


@dataclass(frozen=True)
class PersonaControls:
    """Conversation-style controls requested by a tau voice profile."""

    interrupt_tendency: str
    verbosity: str


@dataclass(frozen=True)
class TauVoiceRuntimeProfile:
    """Immutable, lossless representation of one checked-in tau voice profile."""

    schema_version: int
    source_benchmark: str
    source_revision: str
    source_domain: str
    source_task_id: str
    complexity: SpeechComplexity
    backchannel_min_threshold: int | None
    background_noise_file: str | None
    burst_noise_files: tuple[str, ...]
    channel_effects_config: ChannelEffects
    enable_interruptions: bool
    environment: str | None
    persona_config: PersonaControls
    persona_name: str
    source_effects_config: SourceEffects
    speech_effects_config: SpeechEffects
    telephony_enabled: bool
    use_llm_backchannel: bool

    @property
    def active_axes(self) -> tuple[str, ...]:
        """Return the active axes declared by the selected upstream preset."""
        return self.complexity.active_axes

    def to_dict(self) -> dict[str, Any]:
        """Serialize the profile to stable JSON-compatible primitives."""
        payload = asdict(self)
        payload["complexity"] = self.complexity.value
        payload["active_axes"] = list(self.active_axes)
        return payload

    def report(self) -> dict[str, Any]:
        """Describe declaration-only support without claiming runtime application."""
        unsupported = [
            {
                "control": "voice_binding",
                "reason": "profile_parsed_but_not_applied",
            }
        ]
        unsupported.extend(
            {
                "control": f"{axis}_realization",
                "reason": "profile_parsed_but_not_applied",
            }
            for axis in self.active_axes
        )
        return {
            "schema_version": RUNTIME_PROFILE_SCHEMA_VERSION,
            "mode": "report_only",
            "requested": self.to_dict(),
            "applied": {},
            "unsupported": unsupported,
            "benchmark_comparable": False,
        }


_TOP_LEVEL_FIELDS = {
    "backchannel_min_threshold",
    "background_noise_file",
    "burst_noise_files",
    "channel_effects_config",
    "complexity",
    "enable_interruptions",
    "environment",
    "persona_config",
    "persona_name",
    "source_effects_config",
    "speech_effects_config",
    "telephony_enabled",
    "use_llm_backchannel",
}

_CHANNEL_FIELDS = {
    "enable_frame_drops",
    "frame_drop_burst_duration_ms",
    "frame_drop_count",
    "frame_drop_duration_ms",
    "frame_drop_rate",
}

_SOURCE_FIELDS = {
    "burst_noise_events_per_minute",
    "burst_snr_range_db",
    "enable_background_noise",
    "enable_burst_noise",
    "noise_snr_db",
    "noise_snr_drift_db",
    "noise_variation_speed",
}

_SPEECH_FIELDS = {
    "enable_dynamic_muffling",
    "enable_non_directed_phrases",
    "enable_vocal_tics",
    "min_words_for_vocal_tics",
    "muffle_cutoff_freq",
    "muffle_probability",
    "muffle_segment_count",
    "muffle_segment_duration_ms",
    "muffle_transition_ms",
    "non_directed_phrases",
    "speech_insert_events_per_minute",
    "vocal_tics",
}

_PERSONA_FIELDS = {"interrupt_tendency", "verbosity"}


def _expect_exact_fields(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"{path} schema mismatch: missing={missing}, unknown={unknown}")


def _expect_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be an object, got {type(value).__name__}")
    return value


def _expect_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a bool, got {type(value).__name__}")
    return value


def _expect_int(value: Any, path: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{path} must be non-negative, got {value}")
    return value


def _expect_number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be numeric, got {type(value).__name__}")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{path} must be finite, got {value}")
    if minimum is not None and result < minimum:
        raise ValueError(f"{path} must be >= {minimum}, got {result}")
    return result


def _expect_string(value: Any, path: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"{path} must be a non-empty string, got {value!r}")
    return value


def _expect_strings(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{path} must be a list of strings")
    return tuple(value)


def _expect_speech_insertions(value: Any, path: str, expected_type: str) -> tuple[SpeechInsertion, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{path} must be a list")
    result = []
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        item = _expect_mapping(item, item_path)
        _expect_exact_fields(item, {"text", "type"}, item_path)
        insertion_type = _expect_string(item["type"], f"{item_path}.type")
        if insertion_type != expected_type:
            raise ValueError(f"{item_path}.type must be {expected_type!r}, got {insertion_type!r}")
        result.append(
            SpeechInsertion(
                text=_expect_string(item["text"], f"{item_path}.text"),
                type=insertion_type,
            )
        )
    return tuple(result)


def _expect_asset_name(value: str | None, path: str) -> str | None:
    if value is None:
        return None
    if "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError(f"{path} must be a logical asset name, got {value!r}")
    return value


def parse_tau_voice_profile(
    raw: Mapping[str, Any],
    *,
    domain: str,
    task_id: str,
    expected_complexity: SpeechComplexity | str,
) -> TauVoiceRuntimeProfile:
    """Validate and parse one profile from the pinned tau voice fixtures."""
    raw = _expect_mapping(raw, "profile")
    _expect_exact_fields(raw, _TOP_LEVEL_FIELDS, "profile")

    expected = SpeechComplexity(expected_complexity)
    complexity = SpeechComplexity(_expect_string(raw["complexity"], "profile.complexity"))
    if complexity is not expected:
        raise ValueError(f"profile complexity {complexity.value!r} does not match key {expected.value!r}")

    channel = _expect_mapping(raw["channel_effects_config"], "profile.channel_effects_config")
    source = _expect_mapping(raw["source_effects_config"], "profile.source_effects_config")
    speech = _expect_mapping(raw["speech_effects_config"], "profile.speech_effects_config")
    persona = _expect_mapping(raw["persona_config"], "profile.persona_config")
    _expect_exact_fields(channel, _CHANNEL_FIELDS, "profile.channel_effects_config")
    _expect_exact_fields(source, _SOURCE_FIELDS, "profile.source_effects_config")
    _expect_exact_fields(speech, _SPEECH_FIELDS, "profile.speech_effects_config")
    _expect_exact_fields(persona, _PERSONA_FIELDS, "profile.persona_config")

    burst_snr = source["burst_snr_range_db"]
    if not isinstance(burst_snr, list) or len(burst_snr) != 2:
        raise TypeError("profile.source_effects_config.burst_snr_range_db must contain two numbers")
    burst_snr_range = (
        _expect_number(burst_snr[0], "profile.source_effects_config.burst_snr_range_db[0]"),
        _expect_number(burst_snr[1], "profile.source_effects_config.burst_snr_range_db[1]"),
    )
    if burst_snr_range[0] > burst_snr_range[1]:
        raise ValueError("profile.source_effects_config.burst_snr_range_db must be ordered")

    muffle_probability = _expect_number(
        speech["muffle_probability"], "profile.speech_effects_config.muffle_probability", minimum=0.0
    )
    if muffle_probability > 1.0:
        raise ValueError("profile.speech_effects_config.muffle_probability must be <= 1.0")

    background_noise_file = _expect_asset_name(
        _expect_string(raw["background_noise_file"], "profile.background_noise_file", allow_none=True),
        "profile.background_noise_file",
    )
    burst_noise_files = _expect_strings(raw["burst_noise_files"], "profile.burst_noise_files")
    for asset in burst_noise_files:
        _expect_asset_name(asset, "profile.burst_noise_files")

    return TauVoiceRuntimeProfile(
        schema_version=RUNTIME_PROFILE_SCHEMA_VERSION,
        source_benchmark="tau2-bench",
        source_revision=TAU2_VOICE_SOURCE_REVISION,
        source_domain=domain,
        source_task_id=task_id,
        complexity=complexity,
        backchannel_min_threshold=_expect_int(
            raw["backchannel_min_threshold"], "profile.backchannel_min_threshold", allow_none=True
        ),
        background_noise_file=background_noise_file,
        burst_noise_files=burst_noise_files,
        channel_effects_config=ChannelEffects(
            enable_frame_drops=_expect_bool(
                channel["enable_frame_drops"], "profile.channel_effects_config.enable_frame_drops"
            ),
            frame_drop_burst_duration_ms=_expect_number(
                channel["frame_drop_burst_duration_ms"],
                "profile.channel_effects_config.frame_drop_burst_duration_ms",
                minimum=0.0,
            ),
            frame_drop_count=_expect_int(
                channel["frame_drop_count"], "profile.channel_effects_config.frame_drop_count"
            ),
            frame_drop_duration_ms=_expect_int(
                channel["frame_drop_duration_ms"], "profile.channel_effects_config.frame_drop_duration_ms"
            ),
            frame_drop_rate=_expect_number(
                channel["frame_drop_rate"], "profile.channel_effects_config.frame_drop_rate", minimum=0.0
            ),
        ),
        enable_interruptions=_expect_bool(raw["enable_interruptions"], "profile.enable_interruptions"),
        environment=_expect_string(raw["environment"], "profile.environment", allow_none=True),
        persona_config=PersonaControls(
            interrupt_tendency=_expect_string(
                persona["interrupt_tendency"], "profile.persona_config.interrupt_tendency"
            ),
            verbosity=_expect_string(persona["verbosity"], "profile.persona_config.verbosity"),
        ),
        persona_name=_expect_string(raw["persona_name"], "profile.persona_name"),
        source_effects_config=SourceEffects(
            burst_noise_events_per_minute=_expect_number(
                source["burst_noise_events_per_minute"],
                "profile.source_effects_config.burst_noise_events_per_minute",
                minimum=0.0,
            ),
            burst_snr_range_db=burst_snr_range,
            enable_background_noise=_expect_bool(
                source["enable_background_noise"], "profile.source_effects_config.enable_background_noise"
            ),
            enable_burst_noise=_expect_bool(
                source["enable_burst_noise"], "profile.source_effects_config.enable_burst_noise"
            ),
            noise_snr_db=_expect_number(source["noise_snr_db"], "profile.source_effects_config.noise_snr_db"),
            noise_snr_drift_db=_expect_number(
                source["noise_snr_drift_db"], "profile.source_effects_config.noise_snr_drift_db", minimum=0.0
            ),
            noise_variation_speed=_expect_number(
                source["noise_variation_speed"], "profile.source_effects_config.noise_variation_speed", minimum=0.0
            ),
        ),
        speech_effects_config=SpeechEffects(
            enable_dynamic_muffling=_expect_bool(
                speech["enable_dynamic_muffling"], "profile.speech_effects_config.enable_dynamic_muffling"
            ),
            enable_non_directed_phrases=_expect_bool(
                speech["enable_non_directed_phrases"],
                "profile.speech_effects_config.enable_non_directed_phrases",
            ),
            enable_vocal_tics=_expect_bool(
                speech["enable_vocal_tics"], "profile.speech_effects_config.enable_vocal_tics"
            ),
            min_words_for_vocal_tics=_expect_int(
                speech["min_words_for_vocal_tics"], "profile.speech_effects_config.min_words_for_vocal_tics"
            ),
            muffle_cutoff_freq=_expect_number(
                speech["muffle_cutoff_freq"], "profile.speech_effects_config.muffle_cutoff_freq", minimum=0.0
            ),
            muffle_probability=muffle_probability,
            muffle_segment_count=_expect_int(
                speech["muffle_segment_count"], "profile.speech_effects_config.muffle_segment_count"
            ),
            muffle_segment_duration_ms=_expect_int(
                speech["muffle_segment_duration_ms"],
                "profile.speech_effects_config.muffle_segment_duration_ms",
            ),
            muffle_transition_ms=_expect_int(
                speech["muffle_transition_ms"], "profile.speech_effects_config.muffle_transition_ms"
            ),
            non_directed_phrases=_expect_speech_insertions(
                speech["non_directed_phrases"],
                "profile.speech_effects_config.non_directed_phrases",
                "non_directed_phrase",
            ),
            speech_insert_events_per_minute=_expect_number(
                speech["speech_insert_events_per_minute"],
                "profile.speech_effects_config.speech_insert_events_per_minute",
                minimum=0.0,
            ),
            vocal_tics=_expect_speech_insertions(
                speech["vocal_tics"], "profile.speech_effects_config.vocal_tics", "vocal_tic"
            ),
        ),
        telephony_enabled=_expect_bool(raw["telephony_enabled"], "profile.telephony_enabled"),
        use_llm_backchannel=_expect_bool(raw["use_llm_backchannel"], "profile.use_llm_backchannel"),
    )

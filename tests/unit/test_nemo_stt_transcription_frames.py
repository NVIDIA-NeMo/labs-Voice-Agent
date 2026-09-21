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

"""Pins the pipecat frame contract that ``NemoSTTService`` relies on.

``NemoSTTService.run_stt`` attaches ``finalized`` only when the frame class is
``TranscriptionFrame``::

    extra = {"finalized": is_final} if frame_type is TranscriptionFrame else {}

That conditional is load-bearing, not stylistic: ``InterimTranscriptionFrame``
has no ``finalized`` field, so passing the kwarg unconditionally raises
``TypeError`` at runtime — on the audio path, once per buffer.

These tests fail loudly if a pipecat upgrade moves the field in either
direction, which is the only way the guard can become wrong.
"""

import contextlib
import inspect
import io
import wave

import pytest
from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame

from nemo_voice_agent.pipecat.services.nemo import stt as nemo_stt


def test_transcription_frame_accepts_finalized():
    """``TranscriptionFrame`` carries the field and round-trips it."""
    frame = TranscriptionFrame("hello", "user", "2026-01-01T00:00:00Z", finalized=True)
    assert frame.finalized is True


def test_transcription_frame_defaults_finalized_false():
    """Omitting it must mean "not finalized", never "unknown"."""
    frame = TranscriptionFrame("hello", "user", "2026-01-01T00:00:00Z")
    assert frame.finalized is False


def test_interim_frame_rejects_finalized():
    """The reason ``run_stt`` guards on frame class.

    If this ever starts passing, pipecat has added the field to the interim
    frame and the conditional in ``run_stt`` can be simplified away.
    """
    with pytest.raises(TypeError):
        InterimTranscriptionFrame(
            "hello",
            "user",
            "2026-01-01T00:00:00Z",
            finalized=True,  # type: ignore[call-arg]
        )


def test_run_stt_guards_finalized_on_frame_class():
    """The guard is present in the source and keyed on ``TranscriptionFrame``.

    A source check rather than a behavioural one: driving ``run_stt`` requires a
    loaded NeMo model plus pipecat metrics plumbing. This at least fails if the
    conditional is deleted, which is the regression that matters.
    """
    source = inspect.getsource(nemo_stt.NemoSTTService.run_stt)
    assert '{"finalized": is_final} if frame_type is TranscriptionFrame else {}' in source
    assert "**extra," in source


# ---------------------------------------------------------------------------
# NemoSpeechLMSTTService.max_tokens_per_sec
# ---------------------------------------------------------------------------


def _wav(seconds: float, rate: int = 16000) -> bytes:
    """A silent 16-bit mono WAV of the given duration, as pipecat builds segments."""
    buf = io.BytesIO()
    with contextlib.closing(wave.open(buf, "wb")) as w:
        w.setsampwidth(2)
        w.setnchannels(1)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


def _svc(**kw):
    """Construct the service without touching the network."""
    return nemo_stt.NemoSpeechLMSTTService(model="m", **kw)


def test_duration_parsed_from_wav_segment():
    assert nemo_stt.NemoSpeechLMSTTService._audio_duration_seconds(_wav(2.0)) == pytest.approx(2.0)
    assert nemo_stt.NemoSpeechLMSTTService._audio_duration_seconds(_wav(0.25)) == pytest.approx(0.25)


def test_unset_rate_leaves_generation_kwargs_untouched():
    """Default behaviour must not change: same dict object, no copy, no cap."""
    svc = _svc(generation_kwargs={"max_tokens": 2048})
    assert svc._generation_kwargs_for(_wav(1.0)) is svc._generation_kwargs


def test_cap_applies_when_shorter_than_configured():
    """10s at 20 tok/s = 200, below the configured 2048, so the cap bites."""
    svc = _svc(generation_kwargs={"max_tokens": 2048}, max_tokens_per_sec=20)
    assert svc._generation_kwargs_for(_wav(10.0))["max_tokens"] == 200


def test_configured_ceiling_still_wins_when_lower():
    """The parameter is a ceiling, never a floor — it must not raise max_tokens."""
    svc = _svc(generation_kwargs={"max_tokens": 64}, max_tokens_per_sec=1000)
    assert svc._generation_kwargs_for(_wav(10.0)) is svc._generation_kwargs


def test_short_audio_keeps_a_usable_budget():
    """A sub-second "yes" must not be clipped to a handful of tokens."""
    svc = _svc(generation_kwargs={"max_tokens": 2048}, max_tokens_per_sec=20)
    assert svc._generation_kwargs_for(_wav(0.1))["max_tokens"] == nemo_stt.NemoSpeechLMSTTService.MIN_TOKEN_BUDGET


def test_cap_applies_with_no_configured_max_tokens():
    svc = _svc(max_tokens_per_sec=20)
    assert svc._generation_kwargs_for(_wav(10.0))["max_tokens"] == 200


def test_unparseable_audio_falls_back_to_flat_budget():
    """Raw PCM or a truncated segment must not produce a guessed cap."""
    svc = _svc(generation_kwargs={"max_tokens": 2048}, max_tokens_per_sec=20)
    assert svc._generation_kwargs_for(b"not a wav") is svc._generation_kwargs


def test_shared_generation_kwargs_never_mutated():
    """The dict is shared across concurrent requests; capping must copy."""
    svc = _svc(generation_kwargs={"max_tokens": 2048}, max_tokens_per_sec=20)
    svc._generation_kwargs_for(_wav(1.0))
    assert svc._generation_kwargs["max_tokens"] == 2048


def test_would_have_bounded_the_observed_hallucination():
    """Regression anchor for the failure this parameter exists to stop.

    A ~16s turn decoded into 9,370 words (~12k tokens) in the SALM airline run.
    At 20 tokens/s that request would have been cut to 320.
    """
    svc = _svc(generation_kwargs={"max_tokens": 10000}, max_tokens_per_sec=20)
    assert svc._generation_kwargs_for(_wav(16.0))["max_tokens"] == 320


# ---------------------------------------------------------------------------
# system_prompt / user_prompt path-or-literal resolution
# ---------------------------------------------------------------------------


def test_prompts_read_from_file(tmp_path):
    """A value naming an existing file is read; both prompts honour it."""
    f = tmp_path / "asr_prompt.txt"
    f.write_text("Transcribe verbatim.\nPreserve identifiers.\n")
    svc = _svc(system_prompt=str(f), user_prompt=str(f))
    assert svc._system_prompt == "Transcribe verbatim.\nPreserve identifiers.\n"
    assert svc._user_prompt == "Transcribe verbatim.\nPreserve identifiers.\n"


def test_literal_prompts_pass_through(tmp_path):
    """Anything that is not an existing file is taken literally."""
    svc = _svc(system_prompt="You are a helpful assistant. /no_think", user_prompt="Transcribe this.")
    assert svc._system_prompt == "You are a helpful assistant. /no_think"
    assert svc._user_prompt == "Transcribe this."


def test_multiline_literal_is_never_mistaken_for_a_path():
    """The rule is unambiguous for real prompts: they are not valid file paths."""
    prompt = "Line one.\nLine two.\nLine three."
    assert _svc(user_prompt=prompt)._user_prompt == prompt


def test_reset_user_prompt_restores_file_contents_not_the_path(tmp_path):
    """Resolution happens once, in the constructor.

    ``_original_user_prompt`` must hold the resolved text, otherwise a runtime
    ``set_user_prompt`` followed by ``reset_user_prompt`` would restore the
    literal path string and send it to the model as the transcription prompt.
    """
    f = tmp_path / "asr_prompt.txt"
    f.write_text("Transcribe verbatim.")
    svc = _svc(user_prompt=str(f))
    svc.set_user_prompt("override")
    svc.reset_user_prompt()
    assert svc._user_prompt == "Transcribe verbatim."


def test_default_user_prompt_still_applies_when_unset():
    assert _svc()._user_prompt == nemo_stt.NemoSpeechLMSTTService.DEFAULT_USER_PROMPT


def test_resolve_prompt_handles_none_and_empty():
    from nemo_voice_agent.utils.misc import resolve_prompt

    assert resolve_prompt(None) is None
    assert resolve_prompt("") == ""

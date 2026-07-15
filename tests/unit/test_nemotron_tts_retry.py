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

"""Tests for ResilientNemotronTTSService's retry-on-transient-gRPC-error behavior.

Mirrors ``test_resilient_nvidia_stt.py``: we don't spin up real Riva/NVCF. We
build a bare service via ``__new__`` (bypassing the connecting ``__init__``),
set a stub synthesis backend + metric hooks, install the resilient wrapper, then
drive ``run_tts`` to assert the bounded retry loop re-establishes a chunk's
synthesis RPC on transient errors and gives up cleanly otherwise.

The retryable error mirrors a real NVCF failure::

    status = StatusCode.DEADLINE_EXCEEDED
    details = "reason:\\"failed to establish link to worker\\" ..."
"""

import asyncio
from types import SimpleNamespace

from pipecat.frames.frames import TTSAudioRawFrame

from nemo_voice_agent.pipecat.services.riva_speech import ResilientNemotronTTSService


def _fake_grpc_error(message: str, code_name: str) -> RuntimeError:
    """Duck-typed gRPC-error stand-in exposing ``.code().name`` (see the STT
    test for rationale) — no hard ``grpc`` import needed."""
    err = RuntimeError(message)
    err.code = lambda: SimpleNamespace(name=code_name)  # type: ignore[attr-defined]
    return err


def _resp(audio: bytes) -> SimpleNamespace:
    """A minimal synthesize_online response with an ``.audio`` payload."""
    return SimpleNamespace(audio=audio)


def _make_tts(synth) -> ResilientNemotronTTSService:
    """Bare ResilientNemotronTTSService with ``synth`` as its synthesis backend
    and the resilient wrapper installed over it."""
    tts = ResilientNemotronTTSService.__new__(ResilientNemotronTTSService)
    tts._name = "TestResilientNemotronTTSService"
    tts._id = 0
    # Attributes read by run_tts' synthesis loop.
    tts._voice_id = "v"
    tts._language_code = "en-US"
    tts._sample_rate = 22050
    tts._tts_timeout = None
    tts._zero_shot_audio_prompt_file = None
    tts._audio_prompt_encoding = None
    tts._zero_shot_quality = 20
    tts._custom_dictionary = None
    tts._encoding = None
    tts._display_text_before_filter = None
    tts.TTS_RETRY_BACKOFF_S = 0  # no real sleeping in tests
    # Stub text pre-processing + metric hooks (require a full pipeline otherwise).
    tts._is_metadata_only = lambda _t: False
    tts._strip_non_speech_content = lambda t: t

    async def _noop(*_a, **_k):
        return None

    tts.start_ttfb_metrics = _noop
    tts.stop_ttfb_metrics = _noop
    tts.start_tts_usage_metrics = _noop
    # Install the resilient wrapper over the stub backend (mirrors __init__).
    tts._service = SimpleNamespace(synthesize_online=synth)
    tts._install_resilient_synthesis()
    return tts


def _drive(tts: ResilientNemotronTTSService, text: str = "hello world") -> list:
    """Run run_tts to completion and collect the emitted frames."""

    async def _collect():
        return [f async for f in tts.run_tts(text)]

    return asyncio.run(_collect())


def _real_audio(frames) -> list:
    """Audio frames excluding the base's trailing all-zero silence padding."""
    return [f for f in frames if isinstance(f, TTSAudioRawFrame) and any(f.audio)]


# --------------------------------------------------------------------------- #
# _is_retryable_tts_error classification
# --------------------------------------------------------------------------- #


def test_is_retryable_classifies_transient_codes():
    tts = _make_tts(lambda *a, **k: iter(()))
    for code in ("DEADLINE_EXCEEDED", "UNAVAILABLE", "INTERNAL"):
        assert tts._is_retryable_tts_error(_fake_grpc_error("x", code)) is True


def test_is_retryable_rejects_non_transient_and_plain_errors():
    tts = _make_tts(lambda *a, **k: iter(()))
    assert tts._is_retryable_tts_error(_fake_grpc_error("bad key", "UNAUTHENTICATED")) is False
    assert tts._is_retryable_tts_error(_fake_grpc_error("nope", "INVALID_ARGUMENT")) is False
    assert tts._is_retryable_tts_error(ValueError("boom")) is False


# --------------------------------------------------------------------------- #
# run_tts retry loop (via the wrapped synthesize_online)
# --------------------------------------------------------------------------- #


def test_recovers_from_single_transient_error():
    """First synth attempt fails before any audio (DEADLINE_EXCEEDED "failed to
    establish link to worker"); the retry succeeds and audio flows."""
    calls = {"n": 0}

    def synth(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:

            def gen():
                raise _fake_grpc_error('reason:"failed to establish link to worker"', "DEADLINE_EXCEEDED")
                yield  # pragma: no cover - makes this a generator

            return gen()
        return iter([_resp(b"\x01\x02"), _resp(b"\x03\x04")])

    tts = _make_tts(synth)
    frames = _drive(tts)

    assert calls["n"] == 2, "expected one retry after the transient error"
    # No duplication: exactly the two synthesized chunks, in order.
    assert [f.audio for f in _real_audio(frames)] == [b"\x01\x02", b"\x03\x04"]


def test_gives_up_after_max_retries():
    """Persistent transient error: re-establishes MAX_TTS_RETRIES times then
    propagates (base logs + drops the utterance — no real audio emitted)."""
    calls = {"n": 0}

    def synth(*_a, **_k):
        calls["n"] += 1

        def gen():
            raise _fake_grpc_error("still down", "UNAVAILABLE")
            yield  # pragma: no cover

        return gen()

    tts = _make_tts(synth)
    frames = _drive(tts)

    assert calls["n"] == ResilientNemotronTTSService.MAX_TTS_RETRIES + 1
    assert _real_audio(frames) == []


def test_non_transient_error_not_retried():
    """A non-retryable status (auth) fails fast — exactly one attempt."""
    calls = {"n": 0}

    def synth(*_a, **_k):
        calls["n"] += 1

        def gen():
            raise _fake_grpc_error("bad key", "UNAUTHENTICATED")
            yield  # pragma: no cover

        return gen()

    tts = _make_tts(synth)
    frames = _drive(tts)

    assert calls["n"] == 1
    assert _real_audio(frames) == []


def test_no_retry_after_audio_already_emitted():
    """If the stream fails AFTER emitting audio, we must not re-establish the RPC
    (would duplicate the opening audio) — one attempt, partial audio kept."""
    calls = {"n": 0}

    def synth(*_a, **_k):
        calls["n"] += 1

        def gen():
            yield _resp(b"\xaa\xbb")  # first frame emitted...
            raise _fake_grpc_error("mid-stream reset", "INTERNAL")  # ...then transient error

        return gen()

    tts = _make_tts(synth)
    frames = _drive(tts)

    assert calls["n"] == 1, "must not retry once audio has been emitted"
    assert [f.audio for f in _real_audio(frames)] == [b"\xaa\xbb"]


def test_clean_synthesis_no_retry():
    """Happy path: one successful attempt, audio flows unchanged."""
    calls = {"n": 0}

    def synth(*_a, **_k):
        calls["n"] += 1
        return iter([_resp(b"\x01"), _resp(b"\x02")])

    tts = _make_tts(synth)
    frames = _drive(tts)

    assert calls["n"] == 1
    assert [f.audio for f in _real_audio(frames)] == [b"\x01", b"\x02"]


def test_wrapper_not_stacked_on_reinstall():
    """Re-installing the wrapper doesn't stack — original is preserved once."""
    real = lambda *a, **k: iter([_resp(b"\x09")])  # noqa: E731
    tts = _make_tts(real)
    first_wrapped = tts._service.synthesize_online
    tts._install_resilient_synthesis()
    # The stored original stays the real backend, not the first wrapper.
    assert tts._service._synthesize_online_original is real
    assert tts._service.synthesize_online is not first_wrapped

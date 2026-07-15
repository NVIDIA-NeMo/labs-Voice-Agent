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

"""Tests for ResilientNvidiaSTTService's retry-on-sequence-state-error behavior.

We don't spin up real Riva/NVCF for unit tests — we patch the inherited
``_response_handler`` to raise (or succeed) per scenario, then assert that
``_thread_task_handler`` retries on the specific Triton sequence-state error
message and propagates everything else.

The error message we match comes from a real NVCF response captured in
bot_agent_nemotron.log on 2026-06-02::

    status = StatusCode.INVALID_ARGUMENT
    details = "inference request for sequence 155182184 to model
               'parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer' must
               specify the START flag on the first request of the sequence"
"""

import asyncio
from types import SimpleNamespace

import pytest

from nemo_voice_agent.pipecat.services.nemo.stt import ResilientNvidiaSTTService


# Real-world error message we extracted from the live bot log (2026-06-01).
_TRITON_SEQUENCE_ERROR = (
    "inference request for sequence 155182184 to model "
    "'parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer' must specify "
    "the START flag on the first request of the sequence"
)


def _fake_grpc_error(message: str, code_name: str) -> RuntimeError:
    """Build a duck-typed gRPC-error stand-in.

    Real ``grpc.RpcError`` subclasses expose ``.code()`` returning a
    ``grpc.StatusCode`` enum value whose ``.name`` is e.g. ``"INTERNAL"``.
    We don't import grpc in tests; we synthesize a matching shape with a
    plain ``RuntimeError`` plus a ``code()`` method that returns a
    ``SimpleNamespace(name=...)``. The resilient service uses duck-typed
    lookup (``exc.code().name``) so this matches the production code path.
    """
    err = RuntimeError(message)
    err.code = lambda: SimpleNamespace(name=code_name)  # type: ignore[attr-defined]
    return err


def _bare_resilient_stt():
    """Construct a ``ResilientNvidiaSTTService`` without invoking ``__init__``
    so we don't need a live NVCF endpoint / valid API key. ``_thread_task_handler``
    only touches ``self._response_handler`` (which we monkey-patch per test),
    ``self._thread_running`` (set inside the method), and ``self._name`` /
    ``self._id`` (used by pipecat's ``BaseObject.__str__`` for log messages).
    Set the latter two manually since we bypass ``__init__``.
    """
    instance = ResilientNvidiaSTTService.__new__(ResilientNvidiaSTTService)
    instance._thread_running = False
    instance._name = "TestResilientNvidiaSTTService"
    instance._id = 0
    return instance


def test_clean_return_no_retry():
    """Normal happy path: _response_handler returns without raising → no retries."""
    stt = _bare_resilient_stt()
    call_count = {"n": 0}

    def fake_handler():
        call_count["n"] += 1
        # success: just return

    stt._response_handler = fake_handler
    asyncio.run(stt._thread_task_handler())
    assert call_count["n"] == 1, "expected exactly one _response_handler call on success"


def test_recovers_from_single_sequence_error():
    """First call raises the Triton sequence error; second call succeeds.
    Total: 2 calls, no exception escapes _thread_task_handler.
    """
    stt = _bare_resilient_stt()
    call_count = {"n": 0}

    def fake_handler():
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError(_TRITON_SEQUENCE_ERROR)
        # second call succeeds

    stt._response_handler = fake_handler
    asyncio.run(stt._thread_task_handler())
    assert call_count["n"] == 2


def test_gives_up_after_max_retries():
    """Persistent sequence error: retries up to MAX_RETRIES then re-raises."""
    stt = _bare_resilient_stt()
    call_count = {"n": 0}

    def fake_handler():
        call_count["n"] += 1
        raise RuntimeError(_TRITON_SEQUENCE_ERROR)

    stt._response_handler = fake_handler
    with pytest.raises(RuntimeError, match="must specify the START flag"):
        asyncio.run(stt._thread_task_handler())
    # First attempt + MAX_RETRIES retries = MAX_RETRIES + 1 total calls.
    assert call_count["n"] == ResilientNvidiaSTTService.MAX_RETRIES + 1


def test_retries_on_grpc_internal_rst_stream():
    """``RST_STREAM with error code 2`` arrives as ``StatusCode.INTERNAL``.

    Observed live on 2026-06-02 as a fresh failure mode after the sequence-state
    retry shipped. The resilient service must treat this as transient too and
    open a new streaming session under the resilient retry policy.
    """
    stt = _bare_resilient_stt()
    call_count = {"n": 0}

    def fake_handler():
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _fake_grpc_error("Received RST_STREAM with error code 2", "INTERNAL")
        # second call succeeds

    stt._response_handler = fake_handler
    asyncio.run(stt._thread_task_handler())
    assert call_count["n"] == 2


def test_retries_on_grpc_unavailable():
    """``StatusCode.UNAVAILABLE`` (cloud endpoint flapping) is retryable."""
    stt = _bare_resilient_stt()
    call_count = {"n": 0}

    def fake_handler():
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _fake_grpc_error("connection refused", "UNAVAILABLE")

    stt._response_handler = fake_handler
    asyncio.run(stt._thread_task_handler())
    assert call_count["n"] == 2


def test_retries_on_grpc_deadline_exceeded():
    """``StatusCode.DEADLINE_EXCEEDED`` (long-stream timeout) is retryable."""
    stt = _bare_resilient_stt()
    call_count = {"n": 0}

    def fake_handler():
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _fake_grpc_error("deadline exceeded", "DEADLINE_EXCEEDED")

    stt._response_handler = fake_handler
    asyncio.run(stt._thread_task_handler())
    assert call_count["n"] == 2


def test_does_not_retry_grpc_unauthenticated():
    """Auth errors are config bugs, not transient; must propagate immediately."""
    stt = _bare_resilient_stt()
    call_count = {"n": 0}

    def fake_handler():
        call_count["n"] += 1
        raise _fake_grpc_error("invalid bearer token", "UNAUTHENTICATED")

    stt._response_handler = fake_handler
    with pytest.raises(RuntimeError, match="invalid bearer token"):
        asyncio.run(stt._thread_task_handler())
    assert call_count["n"] == 1


def test_does_not_retry_grpc_not_found():
    """``StatusCode.NOT_FOUND`` (e.g., wrong model name) is a config bug."""
    stt = _bare_resilient_stt()
    call_count = {"n": 0}

    def fake_handler():
        call_count["n"] += 1
        raise _fake_grpc_error("model 'wrong-name' not found", "NOT_FOUND")

    stt._response_handler = fake_handler
    with pytest.raises(RuntimeError, match="not found"):
        asyncio.run(stt._thread_task_handler())
    assert call_count["n"] == 1


def test_does_not_retry_invalid_argument_when_not_sequence_state():
    """``INVALID_ARGUMENT`` is retried only for the specific sequence-state
    message. Other ``INVALID_ARGUMENT`` errors (malformed audio, bad config)
    must NOT retry — they're client-side bugs that won't fix themselves."""
    stt = _bare_resilient_stt()
    call_count = {"n": 0}

    def fake_handler():
        call_count["n"] += 1
        raise _fake_grpc_error("sample rate must be 16000Hz", "INVALID_ARGUMENT")

    stt._response_handler = fake_handler
    with pytest.raises(RuntimeError, match="sample rate"):
        asyncio.run(stt._thread_task_handler())
    assert call_count["n"] == 1


def test_non_sequence_error_propagates_immediately():
    """Random non-sequence error (e.g., auth, network) is NOT retried — bubbles up
    on the first attempt so failures fail fast rather than burn the retry budget."""
    stt = _bare_resilient_stt()
    call_count = {"n": 0}

    def fake_handler():
        call_count["n"] += 1
        raise RuntimeError("UNAUTHENTICATED: invalid API key")

    stt._response_handler = fake_handler
    with pytest.raises(RuntimeError, match="UNAUTHENTICATED"):
        asyncio.run(stt._thread_task_handler())
    assert call_count["n"] == 1, "expected immediate propagation, no retry"


def test_cancellation_does_not_retry():
    """asyncio.CancelledError must propagate cleanly — it's not a retryable error."""
    stt = _bare_resilient_stt()
    call_count = {"n": 0}

    def fake_handler():
        call_count["n"] += 1
        raise asyncio.CancelledError()

    stt._response_handler = fake_handler
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(stt._thread_task_handler())
    assert call_count["n"] == 1
    assert stt._thread_running is False, (
        "CancelledError branch must flip _thread_running to False so __next__ "
        "raises StopIteration and the audio queue iterator unblocks"
    )


def test_thread_running_stays_true_across_retries():
    """Across retries, _thread_running must stay True so __next__ keeps feeding
    audio chunks from the queue (instead of raising StopIteration)."""
    stt = _bare_resilient_stt()
    seen_thread_running = []

    def fake_handler():
        # Snapshot _thread_running each call
        seen_thread_running.append(stt._thread_running)
        if len(seen_thread_running) < 3:
            raise RuntimeError(_TRITON_SEQUENCE_ERROR)
        # 3rd call succeeds

    stt._response_handler = fake_handler
    asyncio.run(stt._thread_task_handler())
    assert seen_thread_running == [
        True,
        True,
        True,
    ], "expected _thread_running=True for all 3 attempts (retries shouldn't tear it down)"

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

"""Retry semantics for ResilientNvidiaTTSService._synthesis_handler.

The handler runs in a worker thread and talks to the event loop through
``run_coroutine_threadsafe``, so these tests drive it directly with a real
loop running in the background and a stubbed gRPC stub.
"""

import asyncio
import queue
import threading
import types

import grpc
import pytest

from nemo_voice_agent.pipecat.services.nvidia.tts import (
    TRANSIENT_STATUS_CODES,
    ResilientNvidiaTTSService,
    _is_transient,
)


class _FakeRpcError(grpc.RpcError):
    def __init__(self, code):
        self._code = code

    def code(self):
        return self._code

    def details(self):
        return "failed to establish link to worker"

    def __str__(self):
        return f"{self._code}: {self.details()}"


class _State:
    """Stand-in for pipecat's _SynthesisStreamState."""

    def __init__(self, texts):
        self.context_id = "ctx"
        self.text_queue = queue.Queue()
        self.response_queue = asyncio.Queue()
        self.stop_event = threading.Event()
        self.rpc_call = None
        for t in texts:
            self.text_queue.put(t)
        self.text_queue.put(None)


def _make_service(attempts_behavior, loop, max_retries=2):
    """Build a service whose SynthesizeOnline follows ``attempts_behavior``.

    Each entry is either an exception to raise on iteration, or a list of
    response objects to yield.
    """
    svc = ResilientNvidiaTTSService.__new__(ResilientNvidiaTTSService)
    svc._max_retries = max_retries
    svc._retry_backoff_secs = 0.0
    svc.get_event_loop = lambda: loop
    svc._build_base_request = lambda: types.SimpleNamespace(text="")
    svc.__class__.__str__ = lambda self: "FakeTTS"

    calls = {"n": 0, "sent": []}

    def synthesize_online(request_iter, metadata=None):
        idx = calls["n"]
        calls["n"] += 1
        sent = [r.text for r in request_iter]
        calls["sent"].append(sent)
        behavior = attempts_behavior[min(idx, len(attempts_behavior) - 1)]

        def gen():
            if isinstance(behavior, BaseException):
                raise behavior
            for item in behavior:
                yield item

        return gen()

    svc._service = types.SimpleNamespace(
        stub=types.SimpleNamespace(SynthesizeOnline=synthesize_online),
        auth=types.SimpleNamespace(get_auth_metadata=lambda: []),
    )
    return svc, calls


def _drain(loop, state):
    async def _collect():
        out = []
        while True:
            item = await asyncio.wait_for(state.response_queue.get(), timeout=5)
            if item is None:
                return out
            out.append(item)

    return asyncio.run_coroutine_threadsafe(_collect(), loop).result(timeout=10)


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    t = threading.Thread(target=lp.run_forever, daemon=True)
    t.start()
    yield lp
    lp.call_soon_threadsafe(lp.stop)
    t.join(timeout=5)


@pytest.mark.unit
def test_transient_failure_before_audio_is_retried_and_replays_text(loop):
    """A DEADLINE_EXCEEDED with no audio yet retries and re-sends the same text."""
    err = _FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED)
    svc, calls = _make_service([err, ["audio-1", "audio-2"]], loop)
    state = _State(["hello ", "world"])

    svc._synthesis_handler(state)
    out = _drain(loop, state)

    assert out == ["audio-1", "audio-2"], "second attempt's audio must reach the response queue"
    assert calls["n"] == 2, "expected exactly one retry"
    # The failed attempt consumed the text queue destructively; without replay
    # the retry would synthesize nothing.
    assert calls["sent"][1] == ["hello ", "world"], "retry must replay the consumed text"


@pytest.mark.unit
def test_failure_after_audio_is_not_retried(loop):
    """Once audio is emitted, retrying would splice a duplicate prefix — so don't."""

    def failing_after_audio():
        yield "audio-1"
        raise _FakeRpcError(grpc.StatusCode.UNAVAILABLE)

    svc = ResilientNvidiaTTSService.__new__(ResilientNvidiaTTSService)
    svc._max_retries = 3
    svc._retry_backoff_secs = 0.0
    svc.get_event_loop = lambda: loop
    svc._build_base_request = lambda: types.SimpleNamespace(text="")
    svc.__class__.__str__ = lambda self: "FakeTTS"
    calls = {"n": 0}

    def synthesize_online(request_iter, metadata=None):
        calls["n"] += 1
        list(request_iter)
        return failing_after_audio()

    svc._service = types.SimpleNamespace(
        stub=types.SimpleNamespace(SynthesizeOnline=synthesize_online),
        auth=types.SimpleNamespace(get_auth_metadata=lambda: []),
    )
    state = _State(["hi"])

    svc._synthesis_handler(state)
    out = _drain(loop, state)

    assert calls["n"] == 1, "must not retry once audio has been emitted"
    assert out[0] == "audio-1"
    assert isinstance(out[-1], grpc.RpcError), "the error must still be reported downstream"


@pytest.mark.unit
def test_non_transient_error_is_not_retried(loop):
    """INTERNAL is a client-side bug signature; retrying only adds latency."""
    err = _FakeRpcError(grpc.StatusCode.INTERNAL)
    svc, calls = _make_service([err, ["never"]], loop)
    state = _State(["hi"])

    svc._synthesis_handler(state)
    out = _drain(loop, state)

    assert calls["n"] == 1
    assert isinstance(out[-1], grpc.RpcError)


@pytest.mark.unit
def test_retries_are_bounded(loop):
    """After max_retries the error is surfaced rather than looping forever."""
    err = _FakeRpcError(grpc.StatusCode.UNAVAILABLE)
    svc, calls = _make_service([err], loop, max_retries=2)
    state = _State(["hi"])

    svc._synthesis_handler(state)
    out = _drain(loop, state)

    assert calls["n"] == 3, "1 initial attempt + 2 retries"
    assert isinstance(out[-1], grpc.RpcError)


@pytest.mark.unit
def test_interruption_aborts_without_retry_or_error(loop):
    """A barge-in cancels the call; that is teardown, not a failure to report."""
    err = _FakeRpcError(grpc.StatusCode.CANCELLED)
    svc, calls = _make_service([err, ["never"]], loop)
    state = _State(["hi"])
    state.stop_event.set()

    svc._synthesis_handler(state)
    out = _drain(loop, state)

    assert out == [], "no audio and no ErrorFrame should be produced for an interruption"
    assert calls["n"] == 1, "must not retry into an abandoned turn"


@pytest.mark.unit
def test_max_retries_zero_matches_upstream_behavior(loop):
    """max_retries=0 is the escape hatch back to upstream's single-shot handler."""
    err = _FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED)
    svc, calls = _make_service([err, ["never"]], loop, max_retries=0)
    state = _State(["hi"])

    svc._synthesis_handler(state)
    out = _drain(loop, state)

    assert calls["n"] == 1
    assert isinstance(out[-1], grpc.RpcError)


@pytest.mark.unit
def test_sentinel_always_terminates_the_response_queue(loop):
    """_process_responses waits on the None sentinel; it must arrive on every path."""
    svc, _ = _make_service([["a"]], loop)
    state = _State(["hi"])

    svc._synthesis_handler(state)

    # _drain returns only when it sees the sentinel; a hang here is the failure.
    assert _drain(loop, state) == ["a"]
    assert state.rpc_call is None, "rpc_call must be cleared so teardown cannot cancel a dead call"


@pytest.mark.unit
@pytest.mark.parametrize(
    "code, expected",
    [
        (grpc.StatusCode.DEADLINE_EXCEEDED, True),
        (grpc.StatusCode.UNAVAILABLE, True),
        (grpc.StatusCode.RESOURCE_EXHAUSTED, True),
        (grpc.StatusCode.ABORTED, True),
        (grpc.StatusCode.INTERNAL, False),
        (grpc.StatusCode.INVALID_ARGUMENT, False),
        (grpc.StatusCode.UNAUTHENTICATED, False),
    ],
)
def test_transient_classification(code, expected):
    assert _is_transient(_FakeRpcError(code)) is expected
    assert (code in TRANSIENT_STATUS_CODES) is expected


@pytest.mark.unit
def test_non_grpc_exception_is_never_transient():
    assert _is_transient(ValueError("boom")) is False

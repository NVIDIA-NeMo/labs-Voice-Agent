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

"""Retry wrapper around pipecat's :class:`NvidiaTTSService`.

Upstream treats any ``SynthesizeOnline`` exception as terminal for the stream
(``pipecat/services/nvidia/tts.py``: *"Once SynthesizeOnline raises, no further
reliable audio is expected"*). For a locally-hosted Riva server that is the
right call, but against NVCF the first request to a cold or rescheduled worker
routinely fails with::

    StatusCode.DEADLINE_EXCEEDED: failed to establish link to worker

That is a routing/capacity condition, not a synthesis failure — the next
attempt normally succeeds within a second. Upstream's terminal handling turns
it into a silently dropped bot turn: the pipeline gets an ``ErrorFrame`` and
the user simply hears nothing.

The STT side needs no equivalent: pipecat's ``NvidiaSTTService`` already
reconnects on any ``grpc.RpcError`` via ``_handle_stream_drop`` ->
``_do_reconnect``, and does so turn-aware. Only TTS lacks a retry path.
"""

import asyncio
import threading
from typing import Any, Iterable, List, Set

import grpc
from loguru import logger
from pipecat.services.nvidia.tts import NvidiaTTSService


# gRPC statuses worth a second attempt. Deliberately narrow:
#
# - DEADLINE_EXCEEDED  NVCF could not route to a worker in time (the observed
#                      "failed to establish link to worker").
# - UNAVAILABLE        transport dropped / worker restarting.
# - RESOURCE_EXHAUSTED capacity throttle; backing off is exactly right.
# - ABORTED            transient server-side abort.
#
# INTERNAL is excluded on purpose: that is what a request-serialization bug
# looks like (e.g. feeding a generator to a unary stub), and retrying a
# deterministic client-side error just multiplies the latency before failing.
TRANSIENT_STATUS_CODES: Set[grpc.StatusCode] = {
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
    grpc.StatusCode.ABORTED,
}


def _is_transient(error: BaseException) -> bool:
    """True when ``error`` is a gRPC failure that is worth retrying."""
    if not isinstance(error, grpc.RpcError):
        return False
    code = error.code() if hasattr(error, "code") else None
    return code in TRANSIENT_STATUS_CODES


class ResilientNvidiaTTSService(NvidiaTTSService):
    """``NvidiaTTSService`` that retries a synthesis stream that never produced audio.

    Args:
        max_retries: Additional attempts after the first failure. ``0`` restores
            upstream behaviour exactly.
        retry_backoff_secs: Base delay between attempts; doubled each retry.
        **kwargs: Forwarded to :class:`NvidiaTTSService`.
    """

    def __init__(self, *, max_retries: int = 2, retry_backoff_secs: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self._max_retries = max(0, int(max_retries))
        self._retry_backoff_secs = max(0.0, float(retry_backoff_secs))

    def _synthesis_handler(self, state) -> None:
        """Run the ``SynthesizeOnline`` stream, retrying establishment failures.

        Overrides upstream's single-shot handler. Same contract: forward each
        response onto ``state.response_queue``, push the exception on failure,
        and always terminate with a ``None`` sentinel so ``_process_responses``
        can finish.

        Retry is deliberately restricted to attempts that produced **no audio**.
        Once any frame has reached the audio context, re-running the request
        would splice a duplicate prefix into the middle of an utterance, which
        is worse than the dropped turn we are trying to avoid. Since
        "failed to establish link to worker" fails before the first response,
        the narrow rule still covers the case that motivated this class.
        """
        event_loop = self.get_event_loop()
        base_req = self._build_base_request()

        # Text chunks already pulled off the queue. A failed attempt has
        # destructively consumed them, so they must be replayed for the retry to
        # synthesize the same utterance.
        replayed: List[str] = []
        saw_end_of_turn = threading.Event()

        def request_generator() -> Iterable[Any]:
            for text in list(replayed):
                if state.stop_event.is_set():
                    return
                base_req.text = text
                yield base_req
            if saw_end_of_turn.is_set():
                return
            while True:
                if state.stop_event.is_set():
                    return
                text = state.text_queue.get()
                if text is None:
                    saw_end_of_turn.set()
                    return
                if state.stop_event.is_set():
                    return
                replayed.append(text)
                base_req.text = text
                yield base_req

        def emit(item: Any) -> None:
            asyncio.run_coroutine_threadsafe(state.response_queue.put(item), event_loop)

        attempt = 0
        try:
            while True:
                produced_audio = False
                try:
                    call = self._service.stub.SynthesizeOnline(
                        request_generator(),
                        metadata=self._service.auth.get_auth_metadata(),
                    )
                    state.rpc_call = call
                    for resp in call:
                        if state.stop_event.is_set():
                            break
                        produced_audio = True
                        emit(resp)
                    return
                except Exception as e:  # noqa: BLE001 — mirrors upstream's catch-all
                    # An interruption cancels the in-flight call, which surfaces
                    # here as CANCELLED. That is expected teardown, not failure:
                    # stay silent and never retry into a turn the user abandoned.
                    if state.stop_event.is_set():
                        return
                    retriable = not produced_audio and _is_transient(e) and attempt < self._max_retries
                    if not retriable:
                        if produced_audio and _is_transient(e):
                            logger.error(
                                f"{self} synthesis stream failed after emitting audio; not retrying "
                                f"(would duplicate the spoken prefix): {e}"
                            )
                        else:
                            logger.error(f"{self} gRPC synthesis stream error: {e}")
                        emit(e)
                        return
                    attempt += 1
                    delay = self._retry_backoff_secs * (2 ** (attempt - 1))
                    logger.warning(
                        f"{self} synthesis stream failed before producing audio "
                        f"(attempt {attempt}/{self._max_retries}), retrying in {delay:.2f}s: {e}"
                    )
                    if state.stop_event.wait(delay):
                        return
        finally:
            state.rpc_call = None
            emit(None)

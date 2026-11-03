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

"""Tests for the get_context_history wait-for-drain race fix.

Observed live on 2026-06-04 in ``eval_20260604_074642/tau2_retail__57``:
the agent's final ``EndConversationTool`` tool_call fired and the bridge's
exit-message handler triggered ``stop_event``, but when the bridge then issued
``get_context_history`` the assistant aggregator hadn't yet committed the
tool_call to its message list. The judge received a stale context, didn't
see the EndConversationTool call, and (incorrectly) deducted 0.15.

Fix: the action's handler polls ``has_function_calls_in_progress`` (a
``@property`` on pipecat's ``LLMResponseAggregator``, NOT a method) until
it returns False, with a 3 s deadline to prevent deadlocks. These tests
verify the polling + commit + timeout behavior with a fake aggregator.
"""

import asyncio
from types import SimpleNamespace

from nemo_voice_agent.pipecat.processors.frameworks.rtvi_actions import (
    create_get_context_history_action,
)


class _FakeAggregator:
    """Stand-in for ``LLMResponseAggregator`` exposing the bits the handler reads.

    ``_in_progress`` flips True when a function call is "in flight" and back
    to False when it's committed; the tests use ``schedule_commit_in`` to
    simulate the race by flipping it asynchronously after a delay.

    **CRITICAL**: ``has_function_calls_in_progress`` must be defined as a
    ``@property`` (not a regular method) to match the real pipecat
    ``LLMResponseAggregator`` shape. Earlier versions of this fake exposed it
    as a method, which silently let the handler ``bug-call`` it as
    ``has_function_calls_in_progress()`` instead of ``has_function_calls_in_progress``
    — that triggered ``'bool' object is not callable`` at runtime in
    production but went undetected in tests because the fake matched the
    buggy call shape. See the 2026-06-04 incident (eval_20260604_103340).
    """

    def __init__(self, in_progress: bool = False, messages=None):
        self._in_progress = in_progress
        self._context = SimpleNamespace(get_messages=lambda: messages or [])

    @property
    def has_function_calls_in_progress(self) -> bool:
        return self._in_progress

    async def schedule_commit_in(self, seconds: float):
        """Simulate the aggregator finishing its commit cycle after `seconds`."""
        await asyncio.sleep(seconds)
        self._in_progress = False


def _build_handler(aggregator, task_ref=None):
    """Extract the inner async handler from the (name, handler) factory pair.

    The handler no longer calls ``_maybe_end_task`` (read-only handlers
    shouldn't end the pipeline task), but the factory function
    still requires a ``task_ref`` parameter for signature compatibility
    with its siblings.
    """
    if task_ref is None:
        task_ref = SimpleNamespace(task=None, running=False)
    _name, handler = create_get_context_history_action(task_ref, aggregator)
    return handler


def test_no_pending_calls_returns_immediately():
    """Common case: aggregator has no in-flight calls → handler returns immediately."""
    agg = _FakeAggregator(in_progress=False, messages=[{"role": "user", "content": "hi"}])
    handler = _build_handler(agg)

    start = asyncio.new_event_loop().time()

    async def run():
        return await handler(None, {})

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(run())
    finally:
        loop.close()
    elapsed = asyncio.new_event_loop().time() - start
    assert result["context"] == str([{"role": "user", "content": "hi"}])
    # Should be well under the 50 ms polling interval since we never entered the loop.
    assert elapsed < 0.5, f"handler took {elapsed:.3f}s with no pending calls — expected near-instant"


def test_waits_for_pending_call_to_commit():
    """Race case: aggregator has an in-flight call that commits after 200 ms.
    Handler should wait, then snapshot the updated context."""

    async def run():
        agg = _FakeAggregator(
            in_progress=True,
            messages=[
                {"role": "assistant", "tool_calls": [{"name": "EndConversationTool"}]},
            ],
        )
        # Simulate the aggregator finishing its commit cycle after 200 ms.
        asyncio.create_task(agg.schedule_commit_in(0.2))

        handler = _build_handler(agg)
        start = asyncio.get_event_loop().time()
        result = await handler(None, {})
        return result, asyncio.get_event_loop().time() - start

    loop = asyncio.new_event_loop()
    try:
        result, elapsed = loop.run_until_complete(run())
    finally:
        loop.close()
    # Should have waited at least 150 ms (close to the 200 ms commit time)
    assert elapsed >= 0.15, f"handler returned in {elapsed:.3f}s — expected to wait for commit"
    # And should have returned the final context with the tool_call
    assert "EndConversationTool" in result["context"]


def test_timeout_falls_back_to_return_with_warning():
    """Stuck case: aggregator never commits → handler must return within ~3 s anyway.
    The 3 s deadline is a hard bound to prevent scenario-cleanup deadlock."""

    async def run():
        agg = _FakeAggregator(in_progress=True, messages=[{"role": "user", "content": "stuck"}])
        # Don't schedule any commit — aggregator stays stuck.
        handler = _build_handler(agg)
        start = asyncio.get_event_loop().time()
        result = await handler(None, {})
        return result, asyncio.get_event_loop().time() - start

    loop = asyncio.new_event_loop()
    try:
        result, elapsed = loop.run_until_complete(run())
    finally:
        loop.close()
    # Should fall back at the 3 s deadline (with small tolerance for polling-interval overshoot)
    assert 2.9 <= elapsed <= 3.5, f"handler took {elapsed:.3f}s — expected ~3s timeout fallback"
    # Returns the stale context (with a warning logged) rather than deadlocking
    assert "stuck" in result["context"]


def test_aggregator_without_property_does_not_crash():
    """Defensive: if the aggregator doesn't expose ``has_function_calls_in_progress`` (e.g.,
    older pipecat or custom aggregator), the handler should just skip the wait and read.
    """

    async def run():
        agg = SimpleNamespace(_context=SimpleNamespace(get_messages=lambda: [{"role": "user", "content": "ok"}]))
        # No has_function_calls_in_progress method.
        handler = _build_handler(agg)
        return await handler(None, {})

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(run())
    finally:
        loop.close()
    assert result["context"] == str([{"role": "user", "content": "ok"}])


def test_handler_does_not_end_pipeline_task():
    """Regression test: the handler must NOT call ``_maybe_end_task`` (which would
    enqueue ``EndWorkerFrame`` and cancel the drain loop's ``await asyncio.sleep``).

    Observed live on 2026-06-04 in bot_agent_server.log: the action arrived at the
    bot but the handler never logged anything for 15 s because EndWorkerFrame
    interrupted the drain mid-wait. The handler is now read-only with no side
    effects on the pipeline.
    """
    queue_calls = []

    class _FakeTask:
        async def queue_frames(self, frames):
            queue_calls.append(frames)

    task_ref = SimpleNamespace(task=_FakeTask(), running=True)
    agg = _FakeAggregator(in_progress=False, messages=[{"role": "user", "content": "x"}])

    async def run():
        handler = _build_handler(agg, task_ref=task_ref)
        return await handler(None, {})

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(run())
    finally:
        loop.close()
    # Handler returned the context successfully...
    assert result["context"] == str([{"role": "user", "content": "x"}])
    # ...AND did not enqueue any frames on the pipeline task (no _maybe_end_task call).
    assert queue_calls == [], (
        "handler must not call _maybe_end_task / queue_frames — that would defeat the drain loop "
        f"by cancelling it mid-wait. queue_frames was called with: {queue_calls}"
    )


def test_exception_in_get_messages_returns_empty_context():
    """Existing behavior preserved: if reading messages raises, handler logs and returns empty.
    The wait-loop must not interact with this failure path."""

    async def run():
        def boom():
            raise RuntimeError("aggregator exploded")

        # has_function_calls_in_progress is a property on the real aggregator, but
        # SimpleNamespace doesn't support @property semantics. The handler reads it
        # as an attribute (no call), so a plain bool value matches the real shape.
        agg = SimpleNamespace(
            has_function_calls_in_progress=False,
            _context=SimpleNamespace(get_messages=boom),
        )
        handler = _build_handler(agg)
        return await handler(None, {})

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(run())
    finally:
        loop.close()
    assert result == {"context": []}


def test_rtvi_handlers_never_end_the_pipeline():
    """No RTVI handler may end the pipeline worker.

    Before pipecat 1.0 these queued an EndTaskFrame, which was inert:
    queue_frames injects downstream while the frame was only handled upstream.
    Pipecat 1.6 handles it downstream too, so the same call now tears down the
    pipeline — taking the WebSocket server (owned by the input transport) with
    it. ``reset`` and ``update_system_prompt`` run at the start of every
    evaluation scenario, so this would kill the bot before its first turn.
    """
    import asyncio as _asyncio

    from nemo_voice_agent.pipecat.processors.frameworks import rtvi_actions

    queued = []

    class _FakeTask:
        async def queue_frames(self, frames):
            queued.extend(frames)

    task_ref = rtvi_actions.TaskRef()
    task_ref.task = _FakeTask()
    task_ref.running = True
    shared_state_ref = rtvi_actions.SharedStateRef()
    agg = _FakeAggregator(in_progress=False, messages=[])

    _, reset_handler = rtvi_actions.create_reset_context_action(task_ref, agg, agg, [], [])
    _, summary_handler = rtvi_actions.create_get_scenario_summary_action(task_ref, shared_state_ref)
    _, history_handler = rtvi_actions.create_get_context_history_action(task_ref, agg)

    async def run():
        await reset_handler(None, {})
        await summary_handler(None, {})
        await history_handler(None, {})

    _asyncio.run(run())

    assert queued == [], f"an RTVI handler queued pipeline frames: {queued}"
    # The helper that used to do it should be gone, not merely unused.
    assert not hasattr(rtvi_actions, "_maybe_end_task")

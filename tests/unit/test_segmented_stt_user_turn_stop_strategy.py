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

"""Unit coverage for the stop strategy that waits for every segment of a segmented STT service."""

import asyncio

from pipecat.frames.frames import TranscriptionFrame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager

from nemo_voice_agent.pipecat.frames.frames import STTSegmentDoneFrame
from nemo_voice_agent.pipecat.turns import SegmentedSTTUserTurnStopStrategy


def _start():
    return VADUserStartedSpeakingFrame(start_secs=0.1)


def _stop():
    # The offline SpeechLM configs wait 1.2 s of silence, which is longer than pipecat's default STT p99 latency of
    # 1.0 s, so the base strategy's own STT wait is zero.
    return VADUserStoppedSpeakingFrame(stop_secs=1.2)


def _text(text):
    # A segmented STT service marks every transcript finalized.
    return TranscriptionFrame(text, "user", "now", finalized=True)


def _done():
    return STTSegmentDoneFrame()


def _events(stop_strategy, script):
    """Run ``script`` through a VAD-driven turn controller and return the turn events interleaved with markers.

    ``script`` holds frames, sleeps (floats, in seconds) and markers (strings).
    """

    async def run():
        controller = UserTurnController(
            user_turn_strategies=UserTurnStrategies(start=[VADUserTurnStartStrategy()], stop=[stop_strategy]),
            user_turn_stop_timeout=None,
        )
        events = []

        def record(name):
            async def handler(*_args, **_kwargs):
                events.append(name)

            return handler

        for name in ("on_user_turn_started", "on_user_turn_stopped"):
            controller.add_event_handler(name, record(name.removeprefix("on_user_turn_")))
        for name in (
            "on_push_frame",
            "on_broadcast_frame",
            "on_user_turn_inference_triggered",
            "on_reset_aggregation",
        ):
            controller.add_event_handler(name, record(None))

        await controller.setup(TaskManager())
        for step in script:
            if isinstance(step, float):
                await asyncio.sleep(step)
            elif isinstance(step, str):
                events.append(step)
            else:
                await controller.process_frame(step)
        await asyncio.sleep(0.05)
        await controller.cleanup()
        return [e for e in events if e is not None]

    return asyncio.run(run())


# The order in the GA run's agent log: the user paused mid-utterance, and the second segment's VAD start (a system
# frame) overtook the first segment's transcript (a data frame) on the way to the aggregator.
_SECOND_VAD_START_OVERTAKES_FIRST_TRANSCRIPT = [
    _start(),
    _stop(),
    _start(),
    _text("Hi, can you tell me what is going on with my flight?"),
    _done(),
    _stop(),
    0.3,
    "second transcript",
    _text("We took off at the scheduled time and then had to turn around."),
    _done(),
]


def test_pipecat_strategy_ends_the_turn_before_the_second_segment_is_transcribed():
    """Pins the bug: the first transcript leaves pipecat's finalized flag set, so the turn ends at the second VAD stop.

    If this starts failing, pipecat has fixed the race and the segmented strategy may no longer be needed.
    """
    strategy = SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.0)

    assert _events(strategy, _SECOND_VAD_START_OVERTAKES_FIRST_TRANSCRIPT) == [
        "started",
        "stopped",
        "second transcript",
    ]


def test_turn_waits_for_the_second_segment_transcript():
    strategy = SegmentedSTTUserTurnStopStrategy(user_speech_timeout=0.0)

    assert _events(strategy, _SECOND_VAD_START_OVERTAKES_FIRST_TRANSCRIPT) == [
        "started",
        "second transcript",
        "stopped",
    ]
    assert strategy._pending_segments == 0


def test_single_segment_turn_ends_when_its_transcript_arrives():
    """A turn without a mid-utterance pause ends as it does with pipecat's strategy."""
    strategy = SegmentedSTTUserTurnStopStrategy(user_speech_timeout=0.0)

    assert _events(strategy, [_start(), _stop(), 0.1, "transcript", _text("Yes, please."), _done()]) == [
        "started",
        "transcript",
        "stopped",
    ]


def test_turn_ends_at_a_pause_whose_transcript_arrives_first():
    """A transcript that arrives before the user resumes ends the turn, as with pipecat's strategy.

    That is a pause longer than ``vad.stop_secs`` ending the turn, not a pending segment, so nothing is held.
    """
    strategy = SegmentedSTTUserTurnStopStrategy(user_speech_timeout=0.0)

    assert _events(strategy, [_start(), _stop(), _text("First part."), _done(), 0.05, "resumed", _start()]) == [
        "started",
        "stopped",
        "resumed",
        "started",
    ]


def test_an_empty_last_segment_ends_the_turn_when_it_is_done():
    """An empty or failed transcription sends no transcript, only the done frame, and the turn ends on it."""
    strategy = SegmentedSTTUserTurnStopStrategy(user_speech_timeout=0.0)

    script = [_start(), _stop(), _start(), _text("First part."), _done(), _stop(), 0.1, "empty segment done", _done()]
    assert _events(strategy, script) == ["started", "empty segment done", "stopped"]


def test_an_empty_segment_does_not_hold_the_next_utterance():
    """A turn whose only segment was empty stays open, and the next utterance ends it as soon as it is transcribed."""
    strategy = SegmentedSTTUserTurnStopStrategy(user_speech_timeout=0.0)

    script = [_start(), _stop(), _done(), 0.1, _start(), _stop(), 0.1, "transcript", _text("Hello?"), _done()]
    assert _events(strategy, script) == ["started", "transcript", "stopped"]
    assert strategy._pending_segments == 0


def test_a_segment_that_never_finishes_is_given_up_after_the_timeout():
    strategy = SegmentedSTTUserTurnStopStrategy(user_speech_timeout=0.0, pending_transcript_timeout=0.3)

    script = [_start(), _stop(), _start(), _text("First part."), _done(), _stop(), 0.15, "before", 0.3, "after"]
    assert _events(strategy, script) == ["started", "before", "stopped", "after"]
    assert strategy._pending_segments == 0


def test_each_vad_stop_restarts_the_timeout():
    """The backstop counts from the latest VAD stop, not from the segment that never finished."""
    strategy = SegmentedSTTUserTurnStopStrategy(user_speech_timeout=0.0, pending_transcript_timeout=0.3)

    # The first segment never finishes. The second VAD stop comes 0.2 s later, so the backstop fires at about 0.5 s.
    script = [_start(), _stop(), 0.2, _start(), _stop(), _text("Hello?"), _done(), 0.2, "0.4 s", 0.2, "0.6 s"]
    assert _events(strategy, script) == ["started", "0.4 s", "stopped", "0.6 s"]


def test_a_segment_given_up_on_cannot_clear_a_newer_segment():
    """A late done frame belongs to the segment the backstop gave up on, so the newer segment is still waited for.

    The STT service is still busy with the first segment when the user speaks again, so the second segment's VAD
    frames (system frames) reach the aggregator before the first segment's late transcript and done frame.
    """
    strategy = SegmentedSTTUserTurnStopStrategy(user_speech_timeout=0.0, pending_transcript_timeout=0.3)

    script = [
        _start(),
        _stop(),
        0.4,
        "first segment given up",
        _start(),
        _stop(),
        _text("First part, late."),
        _done(),
        0.2,
        "second transcript",
        _text("Second part."),
        _done(),
    ]
    assert _events(strategy, script) == ["started", "first segment given up", "second transcript", "stopped"]
    assert (strategy._pending_segments, strategy._abandoned_segments) == (0, 0)


def test_a_segment_finishing_after_the_timeout_does_not_unbalance_the_count():
    """The count floors at zero, so a segment already given up on cannot hold a later turn when it finishes."""
    strategy = SegmentedSTTUserTurnStopStrategy(user_speech_timeout=0.0, pending_transcript_timeout=0.1)

    script = [
        _start(),
        _stop(),
        _start(),
        _text("First part."),
        _done(),
        _stop(),
        0.2,
        "given up",
        _text("Second part, after the timeout."),
        _done(),
        "next turn",
        _start(),
        _stop(),
        _text("Next utterance."),
        _done(),
    ]
    assert _events(strategy, script) == ["started", "stopped", "given up", "next turn", "started", "stopped"]
    assert (strategy._pending_segments, strategy._abandoned_segments) == (0, 0)

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

"""User-turn stop strategy for a segmented STT service."""

import asyncio
from typing import Optional

from loguru import logger
from pipecat.frames.frames import Frame, VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy

from nemo_voice_agent.pipecat.frames.frames import STTSegmentDoneFrame


class SegmentedSTTUserTurnStopStrategy(SpeechTimeoutUserTurnStopStrategy):
    """Speech-timeout stop strategy that keeps the user turn open until every speech segment is transcribed.

    A segmented STT service such as ``NemoSpeechLMSTTService`` transcribes each VAD segment after the segment's
    ``VADUserStoppedSpeakingFrame``. A user who pauses mid-utterance for longer than ``vad.stop_secs`` produces two
    segments in one turn, and the second segment's transcript arrives after its VAD stop.

    Pipecat's ``SpeechTimeoutUserTurnStopStrategy`` waits for that transcript through one "finalized" flag per turn,
    which the first segment's transcript can leave set. VAD frames are system frames, so the second segment's VAD
    start can overtake the first transcript on the way to the aggregator and clear the flag before that transcript
    sets it again. The turn then ends at the second VAD stop, and the second transcript lands in the next user turn.

    This strategy counts segments instead: one per VAD stop, cleared by the ``STTSegmentDoneFrame`` that the STT
    service pushes after each segment, whether the transcription had text, came back empty, or failed. The count does
    not depend on the order in which VAD frames and transcripts arrive. While a segment is pending, the turn is held,
    and the frame that clears the last pending segment ends it. ``pending_transcript_timeout`` is a backstop for a
    segment that does not finish in time: the strategy gives up on it that many seconds after the latest VAD stop, and
    ignores its done frame if it finishes later.

    Use it only with an STT service that pushes one ``STTSegmentDoneFrame`` per VAD stop, and only when the turn's
    text comes from those transcripts. An omni LLM hears the audio itself, and ``UserAudioBuffer`` consumes the
    transcripts before the aggregator, so its turn has no reason to wait for them.
    """

    def __init__(self, *, pending_transcript_timeout: float = 10.0, **kwargs):
        """Initialize the strategy.

        Args:
            pending_transcript_timeout: Seconds after the latest VAD stop to wait for a pending segment to finish
                before giving up on it. Set it above the slowest transcription you expect: a segment given up on
                ends the turn without its text, which then starts the next user message.
            **kwargs: Passed to ``SpeechTimeoutUserTurnStopStrategy``.
        """
        super().__init__(**kwargs)
        self._pending_transcript_timeout = pending_transcript_timeout
        # Segments sent to the STT service that have not finished yet. Like the live VAD state, this describes the
        # STT pipeline rather than the turn, so it survives turn boundaries.
        self._pending_segments = 0
        # Segments the backstop gave up on that have not finished yet. The STT service transcribes one segment at a
        # time and its done frames stay in order, so the next done frames belong to these, not to newer segments.
        self._abandoned_segments = 0
        # The base strategy decided to end the turn while a segment was pending.
        self._stop_held = False
        self._pending_timeout_task: Optional[asyncio.Task] = None

    async def handle_user_turn_started(self):
        """Ready the strategy to detect the end of the turn now starting."""
        await super().handle_user_turn_started()
        self._stop_held = False

    async def handle_user_turn_stopped(self):
        """Clear per-turn state once the turn has ended."""
        await super().handle_user_turn_stopped()
        self._stop_held = False

    async def cleanup(self):
        """Cleanup the strategy."""
        await super().cleanup()
        await self._cancel_pending_timeout()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Count pending segments, then let the speech-timeout logic run.

        Args:
            frame: The frame to be analyzed.

        Returns:
            Always returns CONTINUE so subsequent stop strategies are evaluated.
        """
        if isinstance(frame, STTSegmentDoneFrame):
            if self._abandoned_segments:
                # A segment the backstop gave up on finished late. It must not clear a newer segment's count.
                self._abandoned_segments -= 1
                return ProcessFrameResult.CONTINUE
            # Floored at zero, so a done frame without a counted VAD stop cannot hold a later turn.
            self._pending_segments = max(0, self._pending_segments - 1)
            if not self._pending_segments:
                await self._cancel_pending_timeout()
                # The base strategy wanted to end the turn while this segment was pending. Nothing that would have
                # changed its mind has happened since, because a VAD start or a turn boundary clears the hold.
                if self._stop_held:
                    await self.trigger_user_turn_stopped()
            return ProcessFrameResult.CONTINUE

        if isinstance(frame, VADUserStartedSpeakingFrame):
            # The user is speaking again, so the base strategy decides afresh after the next VAD stop.
            self._stop_held = False
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._pending_segments += 1
            await self._restart_pending_timeout()
        return await super().process_frame(frame)

    async def trigger_user_turn_stopped(self):
        """End the user turn, or hold it while a segment is still being transcribed."""
        if self._pending_segments:
            if not self._stop_held:
                logger.debug(f"{self}: holding the user turn for {self._pending_segments} pending segment(s)")
            self._stop_held = True
            return
        self._stop_held = False
        await super().trigger_user_turn_stopped()

    async def _restart_pending_timeout(self):
        await self._cancel_pending_timeout()
        self._pending_timeout_task = self.task_manager.create_task(
            self._pending_timeout_handler(),
            f"{self}::_pending_timeout_handler",
        )

    async def _cancel_pending_timeout(self):
        # VAD frames and segment-done frames reach the strategy from different aggregator tasks, so another call can
        # run while this one awaits the cancellation. Clear the handle first, so that call's new timer is not lost.
        task, self._pending_timeout_task = self._pending_timeout_task, None
        if task:
            await self.task_manager.cancel_task(task)

    async def _pending_timeout_handler(self):
        try:
            await asyncio.sleep(self._pending_transcript_timeout)
        except asyncio.CancelledError:
            return
        if self._pending_timeout_task is asyncio.current_task():
            self._pending_timeout_task = None

        logger.warning(
            f"{self}: {self._pending_segments} segment(s) still not transcribed "
            f"{self._pending_transcript_timeout}s after the last VAD stop; giving up on them"
        )
        self._abandoned_segments += self._pending_segments
        self._pending_segments = 0
        if self._stop_held:
            await self.trigger_user_turn_stopped()

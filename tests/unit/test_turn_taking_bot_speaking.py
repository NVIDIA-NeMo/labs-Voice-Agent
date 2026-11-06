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

"""``_bot_speaking`` must stay True across the sentence boundaries of one bot reply.

pipecat emits ``BotStoppedSpeakingFrame`` followed right away by ``BotStartedSpeakingFrame`` between TTS sentences.
The stop starts the ``bot_stop_delay`` timer; the next start has to cancel it, otherwise the flag drops to False
while the bot is still talking and backchannels from then on interrupt the bot.
"""

import asyncio
from types import SimpleNamespace

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transcriptions.language import Language

from nemo_voice_agent.pipecat.services.nemo import turn_taking as turn_taking_module
from nemo_voice_agent.pipecat.services.nemo.turn_taking import NeMoTurnTakingService


DOWN = FrameDirection.DOWNSTREAM


def _service(monkeypatch, **kwargs):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(turn_taking_module, "time", SimpleNamespace(time=lambda: clock.now))
    kwargs.setdefault("backchannel_phrases", ["mmhmm"])
    kwargs.setdefault("max_buffer_size", 100)  # no interim frames
    kwargs.setdefault("bot_stop_delay", 0.5)
    service = NeMoTurnTakingService(**kwargs)
    service.pushed = []

    async def push(frame, direction=None):
        service.pushed.append((frame, direction))

    service.push_frame = push
    return service, clock


def _process(service, frame):
    asyncio.run(service.process_frame(frame, DOWN))


def _sentence_boundary(service):
    _process(service, BotStoppedSpeakingFrame())
    _process(service, BotStartedSpeakingFrame())


def test_bot_speaking_survives_sentence_boundary(monkeypatch):
    service, clock = _service(monkeypatch)
    _process(service, BotStartedSpeakingFrame())
    clock.now += 1.2
    _sentence_boundary(service)
    clock.now += 0.6  # longer than bot_stop_delay, still inside the second sentence
    _process(service, TextFrame("any frame"))

    assert service._bot_speaking is True


def test_bot_speaking_turns_false_after_real_stop(monkeypatch):
    service, clock = _service(monkeypatch)
    _process(service, BotStartedSpeakingFrame())
    clock.now += 1.2
    _process(service, BotStoppedSpeakingFrame())
    clock.now += 0.6
    _process(service, TextFrame("any frame"))

    assert service._bot_speaking is False


def test_backchannel_in_second_sentence_does_not_interrupt(monkeypatch):
    service, clock = _service(monkeypatch)
    _process(service, BotStartedSpeakingFrame())
    clock.now += 1.2
    _sentence_boundary(service)
    clock.now += 0.6
    asyncio.run(service._handle_vad_user_started_speaking(VADUserStartedSpeakingFrame(), DOWN))
    service.pushed.clear()
    for text in [" mmhmm", "<EOU>"]:
        frame = InterimTranscriptionFrame(text, "", "now", Language.EN_US, result={"text": text})
        _process(service, frame)

    frames = [frame for frame, _ in service.pushed]
    assert not any(isinstance(frame, InterruptionFrame) for frame in frames)
    transcripts = [
        (frame.text, direction) for frame, direction in service.pushed if isinstance(frame, TranscriptionFrame)
    ]
    assert transcripts == [("(mmhmm)", FrameDirection.UPSTREAM)]


@pytest.mark.parametrize("bot_stop_delay", [None, 0.0])
def test_no_bot_stop_delay_stops_bot_right_away(monkeypatch, bot_stop_delay):
    """``bot_stop_delay: null`` means no delay, like ``0``. It used to raise ``TypeError`` on every later frame."""
    service, clock = _service(monkeypatch, bot_stop_delay=bot_stop_delay)
    _process(service, BotStartedSpeakingFrame())
    clock.now += 1.2
    _process(service, BotStoppedSpeakingFrame())

    assert service._bot_speaking is False

    clock.now += 0.1
    _process(service, TextFrame("next frame"))

    assert service._bot_speaking is False
    assert [frame.text for frame, _ in service.pushed] == ["next frame"]


@pytest.mark.parametrize("bot_stop_delay", [None, 0.0])
def test_no_bot_stop_delay_keeps_bot_speaking_across_sentence_boundary(monkeypatch, bot_stop_delay):
    service, clock = _service(monkeypatch, bot_stop_delay=bot_stop_delay)
    _process(service, BotStartedSpeakingFrame())
    clock.now += 1.2
    _sentence_boundary(service)
    clock.now += 0.6
    _process(service, TextFrame("any frame"))

    assert service._bot_speaking is True

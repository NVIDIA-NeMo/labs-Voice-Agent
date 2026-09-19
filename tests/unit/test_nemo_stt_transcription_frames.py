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

import inspect

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

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

"""Offline unit coverage for diarization response handling."""

import asyncio
from types import SimpleNamespace

import numpy as np
from pipecat.frames.frames import ErrorFrame, TranscriptionFrame
from pipecat.transcriptions.language import Language

from nemo_voice_agent.pipecat.frames.frames import DiarResultFrame
from nemo_voice_agent.pipecat.services.nemo.diar import NemoDiarService


def _diar():
    service = NemoDiarService.__new__(NemoDiarService)
    service._params = SimpleNamespace(threshold=0.5, buffer_size=2, language=Language.EN_US)
    service._current_speaker_id = None
    service._audio_buffer = []
    service._vad_user_speaking = True
    service._enabled = True
    service._queue = asyncio.Queue()
    service._has_generated_metrics = False
    service.frames = []

    async def push(frame, direction=None):
        service.frames.append(frame)

    async def noop(*args, **kwargs):
        return None

    service.push_frame = push
    service.start_ttfb_metrics = service.start_processing_metrics = noop
    return service


def test_diar_dominant_speaker_result_error_buffer_and_reset():
    service = _diar()
    assert service.can_generate_metrics() is True
    assert service._get_dominant_speaker_id(np.zeros((2, 2))) is None
    assert service._get_dominant_speaker_id(np.array([[0.9, 0.1], [0.8, 0.2]])) == 0
    assert service._get_dominant_speaker_id(np.array([[0.9, 0.1], [0.1, 0.9], [0.1, 0.8]])) == 1
    asyncio.run(service._handle_diarization_result(np.array([[0.9, 0.1]])))
    assert isinstance(service.frames[-1], DiarResultFrame)
    asyncio.run(service._handle_diarization_result(None))
    service._get_dominant_speaker_id = lambda value: (_ for _ in ()).throw(RuntimeError("bad diar"))
    asyncio.run(service._handle_diarization_result(np.ones((1, 1))))
    assert isinstance(service.frames[-1], ErrorFrame)
    assert asyncio.run(_collect(service.run_stt(b"a"))) == [None]
    assert asyncio.run(_collect(service.run_stt(b"b"))) == [None]
    assert asyncio.run(service._queue.get()) == b"ab"
    service._model = SimpleNamespace(reset_state=lambda: None)
    service.reset()
    assert service._audio_buffer == [] and service._vad_user_speaking is False


async def _collect(generator):
    return [value async for value in generator]

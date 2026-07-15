# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline unit coverage for diarization and Nemotron ASR response handling."""

import asyncio
from types import SimpleNamespace

import numpy as np
from pipecat.frames.frames import ErrorFrame, TranscriptionFrame
from pipecat.transcriptions.language import Language

from nemo_voice_agent.pipecat.frames.frames import DiarResultFrame
from nemo_voice_agent.pipecat.frames.riva import RivaInterimTranscriptionFrame
from nemo_voice_agent.pipecat.services.nemo.diar import NemoDiarService
from nemo_voice_agent.pipecat.services.riva_speech import NemotronASRService


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


def _result(text, *, final=False, stability=0.0, audio_processed=0.0, alternatives=True):
    alts = [SimpleNamespace(transcript=text)] if alternatives else []
    return SimpleNamespace(alternatives=alts, is_final=final, stability=stability, audio_processed=audio_processed)


def _asr():
    service = NemotronASRService.__new__(NemotronASRService)
    service._name = "test-asr"
    service._generate_interruptions = False
    service._vad_state = None
    service._audio_duration_counter = 1.0
    service._language_code = Language.EN_US
    service.last_transcript_frame = None
    service.frames, service.transcriptions = [], []

    async def push(frame, direction=None):
        service.frames.append(frame)

    async def transcription(text, final, language):
        service.transcriptions.append((text, final, language))

    async def noop(*args, **kwargs):
        return None

    service.push_frame = push
    service._handle_transcription = transcription
    service.start_ttfb_metrics = service.stop_ttfb_metrics = service.stop_processing_metrics = noop
    return service


def test_nemotron_asr_handles_final_stable_partial_and_empty_results():
    service = _asr()
    response = SimpleNamespace(
        results=[_result("", alternatives=False), _result("final", final=True, audio_processed=0.5)]
    )
    asyncio.run(service._handle_response(response))
    assert any(isinstance(frame, TranscriptionFrame) and frame.text == "final" for frame in service.frames)
    assert service.transcriptions[-1][0:2] == ("final", True)
    service.frames.clear()
    asyncio.run(service._handle_response(SimpleNamespace(results=[_result("stable", stability=1.0)])))
    assert isinstance(service.frames[-1], RivaInterimTranscriptionFrame)
    assert service.transcriptions[-1][0:2] == ("stable", False)
    before = len(service.frames)
    asyncio.run(service._handle_response(SimpleNamespace(results=[_result("stable", stability=1.0)])))
    assert len(service.frames) == before
    asyncio.run(
        service._handle_response(
            SimpleNamespace(results=[_result("par", stability=0.2), _result("tial", stability=0.2)])
        )
    )
    assert isinstance(service.frames[-1], RivaInterimTranscriptionFrame)
    assert service.frames[-1].text == "partial"


def test_nemotron_asr_iterator_and_run_stt_queue():
    service = _asr()
    service._thread_running = False
    assert iter(service) is service
    try:
        next(service)
    except StopIteration:
        pass
    service._thread_task = SimpleNamespace(done=lambda: False)
    service._queue = asyncio.Queue()
    assert asyncio.run(_collect(service.run_stt(b"pcm"))) == [None]
    assert asyncio.run(service._queue.get()) == b"pcm"

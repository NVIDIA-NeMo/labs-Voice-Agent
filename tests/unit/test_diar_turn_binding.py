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

"""Speaker tags must describe the utterance they are attached to (issue #47).

Diarization results are produced in a worker thread and can arrive after the VAD segment, or the utterance, whose
audio they describe has ended. pipecat also processes VAD frames (SystemFrames) ahead of queued DataFrames such as
``DiarResultFrame``. These tests pin down that evidence is fenced by VAD segment on both sides, and that the tag of
an emitted utterance comes only from that utterance's evidence.
"""

import asyncio
from types import SimpleNamespace

import numpy as np
from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    InterruptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transcriptions.language import Language

from nemo_voice_agent.pipecat.frames.frames import DiarResultFrame
from nemo_voice_agent.pipecat.services.nemo import diar as diar_module
from nemo_voice_agent.pipecat.services.nemo.diar import NeMoDiarInputParams, NemoDiarService, _DiarJob, _DiarResult
from nemo_voice_agent.pipecat.services.nemo.streaming_diar import NeMoStreamingDiarService
from nemo_voice_agent.pipecat.services.nemo.turn_taking import NeMoTurnTakingService


DOWN = FrameDirection.DOWNSTREAM


def _window(*speakers):
    """Build a Sortformer-shaped output window, one row per frame; ``None`` is a silent frame."""
    preds = np.zeros((len(speakers), 4))
    for row, speaker in enumerate(speakers):
        if speaker is not None:
            preds[row, speaker] = 0.9
    return preds


# --- NemoDiarService -----------------------------------------------------------------------------------------------


def _diar(monkeypatch=None):
    """Construct NemoDiarService without a model or a running pipeline, capturing pushed frames."""
    service = NemoDiarService.__new__(NemoDiarService)
    service._name = "diar-binding-test"
    service._enabled = True
    service._params = NeMoDiarInputParams(threshold=0.5)
    service._queue = asyncio.Queue()
    service._response_queue = asyncio.Queue()
    service._use_vad = True
    service._vad_user_speaking = False
    service._audio_buffer = []
    service._current_speaker_id = None
    service._segment_id = None
    service._worker_segment_id = None
    service._model_reset_pending = False
    service._processing_running = False
    service._model = None
    service.pushed = []

    async def push(frame, direction=None):
        service.pushed.append(frame)

    async def noop(*args, **kwargs):
        return None

    service.push_frame = push
    service.start_ttfb_metrics = service.start_processing_metrics = noop
    if monkeypatch is not None:

        async def base_process_frame(self, frame, direction):
            # What the VAD frame carries downstream is whatever state diar holds when STTService forwards it.
            service.forwarded = (frame, service._segment_id, service._vad_user_speaking)

        monkeypatch.setattr(diar_module.STTService, "process_frame", base_process_frame)
    return service


def _diar_frames(service):
    return [(frame.diar_result, frame.segment_id) for frame in service.pushed if isinstance(frame, DiarResultFrame)]


def test_diar_drops_results_whose_vad_segment_has_ended(monkeypatch):
    """A result for segment N is pushed during N, and dropped after N's VAD stop or once N+1 has started."""
    service = _diar(monkeypatch)
    first, second = VADUserStartedSpeakingFrame(), VADUserStartedSpeakingFrame()

    asyncio.run(service.process_frame(first, DOWN))
    asyncio.run(service._handle_diarization_result(_window(0, 0), first.id))
    asyncio.run(service.process_frame(VADUserStoppedSpeakingFrame(), DOWN))
    # In flight when the user stopped: the VAD-stop reset of the speaker must not let it through.
    asyncio.run(service._handle_diarization_result(_window(0, 0), first.id))
    asyncio.run(service.process_frame(second, DOWN))
    asyncio.run(service._handle_diarization_result(_window(0, 0), first.id))
    asyncio.run(service._handle_diarization_result(_window(1, 1), second.id))

    assert _diar_frames(service) == [(0, first.id), (1, second.id)]


def test_diar_updates_segment_state_before_forwarding_vad_frames(monkeypatch):
    """Downstream never sees a VAD frame while diar still accepts results from the previous segment."""
    service = _diar(monkeypatch)
    start = VADUserStartedSpeakingFrame()

    asyncio.run(service.process_frame(start, DOWN))
    assert service.forwarded == (start, start.id, True)

    stop = VADUserStoppedSpeakingFrame()
    asyncio.run(service.process_frame(stop, DOWN))
    assert service.forwarded == (stop, start.id, False)


def test_diar_pushes_evidence_for_every_chunk_with_a_speaker(monkeypatch):
    """Repeated chunks from the same speaker are each pushed (they are votes); silent chunks are not."""
    service = _diar(monkeypatch)
    start = VADUserStartedSpeakingFrame()
    asyncio.run(service.process_frame(start, DOWN))

    for preds in (_window(0, 0), _window(0, 0), _window(0, None), _window(None, 1)):
        asyncio.run(service._handle_diarization_result(preds, start.id))

    assert _diar_frames(service) == [(0, start.id), (0, start.id), (1, start.id)]


def test_speaker_evidence_reads_the_newest_frame_of_the_window():
    """The window overlaps earlier results, so only its newest frame is new evidence."""
    service = _diar()

    # The window is mostly the previous speaker, but the newest audio is speaker 1.
    assert service._get_speaker_evidence(_window(0, 0, 0, 0, 0, 1)) == 1
    # The previous speaker's tail is still in the window, but the newest audio is silence.
    assert service._get_speaker_evidence(_window(0, 0, 0, 0, 0, None)) is None
    # Overlap in the newest frame: the most probable speaker wins, not the lowest index.
    assert service._get_speaker_evidence(np.array([[0.0, 0.0, 0.0, 0.0], [0.55, 0.99, 0.0, 0.0]])) == 1
    # A failed model step returns no frames; that is no evidence, not an error.
    assert service._get_speaker_evidence(np.zeros((0, 4))) is None


def test_reset_drops_in_flight_results_and_defers_the_model_reset(monkeypatch):
    """After an RTVI reset, a result computed before it is dropped, and the model reset waits for the worker."""
    service = _diar(monkeypatch)
    start = VADUserStartedSpeakingFrame()
    asyncio.run(service.process_frame(start, DOWN))

    service.reset()
    asyncio.run(service._handle_diarization_result(_window(0, 0), start.id))

    assert _diar_frames(service) == []
    assert service._model_reset_pending is True
    assert service._segment_id is None and service._vad_user_speaking is False

    service._use_vad = False
    service.reset()
    assert service._vad_user_speaking is True  # without VAD, diar keeps listening after a reset


class _RecordingModel:
    def __init__(self):
        self.calls = []

    def reset_state(self):
        self.calls.append("reset_state")

    def reset_audio_context(self):
        self.calls.append("reset_audio_context")

    def diarize(self, audio):
        self.calls.append(audio)
        return _window(0)


def _run_worker(service, jobs, expected_results):
    """Feed ``jobs`` to the real worker loop on a thread and collect what it sends back."""

    async def scenario():
        loop = asyncio.get_running_loop()
        service.get_event_loop = lambda: loop
        service._processing_running = True
        for job in [*jobs, None]:
            await service._queue.put(job)
        worker = asyncio.ensure_future(asyncio.to_thread(service._diarization_processor))
        try:
            await asyncio.wait_for(asyncio.shield(worker), timeout=10)
        except asyncio.TimeoutError:
            # Unblock the thread so the test fails instead of hanging.
            service._processing_running = False
            await service._queue.put(None)
            await worker
            raise AssertionError("the diarization worker did not stop at the None job")
        # Results are handed back with run_coroutine_threadsafe; let those puts land.
        for _ in range(100):
            if service._response_queue.qsize() >= expected_results:
                break
            await asyncio.sleep(0.01)
        return [service._response_queue.get_nowait() for _ in range(service._response_queue.qsize())]

    return asyncio.run(scenario())


def test_run_stt_stamps_chunks_with_the_open_vad_segment(monkeypatch):
    """Audio queued during a VAD segment carries that segment, and the worker then runs it."""
    service = _diar(monkeypatch)
    service._params.buffer_size = 2
    service._model = _RecordingModel()
    start = VADUserStartedSpeakingFrame()
    asyncio.run(service.process_frame(start, DOWN))

    async def feed():
        for audio in (b"a", b"b"):
            async for _ in service.run_stt(audio):
                pass

    asyncio.run(feed())
    job = service._queue.get_nowait()
    assert job == _DiarJob(start.id, b"ab")

    results = _run_worker(service, [job], 1)
    asyncio.run(service._handle_diarization_result(results[0].preds, results[0].segment_id))
    assert _diar_frames(service) == [(0, start.id)]


def test_reset_audio_context_clears_audio_but_keeps_the_speaker_cache():
    """Speaker ids stay stable across VAD segments only if the streaming state survives the reset."""
    model = NeMoStreamingDiarService.__new__(NeMoStreamingDiarService)
    calls = []
    model.feature_bufferer = SimpleNamespace(reset=lambda: calls.append("feature_bufferer.reset"))
    model.streaming_state = streaming_state = object()
    model.total_preds = total_preds = object()

    model.reset_audio_context()

    assert calls == ["feature_bufferer.reset"]
    assert model.streaming_state is streaming_state and model.total_preds is total_preds


def test_worker_skips_audio_from_ended_segments_and_clears_audio_context_per_segment():
    """A backlog from an ended segment is not run, and the new segment starts from a clean audio window."""
    service = _diar()
    service._model = _RecordingModel()
    service._vad_user_speaking = True
    service._segment_id = 20
    service._worker_segment_id = 10

    results = _run_worker(service, [_DiarJob(10, b"old"), _DiarJob(20, b"a"), _DiarJob(20, b"b")], 2)

    assert service._model.calls == ["reset_audio_context", b"a", b"b"]
    assert [result.segment_id for result in results] == [20, 20]
    assert all(isinstance(result, _DiarResult) and result.error is None for result in results)


def test_worker_applies_a_pending_model_reset_before_the_next_chunk():
    """reset() only flags the model reset; the worker applies it between chunks, never during diarize()."""
    service = _diar()
    service._model = _RecordingModel()
    service._vad_user_speaking = True
    service._segment_id = 20
    service._worker_segment_id = 20
    service._model_reset_pending = True

    _run_worker(service, [_DiarJob(20, b"a")], 1)

    assert service._model.calls == ["reset_state", "reset_audio_context", b"a"]
    assert service._model_reset_pending is False


# --- NeMoTurnTakingService -----------------------------------------------------------------------------------------


def _turn_taking(**kwargs):
    kwargs.setdefault("backchannel_phrases", None)
    kwargs.setdefault("max_buffer_size", 100)  # no interim frames unless a test asks for them
    service = NeMoTurnTakingService(use_diar=True, **kwargs)
    service.pushed = []

    async def push(frame, direction=None):
        service.pushed.append((frame, direction))

    service.push_frame = push
    return service


def _start(service):
    frame = VADUserStartedSpeakingFrame()
    asyncio.run(service._handle_vad_user_started_speaking(frame, DOWN))
    return frame.id


def _stop(service):
    asyncio.run(service._handle_vad_user_stopped_speaking(VADUserStoppedSpeakingFrame(), DOWN))


def _say(service, text):
    frame = InterimTranscriptionFrame(text, "", "now", Language.EN_US, result={"text": text})
    asyncio.run(service._handle_transcription(frame, DOWN))


def _hear(service, speaker, segment_id, chunks=1):
    for _ in range(chunks):
        asyncio.run(service._handle_diar_result(DiarResultFrame(speaker, segment_id=segment_id), DOWN))


def _finals(service):
    return [
        frame.text
        for frame, direction in service.pushed
        if type(frame) is TranscriptionFrame and direction is not FrameDirection.UPSTREAM
    ]


def test_evidence_from_an_ended_segment_does_not_tag_the_next_turn():
    """Forward direction: late results for turn N reach turn-taking after N's VAD stop, or after N+1 started."""
    service = _turn_taking()
    first = _start(service)
    _hear(service, 0, first, chunks=3)
    _say(service, "hello there<EOU>")
    _stop(service)
    _hear(service, 0, first, chunks=3)  # processed after the VAD stop that ended their segment
    second = _start(service)
    _hear(service, 0, first)  # processed after the next segment started
    _say(service, "yes<EOU>")
    _hear(service, 1, second, chunks=3)
    _say(service, "add fries<EOU>")

    assert _finals(service) == ["<speaker_0> hello there", "yes", "<speaker_1> add fries"]


def test_continuation_after_eou_keeps_the_segment_speaker():
    """A continuation with no new evidence after the <EOU> flush keeps this segment's speaker, not an older one."""
    service = _turn_taking()
    first = _start(service)
    _hear(service, 1, first, chunks=3)
    _say(service, "one moment<EOU>")
    _stop(service)
    second = _start(service)
    _hear(service, 0, second)  # all of this segment's evidence arrives before the first flush
    _say(service, "first sentence<EOU>")
    _say(service, "second sentence<EOU>")
    _stop(service)

    assert _finals(service) == [
        "<speaker_1> one moment",
        "<speaker_0> first sentence",
        "<speaker_0> second sentence",
    ]


def test_barge_in_before_eou_does_not_relabel_the_open_utterance():
    """Reverse direction: B is heard before A's <EOU> arrives; A's text keeps A, and B's next text gets B."""
    service = _turn_taking()
    segment = _start(service)
    _hear(service, 0, segment, chunks=6)
    _say(service, "hello there")
    _hear(service, 1, segment, chunks=2)  # B starts before STT has finalized A's utterance
    _say(service, "<EOU>")
    _hear(service, 1, segment, chunks=3)
    _say(service, "how are you<EOU>")
    _say(service, "and fries<EOU>")  # no new evidence: B is still the speaker of this segment

    assert _finals(service) == ["<speaker_0> hello there", "<speaker_1> how are you", "<speaker_1> and fries"]


def test_speaker_evidence_after_the_last_flush_does_not_leak_into_the_next_segment():
    """Evidence without text at VAD stop (a cough, a non-lexical sound) is forgotten with the segment."""
    service = _turn_taking()
    first = _start(service)
    _hear(service, 0, first, chunks=3)
    _say(service, "hello<EOU>")
    _hear(service, 2, first, chunks=2)
    _stop(service)
    _start(service)
    _say(service, "yes<EOU>")

    assert _finals(service) == ["<speaker_0> hello", "yes"]


def test_tag_is_the_majority_vote_and_ties_go_to_the_most_recent_vote():
    service = _turn_taking()
    service._speaker_votes = [0, 0, 1]
    assert service._utterance_speaker_id() == 0
    service._speaker_votes = [1, 0, 0]  # the majority wins over the first vote
    assert service._utterance_speaker_id() == 0
    service._speaker_votes = [0, 0, 1, 1]  # a tie goes to the most recent vote
    assert service._utterance_speaker_id() == 1
    service._speaker_votes = []
    service._segment_speaker_id = 3
    assert service._utterance_speaker_id() == 3


def test_one_noisy_vote_does_not_label_the_utterance():
    service = _turn_taking()
    segment = _start(service)
    _hear(service, 2, segment)
    _hear(service, 0, segment, chunks=4)
    _say(service, "hello<EOU>")

    assert _finals(service) == ["<speaker_0> hello"]


def test_backchannel_votes_do_not_tag_the_next_utterance_in_the_segment():
    """A backchannel suppressed at <EOB> while the bot speaks takes its votes with it."""
    service = _turn_taking(backchannel_phrases=["uh huh"])
    segment = _start(service)
    service._bot_speaking = True
    _hear(service, 1, segment, chunks=3)
    _say(service, "uh huh<EOB>")
    _hear(service, 0, segment, chunks=2)
    _say(service, "wait stop<EOU>")

    assert _finals(service) == ["<speaker_0> wait stop"]


def test_text_transcribed_before_vad_start_is_kept_and_tagged_by_the_new_segment():
    service = _turn_taking()
    _say(service, "hello")  # STT can transcribe the start of an utterance before VAD fires
    segment = _start(service)
    _hear(service, 1, segment, chunks=2)
    _say(service, " there<EOU>")

    assert _finals(service) == ["<speaker_1> hello there"]


def test_without_vad_gating_late_evidence_is_still_fenced_by_segment_id():
    """use_vad=False keeps text buffering open between segments, but segment ids still fence evidence."""
    service = _turn_taking(use_vad=False)
    first = _start(service)
    _hear(service, 0, first, chunks=3)
    _say(service, "hello<EOU>")
    _stop(service)
    second = _start(service)
    _hear(service, 0, first, chunks=3)  # overtaken by the next VAD start
    _hear(service, 1, second)
    _say(service, "add fries<EOU>")

    assert _finals(service) == ["<speaker_0> hello", "<speaker_1> add fries"]


def test_without_vad_gating_reset_and_vad_start_each_forget_evidence():
    service = _turn_taking(use_vad=False)
    segment = _start(service)
    _hear(service, 0, segment, chunks=2)
    service.reset()
    _say(service, "hello<EOU>")

    segment = _start(service)
    _stop(service)
    _hear(service, 2, segment)  # still accepted: without VAD gating the segment is not closed at VAD stop
    _start(service)
    _say(service, "yes<EOU>")

    assert _finals(service) == ["hello", "yes"]


def test_interim_frames_are_tagged_without_consuming_the_utterance_votes():
    service = _turn_taking(max_buffer_size=2)
    segment = _start(service)
    _hear(service, 0, segment, chunks=2)
    _say(service, "one two three")
    _hear(service, 1, segment)
    _say(service, " four<EOU>")

    interims = [frame.text for frame, _ in service.pushed if isinstance(frame, InterimTranscriptionFrame)]
    assert interims == ["<speaker_0> one two"]
    assert _finals(service) == ["<speaker_0> one two three four"]


def test_speaker_tag_does_not_count_toward_max_buffer_size():
    """With diarization on, a two-word backchannel must not interrupt the bot through the interim path."""
    service = _turn_taking(backchannel_phrases=["uh huh"], max_buffer_size=2)
    segment = _start(service)
    service._bot_speaking = True
    _hear(service, 0, segment, chunks=3)
    _say(service, "uh")
    _say(service, " huh")

    assert not any(isinstance(frame, (InterimTranscriptionFrame, InterruptionFrame)) for frame, _ in service.pushed)


def test_unscoped_diar_results_still_tag_the_current_utterance():
    """A producer that does not stamp segment_id (not VAD-scoped) keeps working."""
    service = _turn_taking()
    _start(service)
    asyncio.run(service._handle_diar_result(DiarResultFrame(3), DOWN))
    _say(service, "hi<EOU>")

    assert _finals(service) == ["<speaker_3> hi"]


def test_reset_forgets_speaker_evidence_and_fences_in_flight_results():
    service = _turn_taking()
    segment = _start(service)
    _hear(service, 0, segment, chunks=2)
    service.reset()
    _hear(service, 0, segment)  # computed before the reset
    _start(service)
    _say(service, "hello<EOU>")

    assert _finals(service) == ["hello"]


def test_diar_results_are_ignored_when_diarization_is_off():
    service = _turn_taking()
    service.use_diar = False
    segment = _start(service)
    _hear(service, 0, segment, chunks=3)
    _say(service, "hello<EOU>")

    assert _finals(service) == ["hello"]

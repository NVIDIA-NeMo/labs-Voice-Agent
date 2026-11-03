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

"""Chunked streaming diarization: every 80 ms frame is scored once and becomes at most one speaker vote."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nemo_voice_agent.pipecat.frames.frames import DiarResultFrame
from nemo_voice_agent.pipecat.services.nemo import streaming_diar
from nemo_voice_agent.pipecat.services.nemo.diar import NeMoDiarInputParams, NemoDiarService
from nemo_voice_agent.pipecat.services.nemo.streaming_diar import (
    DiarizationConfig,
    NeMoChunkedStreamingDiarService,
    NeMoStreamingDiarService,
)


SPF = 1280  # samples per 80 ms frame


def _chunked(chunk_len, lc, rc, n_spk=8):
    """NeMoChunkedStreamingDiarService without a model: the step records its input and returns one row per frame."""
    svc = NeMoChunkedStreamingDiarService.__new__(NeMoChunkedStreamingDiarService)
    svc.cfg = DiarizationConfig(device="cpu", chunk_len=chunk_len, chunk_left_context=lc, chunk_right_context=rc)
    svc.device, svc.max_num_speakers, svc.samples_per_frame, svc.feat_per_frame = "cpu", n_spk, SPF, 8
    svc.streaming_state = None
    svc.steps = []

    def preprocessor(input_signal, length):  # 8 feature frames per 80 ms frame, plus the centred-STFT extra frame
        values = input_signal[:, ::160]
        values = torch.cat([values, values[:, -1:]], dim=1)
        return values[:, None, :].expand(1, 2, values.shape[1]), None

    def stream_step(
        processed_signal, processed_signal_length, streaming_state, total_preds, left_offset, right_offset
    ):
        svc.steps.append((processed_signal.shape[1], left_offset, right_offset))
        n_frames = (processed_signal.shape[1] - left_offset - right_offset) // 8
        first = processed_signal[0, left_offset, 0].item()  # value of the chunk's first frame, see below
        rows = torch.zeros((1, n_frames, n_spk))
        rows[0, :, 0] = first + torch.arange(n_frames) / 32768  # frame index of each row, scaled like the audio
        return streaming_state, rows

    svc.diarizer = SimpleNamespace(preprocessor=preprocessor)
    svc.stream_step = stream_step
    svc.reset_audio_context()
    return svc


def _frame(k):
    """80 ms of int16 audio whose samples all encode the frame index ``k`` (as k / 32768 after scaling)."""
    return np.full(SPF, k, dtype=np.int16).tobytes()


def test_chunked_service_scores_each_frame_once_in_order():
    svc = _chunked(chunk_len=3, lc=1, rc=1)
    outputs = [svc.diarize(_frame(k)) for k in range(10)]
    # A chunk of 3 frames is scored once its right-context frame has arrived: after frames 3 and 6 and 9.
    assert [len(o) for o in outputs] == [0, 0, 0, 3, 0, 0, 3, 0, 0, 3]
    rows = np.concatenate(outputs)[:, 0] * 32768
    np.testing.assert_allclose(rows, np.arange(9), atol=1e-3)  # frames 0..8, each exactly once
    # The first chunk has no left context; later chunks get chunk_left_context frames, always rc frames after.
    assert svc.steps == [((3 + 1) * 8, 0, 8), ((1 + 3 + 1) * 8, 8, 8), ((1 + 3 + 1) * 8, 8, 8)]


def test_chunked_service_restarts_chunks_at_a_new_vad_segment():
    svc = _chunked(chunk_len=2, lc=1, rc=1)
    assert len(np.concatenate([svc.diarize(_frame(k)) for k in range(4)])) == 2
    svc.reset_audio_context()  # the frames still waiting for their chunk belong to the previous segment
    out = [svc.diarize(_frame(100 + k)) for k in range(3)]
    assert [len(o) for o in out] == [0, 0, 2]
    np.testing.assert_allclose(out[-1][:, 0] * 32768, [100, 101], atol=1e-3)


def test_warmup_runs_the_model_on_silence_then_starts_clean():
    svc = _chunked(chunk_len=3, lc=1, rc=1)
    svc.init_streaming_state = lambda batch_size=1: "fresh"
    svc.streaming_state = "used"
    svc.warmup()
    assert len(svc.steps) == 3  # 12 frames of silence complete the chunks that end at frames 4, 7 and 10
    assert svc.streaming_state == "fresh" and svc._next_frame == 0 and len(svc._audio) == 0
    assert [len(svc.diarize(_frame(k))) for k in range(4)] == [0, 0, 0, 3]  # the first real frame opens a chunk


def test_build_rejects_a_frame_length_other_than_the_encoder_frame(monkeypatch):
    model = SimpleNamespace(
        sortformer_modules=SimpleNamespace(),
        cfg=SimpleNamespace(preprocessor=SimpleNamespace(window_stride=0.01)),
        encoder=SimpleNamespace(subsampling_factor=8),
        output_subsampling_factor=1,  # a high-resolution checkpoint: one row per 10 ms feature frame
        _check_streaming_parameters=lambda: None,
        eval=lambda: None,
    )
    monkeypatch.setattr(streaming_diar.SortformerEncLabelModel, "from_pretrained", lambda *args, **kwargs: model)
    svc = NeMoStreamingDiarService.__new__(NeMoStreamingDiarService)
    svc.cfg = DiarizationConfig(model_path="nvidia/Nemotron-3-Diarization", device="cpu")
    svc.frame_len_in_secs = 0.04
    with pytest.raises(ValueError, match="set diar.frame_len_in_secs to 0.08"):
        svc.build_diarizer()
    svc.frame_len_in_secs = 0.08
    assert svc.build_diarizer() is model and model.output_subsampling_factor == 8  # pooled to 80 ms rows


def test_diar_service_pushes_one_vote_per_new_row_for_chunked_models():
    service = NemoDiarService.__new__(NemoDiarService)
    service._name = "diar-chunked-test"
    service._params = NeMoDiarInputParams(threshold=0.5)
    service._model = SimpleNamespace(returns_new_rows_only=True)
    service._vad_user_speaking, service._segment_id, service._current_speaker_id = True, 7, None
    pushed = []

    async def push(frame, direction=None):
        pushed.append(frame)

    service.push_frame = push
    rows = np.zeros((3, 8))
    rows[0, 2], rows[2, 5] = 0.9, 0.8  # row 1 is silence
    asyncio.run(service._handle_diarization_result(rows, 7))
    asyncio.run(service._handle_diarization_result(np.zeros((0, 8)), 7))  # no complete chunk yet
    votes = [(f.diar_result, f.segment_id) for f in pushed if isinstance(f, DiarResultFrame)]
    assert votes == [(2, 7), (5, 7)]

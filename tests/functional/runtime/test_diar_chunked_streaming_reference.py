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

"""Chunked streaming diarization checked against NeMo's own streaming loop, with a seeded random model.

The model is built from the Nemotron-3-Diarization architecture config with random weights, so the test needs no
checkpoint download. Random weights make the outputs meaningless as speaker labels, but for the same weights and
input, NeMo's ``forward_streaming`` over the whole signal computes the rows that frame-by-frame streaming must return.
"""

from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.functional

MODEL_CONFIG = Path(__file__).parent / "data" / "nemotron_3_diarization_model_config.yaml"
SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # one 80 ms encoder frame


def _gated_noise(seconds: float = 4.0) -> np.ndarray:
    """Seeded int16 noise switched on and off in 200 ms blocks, so the model input varies between chunks."""
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(int(seconds * SAMPLE_RATE))
    gate = np.repeat(rng.random(int(seconds * 5)) > 0.3, SAMPLE_RATE // 5)
    return (noise * gate * 3000).astype(np.int16)


def test_nemotron_chunked_streaming_matches_nemo_forward_streaming(tmp_path):
    """Frame-by-frame chunked streaming returns the rows NeMo's streaming loop computes for the whole signal."""
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel
    from omegaconf import OmegaConf

    from nemo_voice_agent.pipecat.services.nemo.streaming_diar import (
        DiarizationConfig,
        NeMoChunkedStreamingDiarService,
    )

    torch.manual_seed(0)
    model_path = tmp_path / "random_nemotron_3_diarization.nemo"
    SortformerEncLabelModel(cfg=OmegaConf.load(MODEL_CONFIG)).save_to(str(model_path))

    # The model's streaming settings from server_configs/diar_configs/.
    cfg = DiarizationConfig(
        device="cpu", spkcache_len=264, fifo_len=264, spkcache_update_period=222, chunk_len=3, chunk_right_context=1
    )
    service = NeMoChunkedStreamingDiarService(cfg, str(model_path), sample_rate=SAMPLE_RATE)
    audio = _gated_noise()
    rows = [service.diarize(audio[i : i + FRAME_SAMPLES].tobytes()) for i in range(0, len(audio), FRAME_SAMPLES)]

    assert service.max_num_speakers == 8
    # A chunk of 3 frames is scored once its 1-frame look-ahead arrives: after frames 4, 7, 10, ...
    assert [len(r) for r in rows] == [3 if i >= 3 and i % 3 == 0 else 0 for i in range(len(rows))]

    signal = torch.from_numpy(audio.astype(np.float32) / 32768.0).unsqueeze(0)
    with torch.inference_mode():
        feats, feats_len = service.diarizer.preprocessor(input_signal=signal, length=torch.tensor([signal.shape[1]]))
        expected = service.diarizer.forward_streaming(feats, feats_len)[0].numpy()
    streamed = np.concatenate(rows)

    assert streamed.shape == (48, 8)
    # Chunk-local feature extraction differs from whole-signal extraction by about 6e-5. Dropping the STFT margin
    # frame or the left-context frame moves rows by 1e-3 or more.
    np.testing.assert_allclose(streamed, expected[: len(streamed)], rtol=0, atol=2e-4)

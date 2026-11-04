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
# NOTE: This file will be deprecated in the future, as the new inference pipeline will replace it.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from nemo.collections.asr.models import SortformerEncLabelModel
from nemo.collections.asr.modules.sortformer_modules import StreamingSortformerState
from torch import Tensor

from nemo_voice_agent.pipecat.services.nemo.utils import CacheFeatureBufferer


@dataclass
class DiarizationConfig:
    """Diarization configuration parameters for inference.

    Lengths are in model frames (80 ms). The number of speakers comes from the checkpoint, and
    ``spkcache_update_period=None`` keeps the checkpoint's value.
    """

    model_path: str = "nvidia/diar_streaming_sortformer_4spk-v2"
    device: str = "cuda"

    log: bool = False  # If True, log will be printed
    spkcache_len: int = 188
    spkcache_update_period: Optional[int] = None
    fifo_len: int = 188
    chunk_len: int = 6
    chunk_left_context: int = 1
    chunk_right_context: int = 7


class NeMoStreamingDiarService:
    """Sliding-window streaming: every call re-scores the last ``chunk_len`` frames.

    Only the newest row of each result covers audio that no earlier call reported. Use
    ``NeMoChunkedStreamingDiarService`` to score every frame exactly once instead.
    """

    returns_new_rows_only = False

    def __init__(
        self,
        cfg: DiarizationConfig,
        model: str,
        frame_len_in_secs: float = 0.08,
        sample_rate: int = 16000,
        left_offset: int = 8,
        right_offset: int = 8,
        use_amp: bool = False,
        compute_dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.cfg = cfg
        self.cfg.model_path = model
        self.frame_len_in_secs = frame_len_in_secs
        self.samples_per_frame = round(frame_len_in_secs * sample_rate)
        self.diarizer = self.build_diarizer()
        self.device = cfg.device
        self.use_amp = use_amp
        self.compute_dtype = compute_dtype
        self.left_offset = left_offset
        self.right_offset = right_offset
        self.chunk_size = self.cfg.chunk_len
        self.buffer_size_in_secs = (
            self.cfg.chunk_len * self.frame_len_in_secs + (self.left_offset + self.right_offset) * 0.01
        )
        self.max_num_speakers = self.diarizer.sortformer_modules.n_spk

        self.feature_bufferer = CacheFeatureBufferer(
            sample_rate=sample_rate,
            buffer_size_in_secs=self.buffer_size_in_secs,
            chunk_size_in_secs=self.cfg.chunk_len * self.frame_len_in_secs,
            preprocessor_cfg=self.diarizer.cfg.preprocessor,
            device=self.device,
        )
        self.streaming_state = self.init_streaming_state(batch_size=1)
        self.total_preds = torch.zeros((1, 0, self.max_num_speakers), device=self.diarizer.device)

        print(f"{type(self).__name__} initialized with model `{model}` on device `{self.device}`")

    def build_diarizer(self):
        if self.cfg.model_path.endswith(".nemo"):
            diar_model = SortformerEncLabelModel.restore_from(self.cfg.model_path, map_location=self.cfg.device)
        else:
            diar_model = SortformerEncLabelModel.from_pretrained(self.cfg.model_path, map_location=self.cfg.device)

        # Steaming mode setup
        diar_model.sortformer_modules.chunk_len = self.cfg.chunk_len
        diar_model.sortformer_modules.spkcache_len = self.cfg.spkcache_len
        diar_model.sortformer_modules.chunk_left_context = self.cfg.chunk_left_context
        diar_model.sortformer_modules.chunk_right_context = self.cfg.chunk_right_context
        diar_model.sortformer_modules.fifo_len = self.cfg.fifo_len
        diar_model.sortformer_modules.log = self.cfg.log
        if self.cfg.spkcache_update_period is not None:
            diar_model.sortformer_modules.spkcache_update_period = self.cfg.spkcache_update_period
        # Chunks, offsets and speaker votes all count in encoder frames, so the caller's frame must be one.
        encoder_frame_secs = diar_model.cfg.preprocessor.window_stride * diar_model.encoder.subsampling_factor
        if not math.isclose(self.frame_len_in_secs, encoder_frame_secs):
            raise ValueError(
                f"frame_len_in_secs={self.frame_len_in_secs} does not match the {encoder_frame_secs:g} s encoder frame "
                f"of diarization model `{self.cfg.model_path}`; set diar.frame_len_in_secs to {encoder_frame_secs:g}."
            )
        if hasattr(diar_model, "output_subsampling_factor"):
            # High-resolution checkpoints (Nemotron-3-Diarization) predict one row per 10 ms feature frame. Pool them
            # to one row per encoder frame, the frame every caller of `diarize()` works in.
            diar_model.output_subsampling_factor = diar_model.encoder.subsampling_factor
        if hasattr(diar_model, "_check_streaming_parameters"):
            diar_model._check_streaming_parameters()  # validates the cache geometry and the output pooling
        else:
            diar_model.sortformer_modules._check_streaming_parameters()
        diar_model.eval()

        return diar_model

    def print_diar_result(self, diar_result: np.ndarray):
        for t in range(diar_result.shape[0]):
            spk_probs = ""
            for s in range(diar_result.shape[1]):
                spk_probs += f"{diar_result[t, s]:.2f} "
            print(f"Time {t}: {spk_probs}")

    def diarize(self, audio: bytes, stream_id: str = "default") -> np.ndarray:
        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        self.feature_bufferer.update(audio_array)

        features = self.feature_bufferer.get_feature_buffer()
        feature_buffers = features.unsqueeze(0)  # add batch dimension
        feature_buffers = feature_buffers.transpose(1, 2)  # [batch, feature, time] -> [batch, time, feature]
        feature_buffer_lens = torch.tensor([feature_buffers.shape[1]], device=self.device)
        self.streaming_state, chunk_preds = self.stream_step(
            processed_signal=feature_buffers,
            processed_signal_length=feature_buffer_lens,
            streaming_state=self.streaming_state,
            total_preds=self.total_preds,
            left_offset=self.left_offset,
            right_offset=self.right_offset,
        )
        self.total_preds = chunk_preds
        diar_result = chunk_preds[:, -self.chunk_size :, :].clone().cpu().numpy()
        return diar_result[0]  # array of shape [chunk_len, num_speakers]

    def reset_audio_context(self, stream_id: str = "default"):
        """Clear the buffered audio and features but keep the speaker cache, so speaker ids stay stable."""
        self.feature_bufferer.reset()

    def warmup(self, num_frames: int = 12):
        """Run ``num_frames`` frames of silence through the model, then reset it.

        The first forward pass in a process takes seconds while CUDA initializes. Without a warm-up that delay lands
        on the first user turn, whose results then arrive after the turn has ended and are dropped.
        """
        silence = bytes(2 * self.samples_per_frame)  # int16 zeros
        for _ in range(num_frames):
            self.diarize(silence)
        self.reset_state()

    def reset_state(self, stream_id: str = "default"):
        self.feature_bufferer.reset()
        self.streaming_state = self.init_streaming_state(batch_size=1)
        self.total_preds = torch.zeros((1, 0, self.max_num_speakers), device=self.diarizer.device)

    def init_streaming_state(self, batch_size: int = 1) -> StreamingSortformerState:
        """
        Initialize the streaming state for the diarization model.

        Args:
            batch_size: The batch size to use.

        Returns:
            SortformerStreamingState: The initialized streaming state.
        """
        # Use the model's init_streaming_state method but convert to SortformerStreamingState format
        nemo_state = self.diarizer.sortformer_modules.init_streaming_state(
            batch_size=batch_size,
            async_streaming=self.diarizer.async_streaming,
            device=self.device,
        )

        return nemo_state

    def stream_step(
        self,
        processed_signal: Tensor,
        processed_signal_length: Tensor,
        streaming_state: StreamingSortformerState,
        total_preds: Tensor,
        left_offset: int = 0,
        right_offset: int = 0,
    ) -> Tuple[StreamingSortformerState, Tensor]:
        """
        Execute a single streaming step for diarization.

        Args:
            processed_signal: The processed audio signal.
            processed_signal_length: The length of the processed signal.
            streaming_state: The current streaming state.
            total_preds: The total predictions so far.
            left_offset: The left offset for the current chunk.
            right_offset: The right offset for the current chunk.

        Returns:
            Tuple[SortformerStreamingState, Tensor]: The updated streaming state and predictions.
        """
        # Move tensors to correct device
        if processed_signal.device != self.device:
            processed_signal = processed_signal.to(self.device)

        if processed_signal_length.device != self.device:
            processed_signal_length = processed_signal_length.to(self.device)

        if total_preds is not None and total_preds.device != self.device:
            total_preds = total_preds.to(self.device)

        with (
            torch.amp.autocast(device_type=self.device, dtype=self.compute_dtype, enabled=self.use_amp),
            torch.inference_mode(),
            torch.no_grad(),
        ):
            try:
                # Call the model's forward_streaming_step method
                streaming_state, diar_pred_out_stream = self.diarizer.forward_streaming_step(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    streaming_state=streaming_state,
                    total_preds=total_preds,
                    left_offset=left_offset,
                    right_offset=right_offset,
                )
            except Exception as e:
                print(f"Error in diarizer streaming step: {e}")
                # print the stack trace
                import traceback

                traceback.print_exc()
                # Return the existing state and preds if there's an error
                return streaming_state, total_preds

        return streaming_state, diar_pred_out_stream


class NeMoChunkedStreamingDiarService(NeMoStreamingDiarService):
    """Streaming Sortformer run the way NeMo streams it: non-overlapping chunks of ``chunk_len`` frames.

    Each chunk is scored with up to ``chunk_left_context`` frames before it and ``chunk_right_context`` frames after
    it, so every frame enters the FIFO and speaker cache exactly once. ``diarize()`` buffers the incoming audio and
    returns only rows for frames that no earlier call returned: ``chunk_len`` rows once a chunk and its right context
    have arrived, and an empty ``[0, num_speakers]`` array otherwise.
    """

    returns_new_rows_only = True

    def __init__(self, cfg: DiarizationConfig, model: str, frame_len_in_secs: float = 0.08, sample_rate: int = 16000):
        super().__init__(cfg, model, frame_len_in_secs=frame_len_in_secs, sample_rate=sample_rate)
        self.feat_per_frame = self.diarizer.encoder.subsampling_factor  # feature frames per encoder frame
        self.reset_audio_context()

    def reset_audio_context(self, stream_id: str = "default"):
        """Drop buffered audio so the next frame starts a new chunk. The speaker cache is kept."""
        self._audio = np.zeros(0, dtype=np.float32)
        self._first_frame = 0  # frame index of self._audio[0]
        self._next_frame = 0  # first frame not returned yet

    def reset_state(self, stream_id: str = "default"):
        self.reset_audio_context()
        self.streaming_state = self.init_streaming_state(batch_size=1)

    def diarize(self, audio: bytes, stream_id: str = "default") -> np.ndarray:
        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        self._audio = np.concatenate([self._audio, audio_array])
        chunk, max_lc, rc = self.cfg.chunk_len, self.cfg.chunk_left_context, self.cfg.chunk_right_context
        spf = self.samples_per_frame
        rows = [np.zeros((0, self.max_num_speakers), dtype=np.float32)]
        while self._first_frame + len(self._audio) // spf >= self._next_frame + chunk + rc:
            lc = min(max_lc, self._next_frame - self._first_frame)
            # One extra frame before the left context, cut off after feature extraction, gives the first kept
            # feature frame a clean STFT window.
            margin = min(1, self._next_frame - self._first_frame - lc)
            start = self._next_frame - lc - margin
            end = self._next_frame + chunk + rc
            signal = torch.from_numpy(self._audio[(start - self._first_frame) * spf : (end - self._first_frame) * spf])
            signal = signal.to(self.device).unsqueeze(0)
            with torch.inference_mode():
                feats, _ = self.diarizer.preprocessor(
                    input_signal=signal, length=torch.tensor([signal.shape[1]], device=self.device)
                )
            feats = feats[:, :, margin * self.feat_per_frame : (end - start) * self.feat_per_frame].transpose(1, 2)
            self.streaming_state, chunk_preds = self.stream_step(
                processed_signal=feats,
                processed_signal_length=torch.tensor([feats.shape[1]], device=self.device),
                streaming_state=self.streaming_state,
                total_preds=torch.zeros((1, 0, self.max_num_speakers), device=self.device),
                left_offset=lc * self.feat_per_frame,
                right_offset=rc * self.feat_per_frame,
            )
            rows.append(chunk_preds[0].cpu().numpy())
            self._next_frame += chunk
            keep_from = max(self._first_frame, self._next_frame - max_lc - 1)
            self._audio = self._audio[(keep_from - self._first_frame) * spf :]
            self._first_frame = keep_from
        return np.concatenate(rows)

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
    """Diarization configuration parameters for inference."""

    model_path: str = "nvidia/diar_streaming_sortformer_4spk-v2"
    device: str = "cuda"

    log: bool = False  # If True, log will be printed
    max_num_speakers: int = 4
    # ``None`` leaves the checkpoint's own value in place.
    spkcache_len: Optional[int] = None
    spkcache_update_period: int = 144
    fifo_len: int = 188
    chunk_len: int = 6
    chunk_left_context: Optional[int] = None
    chunk_right_context: int = 7


class NeMoStreamingDiarService:
    """Provide stateful streaming Sortformer diarization for Pipecat audio frames.

    Buffers incoming features and maintains Sortformer streaming state across calls.
    """

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
        self.diarizer = self.build_diarizer()
        self.device = cfg.device
        self.use_amp = use_amp
        self.compute_dtype = compute_dtype
        self.frame_len_in_secs = frame_len_in_secs
        self.left_offset = left_offset
        self.right_offset = right_offset
        self.chunk_size = self.cfg.chunk_len
        self.buffer_size_in_secs = (
            self.cfg.chunk_len * self.frame_len_in_secs + (self.left_offset + self.right_offset) * 0.01
        )
        self.max_num_speakers = self.cfg.max_num_speakers

        self.feature_bufferer = CacheFeatureBufferer(
            sample_rate=sample_rate,
            buffer_size_in_secs=self.buffer_size_in_secs,
            chunk_size_in_secs=self.cfg.chunk_len * self.frame_len_in_secs,
            preprocessor_cfg=self.diarizer.cfg.preprocessor,
            device=self.device,
        )
        self.streaming_state = self.init_streaming_state(batch_size=1)
        self.total_preds = torch.zeros((1, 0, self.max_num_speakers), device=self.diarizer.device)

        print(f"NeMoStreamingDiarService initialized with model `{model}` on device `{self.device}`")

    def build_diarizer(self):
        """Load and configure the Sortformer diarization model.

        Loads a local ``.nemo`` checkpoint or a pretrained model, applies the configured streaming overrides,
        validates the resulting streaming parameters, and switches the model to evaluation mode.

        Returns:
            SortformerEncLabelModel: The configured Sortformer diarization model.
        """
        if self.cfg.model_path.endswith(".nemo"):
            diar_model = SortformerEncLabelModel.restore_from(self.cfg.model_path, map_location=self.cfg.device)
        else:
            diar_model = SortformerEncLabelModel.from_pretrained(self.cfg.model_path, map_location=self.cfg.device)

        self.apply_streaming_config(diar_model)
        diar_model.eval()

        return diar_model

    def apply_streaming_config(self, diar_model) -> None:
        """Push the configured streaming overrides onto a loaded Sortformer model, then validate them.

        Args:
            diar_model: A loaded ``SortformerEncLabelModel``.
        """
        modules = diar_model.sortformer_modules

        self._set_streaming_param(modules, ("chunk_len",), self.cfg.chunk_len)
        if self.cfg.spkcache_len is not None:
            self._set_streaming_param(modules, ("spkcache_len",), self.cfg.spkcache_len)
        if self.cfg.chunk_left_context is not None:
            self._set_streaming_param(modules, ("chunk_left_context",), self.cfg.chunk_left_context)
        self._set_streaming_param(modules, ("chunk_right_context",), self.cfg.chunk_right_context)
        self._set_streaming_param(modules, ("fifo_len",), self.cfg.fifo_len)
        self._set_streaming_param(modules, ("log",), self.cfg.log)
        # NeMo renamed this knob from ``spkcache_refresh_rate`` to ``spkcache_update_period``; accept either.
        self._set_streaming_param(
            modules, ("spkcache_update_period", "spkcache_refresh_rate"), self.cfg.spkcache_update_period
        )

        # The model validates its streaming parameters at load time, which is before the overrides above,
        # so re-run the check afterwards. NeMo exposes it on the model in newer releases and only on the
        # sortformer module in 2.7.x -- take whichever this install provides.
        for owner in (diar_model, modules):
            check = getattr(owner, "_check_streaming_parameters", None)
            if check is not None:
                check()
                break

    @staticmethod
    def _set_streaming_param(modules, names: Tuple[str, ...], value) -> None:
        """Assign a Sortformer streaming parameter, tolerating NeMo's attribute renames.

        ``nn.Module.__setattr__`` accepts any name, so writing a parameter that NeMo has since renamed
        fails silently and leaves the checkpoint value in force. Resolve the name against the loaded
        module instead, and raise if none of ``names`` exists.

        Args:
            modules: The model's ``sortformer_modules``.
            names: Accepted attribute names, most current first.
            value: The value to assign.

        Raises:
            AttributeError: If the installed NeMo release exposes none of ``names``.
        """
        for name in names:
            if hasattr(modules, name):
                setattr(modules, name, value)
                return
        raise AttributeError(
            f"None of {names} exist on {type(modules).__name__}. The installed NeMo release may have "
            "renamed this streaming parameter; update DiarizationConfig to match."
        )

    def print_diar_result(self, diar_result: np.ndarray):
        """Print one row of frame-level speaker probabilities per frame.

        Args:
            diar_result: Array of shape ``(frames, speakers)`` holding frame-level speaker probabilities.
        """
        for t in range(diar_result.shape[0]):
            spk_probs = ""
            for s in range(diar_result.shape[1]):
                spk_probs += f"{diar_result[t, s]:.2f} "
            print(f"Time {t}: {spk_probs}")

    def diarize(self, audio: bytes, stream_id: str = "default") -> np.ndarray:
        """Diarize the latest chunk of streaming audio.

        Args:
            audio: Mono signed 16-bit PCM audio bytes.
            stream_id: Identifier retained by the service interface.

        Returns:
            np.ndarray: Frame-level speaker probabilities for the latest chunk, shape ``(frames, speakers)``.
        """
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
        return diar_result[0]  # array of shape [chunk_len, max_num_speakers]

    def reset_state(self, stream_id: str = "default"):
        """Reset the feature buffer, Sortformer streaming state, and accumulated predictions.

        Args:
            stream_id: Identifier retained by the service interface.
        """
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

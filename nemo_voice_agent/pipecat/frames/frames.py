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


from dataclasses import dataclass

import numpy as np
from pipecat.frames.frames import DataFrame, UninterruptibleFrame


@dataclass
class DiarResultFrame(DataFrame):
    """Diarization frame."""

    diar_result: np.ndarray | int
    stream_id: str = "default"


@dataclass
class STTSegmentDoneFrame(DataFrame, UninterruptibleFrame):
    """A segmented STT service has finished transcribing one VAD segment, whatever the result.

    ``NemoSpeechLMSTTService`` pushes one after every segment: behind the segment's ``TranscriptionFrame`` when the
    transcription has text, and on its own when it comes back empty or fails. Both are data frames, so this one
    reaches downstream processors after the transcript. ``SegmentedSTTUserTurnStopStrategy`` counts these to know
    when every segment of a user turn has been transcribed. It is uninterruptible because an interruption flushes
    queued data frames, and a lost one would leave the count waiting for a segment that already finished.
    """

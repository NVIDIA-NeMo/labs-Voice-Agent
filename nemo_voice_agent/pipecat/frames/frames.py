# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import numpy as np
from pipecat.frames.frames import DataFrame


@dataclass
class DiarResultFrame(DataFrame):
    """Diarization frame."""

    diar_result: np.ndarray | int
    stream_id: str = "default"

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Riva-specific text cleaning filter."""

import re

from pipecat.utils.text.base_text_filter import BaseTextFilter


def _normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace/newlines into single spaces; keep edge intent."""
    return re.sub(r"\s+", " ", text)


class RivaTextFilter(BaseTextFilter):
    """Cleans text for TTS by removing markdown, bullets, and excess spacing."""

    async def filter(self, text: str) -> str:
        """Clean and normalize text prior to TTS synthesis."""
        text = re.sub(r"[*_`~\[\]\(\)\{\}<>]", "", text)
        text = re.sub(r"(?m)^\s*\d+\.\s+", "", text)
        text = re.sub(r"(?m)^\s*[•\-]\s+", "", text)
        text = re.sub(r"([\.!\?])(?=[A-Za-z0-9])", r"\1 ", text)
        text = re.sub(r"[^A-Za-z0-9\s\.\,\!\?\-']", " ", text)
        text = _normalize_whitespace(text)
        text = re.sub(r"\s+([,\.!\?])", r"\1", text)
        text = re.sub(r"\s*-\s*", "-", text)
        text = re.sub(r"\s*'\s*", "'", text)
        return text

    async def handle_interruption(self):
        """No-op interruption handler for compatibility.

        Filter is stateless, so nothing to reset on interruption.
        """
        return None

    async def reset_interruption(self):
        """No-op reset handler for compatibility.

        Filter keeps no internal buffers; nothing to restore.
        """
        return None

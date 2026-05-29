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

"""Shared voice-readability rules used across evaluation domains.

Parakeet STT and most TTS engines mangle letter sequences and alphanumeric
identifiers (confirmation numbers, flight numbers, airport codes, SIM PINs,
account numbers) when they're pronounced as words. The rule below tells both
the agent and the simulated user to spell each character one at a time and
defines the prompt convention ``CODE (spelled out as letter, letter, digit, ...)``.

Imported by every domain base scenario that involves spoken alphanumeric IDs.
"""

VOICE_ALPHANUMERIC_RULE = (
    "When speaking confirmation numbers, flight numbers, or airport codes, "
    "spell each character one at a time — letters as letters, digits as words. "
    "Examples: 1A2BC4 (spelled out as one, A, two, B, C, four); "
    "SK123 (spelled out as S, K, one, two, three); "
    "LAX (spelled out as L, A, X); "
    "AUS (spelled out as A, U, S). "
    "Never pronounce these identifiers as words. In prompts and guidelines "
    "you'll see codes written as 'CODE (spelled out as letter, letter, digit, ...)' — "
    "the part in parentheses is how to speak it; the part before is the canonical identifier."
)

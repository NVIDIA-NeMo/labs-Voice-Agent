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

"""A finished decimal must not keep a sentence from flushing."""

import pytest

from nemo_voice_agent.pipecat.utils.text.simple_text_aggregator import find_last_period_index


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "It costs $3.14.",
        "Your total is $19.99.",
        "The average temperature is 72.5.",
    ],
)
def test_sentence_ending_in_decimal_number_is_complete(text):
    idx = find_last_period_index(text)
    assert idx == len(text) - 1


@pytest.mark.unit
def test_decimal_sentence_splits_before_the_next_sentence():
    text = "That will be 3.14. Thank you!"
    assert find_last_period_index(text) == text.index(". Thank you!")


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "It is 3.",
        "It costs $3.",
        "First, let's begin. The meeting is at 3.",
        "3.",
        "1.",
    ],
)
def test_unfinished_digit_period_stays_open(text):
    assert find_last_period_index(text) == -1


@pytest.mark.unit
def test_plain_sentence_still_ends_on_its_period():
    text = "Hello."
    assert find_last_period_index(text) == len(text) - 1

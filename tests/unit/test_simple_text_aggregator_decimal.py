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

import asyncio

import pytest

from nemo_voice_agent.pipecat.utils.text.simple_text_aggregator import (
    SimpleSegmentedTextAggregator,
    find_last_period_index,
)


def _pieces(text: str) -> list[str]:
    """Feed the reply one character at a time, the way a streamed LLM does."""
    aggregator = SimpleSegmentedTextAggregator(punctuation_marks=".!?", min_sentence_length=0)

    async def _run():
        found = []
        for char in text:
            async for item in aggregator.aggregate(char):
                found.append(item.text)
        return found

    return asyncio.run(_run())


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "It costs $3.14. ",
        "Your total is $19.99. ",
        "The average temperature is 72.5. ",
    ],
)
def test_sentence_ending_in_decimal_number_is_complete(text):
    idx = find_last_period_index(text)
    assert idx == len(text) - 2


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


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "Connect to 192.168.",
        "Upgrade to version 3.14.",
        "Your appointment is on 12.25.",
        "Please see section 4.2.",
        "Call us at 555.123.",
        "It costs $3.14.",
    ],
)
def test_number_with_another_dot_stays_open_until_whitespace(text):
    assert find_last_period_index(text) == -1


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        "Upgrade to version 3.14.2 today.",
        "Connect to 192.168.1.1 now.",
        "Your appointment is on 12.25.2026 at noon.",
        "Please see section 4.2.1 for details.",
        "Call us at 555.123.4567 anytime.",
    ],
)
def test_sentence_after_a_multi_dot_number_still_ends(text):
    assert find_last_period_index(text) == len(text) - 1


@pytest.mark.unit
def test_version_and_address_are_not_cut_as_they_arrive():
    version = _pieces("Upgrade to version 3.14.2 today.")
    address = _pieces("Connect to 192.168.1.1 now.")
    assert version == ["Upgrade to version 3.14.2 today."]
    assert address == ["Connect to 192.168.1.1 now."]


@pytest.mark.unit
def test_finished_decimal_flushes_once_a_space_follows():
    assert _pieces("It costs $3.14. Thank you!") == ["It costs $3.14.", " Thank you!"]

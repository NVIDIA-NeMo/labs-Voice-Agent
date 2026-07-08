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

"""Unit tests for lightweight text filtering and segmentation helpers."""

import asyncio

import pytest

from nemo_voice_agent.pipecat.utils.riva_text_filter import RivaTextFilter, _normalize_whitespace
from nemo_voice_agent.pipecat.utils.text.simple_text_aggregator import (
    SimpleSegmentedTextAggregator,
    find_last_comma_index,
    find_last_period_index,
    has_partial_decimal,
)


def _collect_async(async_iterable):
    """Collect an async iterator into a list from a synchronous pytest test."""

    async def _collect():
        """Run the async iteration under a short-lived event loop."""
        return [item async for item in async_iterable]

    return asyncio.run(_collect())


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The price is 3.", True),
        ("The total is ($3.14)", True),
        ("The total is $3.14.", False),
        ("1.", False),
        ("Step 2.", False),
        ("No decimal here", False),
    ],
)
def test_has_partial_decimal_distinguishes_numbers_from_sentence_punctuation(text, expected):
    """Decimal-looking endings are held back, but bullets and complete sentences are not."""
    assert has_partial_decimal(text) is expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello there.", 11),
        ("The fee is 3.", -1),
        ("1. Confirm your name", -1),
        ("Meet at 7 p.m.", -1),
        ("Washington, D.C.", -1),
        ("This is e.g. a sample. Done.", 26),
    ],
)
def test_find_last_period_index_ignores_common_non_sentence_periods(text, expected):
    """The period scanner avoids decimals, bullets, times, and abbreviations."""
    assert find_last_period_index(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello, this is enough", 5),
        ("Price is 1,234,567", -1),
        ("Santa Clara, CA", -1),
        ("No comma here", -1),
    ],
)
def test_find_last_comma_index_requires_a_non_numeric_sentence_boundary(text, expected):
    """Comma segmentation skips number grouping and short geographic suffixes."""
    assert find_last_comma_index(text) == expected


def test_segmented_aggregator_emits_sentence_and_keeps_residual_text():
    """A complete sentence is yielded while text after the delimiter remains buffered."""
    aggregator = SimpleSegmentedTextAggregator(punctuation_marks=".!?", ignore_marks="*", min_sentence_length=4)

    results = _collect_async(aggregator.aggregate("  Hello* there. next"))

    assert [r.text for r in results] == ["  Hello there."]
    assert aggregator._text == " next"


def test_segmented_aggregator_waits_for_partial_decimal():
    """A trailing decimal-like value should not be emitted as a sentence yet."""
    aggregator = SimpleSegmentedTextAggregator(punctuation_marks=".", min_sentence_length=0)

    results = _collect_async(aggregator.aggregate("The cost is 3."))

    assert results == []
    assert aggregator._text == "The cost is 3."


def test_segmented_aggregator_uses_legacy_eos_detection_when_enabled():
    """Legacy EOS detection is used only after the custom punctuation scanner finds no segment."""
    aggregator = SimpleSegmentedTextAggregator(
        punctuation_marks="",
        min_sentence_length=0,
        use_legacy_eos_detection=True,
    )

    results = _collect_async(aggregator.aggregate("Hello world! trailing"))

    assert [r.text for r in results] == ["Hello world!"]
    assert aggregator._text == " trailing"


def test_normalize_whitespace_collapses_runs_without_trimming_edges():
    """Whitespace normalization collapses runs but keeps leading and trailing intent."""
    assert _normalize_whitespace("  hello\n\tthere  ") == " hello there "


def test_riva_text_filter_removes_markup_bullets_and_normalizes_punctuation():
    """The Riva filter strips markup and bullet prefixes while preserving readable text."""
    filtered = asyncio.run(RivaTextFilter().filter("  1. **Hello**(world)!Next - ' spaced ' @@@ "))

    assert filtered == "Hello world! Next-'spaced'"


def test_riva_text_filter_interruption_hooks_are_noops():
    """The interruption hooks are compatibility no-ops for this stateless filter."""
    filter_ = RivaTextFilter()

    assert asyncio.run(filter_.handle_interruption()) is None
    assert asyncio.run(filter_.reset_interruption()) is None

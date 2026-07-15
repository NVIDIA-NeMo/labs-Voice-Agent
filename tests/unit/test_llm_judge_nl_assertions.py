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

"""Tests for ``LLMJudge.judge_scenario`` per-assertion verdict normalization.

The real LLM is patched at the ``requests.post`` boundary so we exercise the
parser/normalizer without an API key. Covers:

- Backward compat: ``nl_assertions=None`` returns the original ``{score, reason}``
  shape with no new fields.
- Happy path: LLM emits a well-formed verdict list → normalized 1:1 with the
  assertion count, ``nl_assertion_pass_rate`` matches the boolean fraction.
- Missing verdicts: LLM omits some entries → filled with ``passed=False``.
- Out-of-range / non-int indices: silently dropped before normalization fills the gaps.
- API error → all verdicts return ``passed=False`` with the error reason.
"""

from types import SimpleNamespace
from unittest.mock import patch

from nemo_voice_agent.evaluation.utils import LLMJudge


def _fake_response(content: str):
    """Build a stand-in for ``requests.Response`` matching the bits the judge uses."""
    return SimpleNamespace(
        json=lambda: {"choices": [{"message": {"content": content}}]},
        raise_for_status=lambda: None,
    )


def _bare_judge() -> LLMJudge:
    """Construct a judge without invoking ``__init__`` (no API key needed)."""
    j = LLMJudge.__new__(LLMJudge)
    j.url = "http://fake"
    j.model = "fake-model"
    j.api_key = "fake"
    j.headers = {}
    j.default_prompt = LLMJudge.DEFAULT_PROMPT
    j.kwargs = {}
    return j


def test_no_nl_assertions_preserves_original_shape():
    """Backward compat: with no nl_assertions, the result has no verdict fields."""
    j = _bare_judge()
    with patch(
        "nemo_voice_agent.evaluation.utils.requests.post",
        return_value=_fake_response('{"score": 0.75, "reason": "Mostly correct"}'),
    ):
        result = j.judge_scenario(reference="r", prediction="p")
    assert result["score"] == 0.75
    assert result["reason"] == "Mostly correct"
    assert "nl_assertion_verdicts" not in result
    assert "nl_assertion_pass_rate" not in result


def test_happy_path_two_passes_one_fail():
    """LLM emits 3 well-formed verdicts → pass rate is 2/3."""
    j = _bare_judge()
    fake = _fake_response(
        '{"score": 0.67, "reason": "Partial", "nl_assertion_verdicts": ['
        '{"index": 1, "passed": true, "reason": "Yes"},'
        '{"index": 2, "passed": false, "reason": "Missed"},'
        '{"index": 3, "passed": true, "reason": "Yes again"}'
        "]}"
    )
    with patch("nemo_voice_agent.evaluation.utils.requests.post", return_value=fake):
        result = j.judge_scenario(reference="r", prediction="p", nl_assertions=["A", "B", "C"])
    assert len(result["nl_assertion_verdicts"]) == 3
    assert [v["passed"] for v in result["nl_assertion_verdicts"]] == [True, False, True]
    assert result["nl_assertion_pass_rate"] == 2 / 3
    # Each verdict includes the assertion text — judge_result.json is self-describing.
    assert [v["assertion"] for v in result["nl_assertion_verdicts"]] == ["A", "B", "C"]


def test_missing_verdict_fills_with_false():
    """LLM omits the verdict for assertion #2 → filled in as passed=False."""
    j = _bare_judge()
    fake = _fake_response(
        '{"score": 0.5, "reason": "x", "nl_assertion_verdicts": ['
        '{"index": 1, "passed": true, "reason": "ok"},'
        '{"index": 3, "passed": true, "reason": "ok"}'
        "]}"
    )
    with patch("nemo_voice_agent.evaluation.utils.requests.post", return_value=fake):
        result = j.judge_scenario(reference="r", prediction="p", nl_assertions=["A", "B", "C"])
    verdicts = result["nl_assertion_verdicts"]
    assert len(verdicts) == 3
    assert verdicts[0]["passed"] is True
    assert verdicts[1]["passed"] is False  # filled in
    assert "Missing" in verdicts[1]["reason"]
    assert verdicts[2]["passed"] is True
    assert result["nl_assertion_pass_rate"] == 2 / 3


def test_out_of_range_indices_are_dropped():
    """Index=0 (1-based start), index=99 (out of range), index='foo' (non-int) — all dropped.
    The 1 valid entry stays; remaining two slots get passed=False fills.
    """
    j = _bare_judge()
    fake = _fake_response(
        '{"score": 0.33, "reason": "x", "nl_assertion_verdicts": ['
        '{"index": 0, "passed": true, "reason": "out-of-range low"},'
        '{"index": 99, "passed": true, "reason": "out-of-range high"},'
        '{"index": "foo", "passed": true, "reason": "non-int"},'
        '{"index": 2, "passed": true, "reason": "valid"}'
        "]}"
    )
    with patch("nemo_voice_agent.evaluation.utils.requests.post", return_value=fake):
        result = j.judge_scenario(reference="r", prediction="p", nl_assertions=["A", "B", "C"])
    verdicts = result["nl_assertion_verdicts"]
    assert [v["passed"] for v in verdicts] == [False, True, False]
    assert result["nl_assertion_pass_rate"] == 1 / 3


def test_strict_passed_boolean():
    """passed=1 (truthy int) and passed='true' (string) should NOT be treated as True —
    we require strict boolean True so the LLM's pseudo-booleans don't sneak in.
    """
    j = _bare_judge()
    fake = _fake_response(
        '{"score": 0.5, "reason": "x", "nl_assertion_verdicts": ['
        '{"index": 1, "passed": 1, "reason": "truthy int"},'
        '{"index": 2, "passed": "true", "reason": "string"},'
        '{"index": 3, "passed": true, "reason": "real bool"}'
        "]}"
    )
    with patch("nemo_voice_agent.evaluation.utils.requests.post", return_value=fake):
        result = j.judge_scenario(reference="r", prediction="p", nl_assertions=["A", "B", "C"])
    assert [v["passed"] for v in result["nl_assertion_verdicts"]] == [False, False, True]
    assert result["nl_assertion_pass_rate"] == 1 / 3


def test_api_error_produces_all_false_verdicts():
    """When the post call raises, every assertion is recorded as failed with the error reason."""
    j = _bare_judge()
    with patch(
        "nemo_voice_agent.evaluation.utils.requests.post",
        side_effect=RuntimeError("connection refused"),
    ):
        result = j.judge_scenario(reference="r", prediction="p", nl_assertions=["A", "B"])
    assert result["score"] == 0.0
    assert "connection refused" in result["reason"]
    verdicts = result["nl_assertion_verdicts"]
    assert len(verdicts) == 2
    assert all(v["passed"] is False for v in verdicts)
    assert all("connection refused" in v["reason"] for v in verdicts)
    assert result["nl_assertion_pass_rate"] == 0.0


def test_empty_assertions_list_treated_as_none():
    """An empty list is falsy → behave like nl_assertions=None (no verdict fields)."""
    j = _bare_judge()
    with patch(
        "nemo_voice_agent.evaluation.utils.requests.post",
        return_value=_fake_response('{"score": 1.0, "reason": "ok"}'),
    ):
        result = j.judge_scenario(reference="r", prediction="p", nl_assertions=[])
    assert "nl_assertion_verdicts" not in result
    assert "nl_assertion_pass_rate" not in result


def test_verdicts_emitted_out_of_order_get_sorted_by_index():
    """The LLM might emit verdicts in any order; the normalized output is index-ordered."""
    j = _bare_judge()
    fake = _fake_response(
        '{"score": 1.0, "reason": "x", "nl_assertion_verdicts": ['
        '{"index": 3, "passed": true, "reason": "third"},'
        '{"index": 1, "passed": true, "reason": "first"},'
        '{"index": 2, "passed": true, "reason": "second"}'
        "]}"
    )
    with patch("nemo_voice_agent.evaluation.utils.requests.post", return_value=fake):
        result = j.judge_scenario(reference="r", prediction="p", nl_assertions=["A", "B", "C"])
    verdicts = result["nl_assertion_verdicts"]
    assert [v["index"] for v in verdicts] == [1, 2, 3]
    assert [v["reason"] for v in verdicts] == ["first", "second", "third"]

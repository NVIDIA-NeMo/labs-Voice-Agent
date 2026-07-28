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

import pytest

from nemo_voice_agent.evaluation.utils import LLMJudge, match_str_and_float, validate_judge_numeric_options


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
    j.timeout = 120.0
    j.compact_context = False
    j.context_message_limit = 40
    j.context_system_string_limit = 2500
    j.context_string_limit = 6000
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


def test_string_matching_uses_local_deterministic_normalization():
    """Text matching should not depend on optional NeMo ASR helper imports."""
    assert match_str_and_float(
        "Flight SK-703!",
        "flight sk703",
        ignore_capitalization=True,
        ignore_punctuation=True,
        clean_text=True,
    )


def test_judge_scenario_uses_available_evidence_without_reference():
    """External agents can be judged from transcript/context/assertions without final artifacts."""
    j = _bare_judge()
    j.timeout = 7
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured["payload"] = json
        captured["timeout"] = timeout
        return _fake_response(
            '{"score": 1.0, "reason": "passed", '
            '"nl_assertion_verdicts": [{"index": 1, "passed": true, "reason": "matched"}]}'
        )

    with patch("nemo_voice_agent.evaluation.utils.requests.post", side_effect=fake_post):
        result = j.judge_scenario(
            conversation=[{"role": "user", "text": "I need to change my flight."}],
            agent_context_history=[{"role": "assistant", "tool_calls": [{"function": {"name": "call_backend"}}]}],
            user_context_history=[{"role": "user", "content": "I need to change my flight."}],
            nl_assertions=["The agent routed the request to an internal task handler."],
        )

    user_content = captured["payload"]["messages"][1]["content"]
    assert "<reference>" not in user_content
    assert "<prediction>" not in user_content
    assert "<conversation>" in user_content
    assert "<agent_context_history>" in user_content
    assert "<user_context_history>" in user_content
    assert "<nl_assertions>" in user_content
    assert captured["timeout"] == 7
    assert result["score"] == 1.0
    assert result["nl_assertion_pass_count"] == 1
    assert result["nl_assertion_total"] == 1
    assert result["nl_assertion_pass_rate"] == 1.0


def test_judge_scenario_returns_zero_without_evidence():
    """The judge endpoint is not called when only expected materials exist."""
    j = _bare_judge()

    with patch(
        "nemo_voice_agent.evaluation.utils.requests.post",
        side_effect=AssertionError("judge endpoint should not be called without evidence"),
    ):
        result = j.judge_scenario(
            reference='{"expected": "answer"}',
            nl_assertions=["The agent should satisfy this assertion."],
        )

    assert result["score"] == 0.0
    assert "No observed agent evidence" in result["reason"]
    assert result["nl_assertion_pass_count"] == 0
    assert result["nl_assertion_total"] == 1
    assert result["nl_assertion_pass_rate"] == 0.0
    assert result["nl_assertion_verdicts"] == [
        {
            "index": 1,
            "assertion": "The agent should satisfy this assertion.",
            "passed": False,
            "reason": result["reason"],
        }
    ]
    assert "<reference>" in result["judge_input"]["user_content"]
    assert "<nl_assertions>" in result["judge_input"]["user_content"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timeout": 0}, "timeout"),
        ({"context_message_limit": 0}, "context_message_limit"),
        ({"context_system_string_limit": -1}, "context_system_string_limit"),
        ({"context_string_limit": 0}, "context_string_limit"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"thinking_token_budget": -1}, "thinking_token_budget"),
    ],
)
def test_judge_init_rejects_invalid_numeric_config(kwargs, message):
    """Direct construction rejects nonsensical numeric judge configuration."""
    with pytest.raises(ValueError, match=message):
        LLMJudge(url="http://fake", model="fake-model", api_key="fake", **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"judge_threshold": -0.1}, "--judge-threshold"),
        ({"judge_threshold": 1.1}, "--judge-threshold"),
        ({"judge_threshold": float("nan")}, "--judge-threshold"),
        ({"judge_timeout": 0}, "--judge-timeout"),
        ({"judge_timeout": float("inf")}, "--judge-timeout"),
        ({"judge_thinking_token_budget": 0}, "--judge-thinking-token-budget"),
        ({"judge_context_message_limit": 0}, "--judge-context-message-limit"),
        ({"judge_context_system_string_limit": -1}, "--judge-context-system-string-limit"),
        ({"judge_context_string_limit": 0}, "--judge-context-string-limit"),
    ],
)
def test_validate_judge_numeric_options_rejects_invalid_values(kwargs, message):
    """Shared CLI validation rejects nonsensical judge numeric options."""
    with pytest.raises(ValueError, match=message):
        validate_judge_numeric_options(**kwargs)


def test_validate_judge_numeric_options_accepts_valid_values():
    """Shared CLI validation accepts valid judge numeric options."""
    validate_judge_numeric_options(
        judge_threshold=0.0,
        judge_timeout=0.1,
        judge_thinking_token_budget=1,
        judge_context_message_limit=1,
        judge_context_system_string_limit=1,
        judge_context_string_limit=1,
    )


def test_context_history_is_not_compacted_by_default():
    """Full context history is sent unless compaction is explicitly enabled."""
    j = _bare_judge()
    captured = {}
    long_content = "x" * 100

    def fake_post(url, *, headers, json, timeout):
        captured["user_content"] = json["messages"][1]["content"]
        return _fake_response('{"score": 1.0, "reason": "ok"}')

    with patch("nemo_voice_agent.evaluation.utils.requests.post", side_effect=fake_post):
        j.judge_scenario(
            agent_context_history=[
                {"role": "system", "content": long_content},
                {"role": "assistant", "content": "first assistant message"},
                {"role": "assistant", "content": "second assistant message"},
            ]
        )

    assert long_content in captured["user_content"]
    assert "first assistant message" in captured["user_content"]
    assert "second assistant message" in captured["user_content"]
    assert "truncated" not in captured["user_content"]


def test_context_history_compaction_is_opt_in_and_configurable():
    """When enabled, compaction uses the limits configured on the judge."""
    j = _bare_judge()
    j.compact_context = True
    j.context_message_limit = 2
    j.context_system_string_limit = 5
    j.context_string_limit = 6
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured["user_content"] = json["messages"][1]["content"]
        return _fake_response('{"score": 1.0, "reason": "ok"}')

    with patch("nemo_voice_agent.evaluation.utils.requests.post", side_effect=fake_post):
        j.judge_scenario(
            agent_context_history=[
                {"role": "system", "content": "system-content"},
                {"role": "assistant", "content": "old-message"},
                {"role": "assistant", "content": "latest-message"},
            ]
        )

    assert "system" in captured["user_content"]
    assert "latest" in captured["user_content"]
    assert "old-message" not in captured["user_content"]
    assert "truncated" in captured["user_content"]


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

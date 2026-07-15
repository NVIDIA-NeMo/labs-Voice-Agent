# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage for the reasoning-budget logits processor."""

from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams

from nemo_voice_agent.vllm.v1.sample.logits_processor.reasoning_budget_logits_processor import (
    ReasoningBudgetLogitsProcessor,
    RequestState,
)


class _Tokenizer:
    _tokens = {"<think>": [1], "</think>": [2], "\n</think>": [3, 2], "\n": [3], "custom": [4]}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return self._tokens[text]


def _processor():
    processor = ReasoningBudgetLogitsProcessor.__new__(ReasoningBudgetLogitsProcessor)
    processor.device = torch.device("cpu")
    processor.pin_memory = False
    processor.tokenizer = _Tokenizer()
    processor.think_start_ids = [1]
    processor.think_end_detect_ids = [2]
    processor.think_end_force_ids = [3, 2]
    processor.newline_ids = [3]
    processor.req_states = {}
    processor.neg_inf = torch.tensor(-float("inf"))
    return processor


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"thinking_budget": -1}, "thinking_budget"),
        ({"thinking_budget": "one"}, "thinking_budget"),
        ({"thinking_budget": 3, "thinking_budget_grace_period": -1}, "grace_period"),
        ({"thinking_budget": 3, "think_start_tokens": 4}, "think_start_tokens"),
        ({"thinking_budget": 3, "think_end_tokens": 4}, "think_end_tokens"),
    ],
)
def test_validate_params_rejects_invalid_options(extra, match):
    with pytest.raises(ValueError, match=match):
        ReasoningBudgetLogitsProcessor.validate_params(SamplingParams.from_optional(extra_args=extra))
    ReasoningBudgetLogitsProcessor.validate_params(SamplingParams.from_optional())
    assert _processor().is_argmax_invariant() is False


def test_new_state_handles_defaults_overrides_prompts_and_existing_tokens():
    processor = _processor()
    assert processor._prompt_ends_with(None, [1], [3]) is False
    assert processor._prompt_ends_with([9, 1, 3], [1], [3]) is True
    assert processor._new_state(SamplingParams.from_optional(), None, []) is None
    assert processor._new_state(SamplingParams.from_optional(extra_args={"thinking_budget": 0}), None, []) is None

    params = SamplingParams.from_optional(
        extra_args={
            "thinking_budget": 5,
            "thinking_budget_grace_period": 0.4,
            "think_start_tokens": "custom",
            "think_end_tokens": "custom",
        }
    )
    state = processor._new_state(params, [4, 3], [8, 8])
    assert state.inside_thinking is True
    assert state.thinking_token_count == 2
    assert state.grace_period == 2
    assert state.detect_start_ids == [4] and state.force_end_ids == [4]

    capped = processor._new_state(
        SamplingParams.from_optional(extra_args={"thinking_budget": 2, "thinking_budget_grace_period": 9}),
        None,
        [1, 7, 2],
    )
    assert capped.grace_period == 2 and capped.stopped_thinking is True


def test_scan_apply_grace_force_and_cleanup(monkeypatch):
    processor = _processor()
    state = RequestState(3, 1, [1], [2], [3, 2], output_tok_ids=[1, 8, 8])
    processor._scan_tokens(state, 0)
    assert state.inside_thinking and state.thinking_token_count == 2
    processor.req_states[0] = state

    logits = torch.zeros((1, 10))
    assert processor.apply(logits) is logits
    assert logits[0, 3] == 5 and logits[0, 2] == 5

    state.output_tok_ids.append(9)
    state.prev_output_length = 3
    processor._scan_tokens(state, state.prev_output_length)
    logits = torch.arange(10, dtype=torch.float32).reshape(1, 10)
    processor.apply(logits)
    assert torch.isneginf(logits[0]).sum() == 9 and logits[0, 3] == 3
    processor.apply(logits)
    assert torch.isneginf(logits[0]).sum() == 9 and logits[0, 2] == 0
    processor.apply(logits)
    assert state.stopped_thinking and not state.inside_thinking

    state.output_tok_ids.append(2)
    state.stopped_thinking = False
    state.inside_thinking = True
    state.prev_output_length = len(state.output_tok_ids) - 1

    def update(states, batch_update, callback):
        assert batch_update is SimpleNamespace
        states[1] = callback(SamplingParams.from_optional(extra_args={"thinking_budget": 2}), [1], [7])

    monkeypatch.setattr(
        "nemo_voice_agent.vllm.v1.sample.logits_processor.reasoning_budget_logits_processor.process_dict_updates",
        update,
    )
    processor.update_state(SimpleNamespace)
    assert 0 not in processor.req_states and 1 in processor.req_states


def test_device_tensor_and_tokenizer_loader(monkeypatch):
    processor = _processor()
    assert processor._device_tensor([1, 2], torch.int64).tolist() == [1, 2]
    seen = {}

    def load(name, **kwargs):
        seen.update(name=name, **kwargs)
        return _Tokenizer()

    monkeypatch.setattr(
        "nemo_voice_agent.vllm.v1.sample.logits_processor.reasoning_budget_logits_processor.get_tokenizer", load
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(tokenizer="fake", trust_remote_code=True, tokenizer_revision="main")
    )
    assert isinstance(processor._load_tokenizer(config), _Tokenizer)
    assert seen == {"name": "fake", "trust_remote_code": True, "revision": "main"}

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the tau2 ``compare_args`` deterministic comparator."""

import pytest

from nemo_voice_agent.evaluation.utils import _filtered_args, _match_action


class TestFilteredArgs:
    """Three cases tau2's ``compare_args`` schema can take."""

    def test_none_compares_all_args(self):
        args = {"user_id": "u1", "city": "NYC", "amount": 42}
        assert _filtered_args(args, None) == args

    def test_empty_list_returns_empty_dict(self):
        # Name-only match semantics — both sides reduce to {} so equality is trivially True.
        args = {"summary": "any free-text content", "user_id": "u1"}
        assert _filtered_args(args, []) == {}

    def test_subset_returns_only_listed_keys(self):
        args = {"user_id": "u1", "city": "NYC", "amount": 42}
        assert _filtered_args(args, ["user_id"]) == {"user_id": "u1"}
        assert _filtered_args(args, ["user_id", "city"]) == {"user_id": "u1", "city": "NYC"}

    def test_subset_missing_key_returns_none(self):
        # Both sides apply the same filter, so missing-on-both still matches.
        args = {"user_id": "u1"}
        assert _filtered_args(args, ["nonexistent"]) == {"nonexistent": None}


class TestMatchAction:
    """End-to-end action-record matching honoring compare_args."""

    def test_name_mismatch_fails(self):
        ref = {"name": "rebook", "arguments": {}}
        pred = {"name": "cancel", "arguments": {}}
        assert _match_action(ref, pred) is False

    def test_no_compare_args_compares_all_args(self):
        # Eva records (no compare_args field) → strict full-dict equality.
        ref = {"name": "rebook", "arguments": {"user_id": "u1", "from": "JFK"}}
        pred_match = {"name": "rebook", "arguments": {"user_id": "u1", "from": "JFK"}}
        pred_diff = {"name": "rebook", "arguments": {"user_id": "u1", "from": "LAX"}}
        assert _match_action(ref, pred_match) is True
        assert _match_action(ref, pred_diff) is False

    def test_compare_args_none_explicit_compares_all(self):
        # Tau2 stores compare_args=None explicitly; same semantics as missing.
        ref = {
            "name": "rebook",
            "arguments": {"user_id": "u1", "from": "JFK"},
            "compare_args": None,
        }
        pred = {"name": "rebook", "arguments": {"user_id": "u1", "from": "JFK"}}
        assert _match_action(ref, pred) is True

    def test_compare_args_empty_list_matches_any_args(self):
        # transfer_to_human_agents pattern — name-only, ignore free-text summary.
        ref = {
            "name": "transfer_to_human_agents",
            "arguments": {"summary": "user wants X"},
            "compare_args": [],
        }
        pred_a = {
            "name": "transfer_to_human_agents",
            "arguments": {"summary": "completely different summary"},
        }
        pred_b = {"name": "transfer_to_human_agents", "arguments": {}}
        assert _match_action(ref, pred_a) is True
        assert _match_action(ref, pred_b) is True

    def test_compare_args_subset_ignores_other_args(self):
        ref = {
            "name": "get_user_details",
            "arguments": {"user_id": "u1"},
            "compare_args": ["user_id"],
        }
        pred_match = {
            "name": "get_user_details",
            "arguments": {"user_id": "u1", "extra_arg": "ignored"},
        }
        pred_diff = {"name": "get_user_details", "arguments": {"user_id": "u2"}}
        assert _match_action(ref, pred_match) is True
        assert _match_action(ref, pred_diff) is False

    def test_falsy_collapse_pitfall_does_not_occur(self):
        # Regression guard: [] must NOT silently behave like None (compare-all).
        # If a future refactor introduces `compare_args or "all"`, this test fails.
        ref = {
            "name": "transfer_to_human_agents",
            "arguments": {"summary": "A"},
            "compare_args": [],
        }
        pred = {
            "name": "transfer_to_human_agents",
            "arguments": {"summary": "B"},  # different summary
        }
        assert _match_action(ref, pred) is True, (
            "compare_args=[] must mean name-only match; equality of empty filtered "
            "dicts is the canonical path. If you see this fail, check whether a "
            "falsy-collapse bug was introduced in _filtered_args."
        )

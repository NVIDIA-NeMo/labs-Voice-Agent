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

"""Tests for the tau2 telecom Pydantic data model port.

Verifies:

1. ``TelecomDB`` validates the imported ``db.json`` cleanly.
2. ``TelecomUserDB`` validates the imported ``user_db.json`` cleanly AND
   materializes the ``surroundings`` block + ``signal_strength`` per-network
   defaults (the whole reason for the Pydantic round-trip in
   ``prepare_telecom.py``).
3. ``extra="forbid"`` rejects unknown fields at validation time.
4. Specific enum values that the predicate code relies on are present.

These tests run against the **imported** JSON files (the output of
``prepare_telecom.py``), not raw upstream TOML — that way they fail
cleanly if a re-import accidentally regresses the defaults-materialization
step.
"""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemo_voice_agent.evaluation import get_eval_data_root
from nemo_voice_agent.evaluation.tools.tau2_telecom_params import (
    NetworkStatus,
    NetworkTechnology,
    SignalStrength,
    SimStatus,
    TelecomDB,
    TelecomUserDB,
    UserSurroundings,
    to_dashed_us_phone,
)


def _telecom_data_dir() -> Path:
    return get_eval_data_root() / "tau2_telecom"


# ---------------------------------------------------------------------------
# 1. Imported JSON validates cleanly through both DB models
# ---------------------------------------------------------------------------


def test_db_json_validates_via_TelecomDB():
    """The agent-side ``db.json`` produced by prepare_telecom.py validates
    cleanly through ``TelecomDB`` — proves the model matches upstream's
    actual data shape (not just the documented one)."""
    with (_telecom_data_dir() / "db.json").open() as f:
        raw = json.load(f)
    db = TelecomDB.model_validate(raw)
    # Spot-check non-empty top-level lists — fixtures should be populated.
    assert len(db.customers) > 0
    assert len(db.plans) > 0
    assert len(db.lines) > 0


def test_user_db_json_validates_via_TelecomUserDB():
    """The user-side ``user_db.json`` produced by prepare_telecom.py validates
    cleanly through ``TelecomUserDB``."""
    with (_telecom_data_dir() / "user_db.json").open() as f:
        raw = json.load(f)
    udb = TelecomUserDB.model_validate(raw)
    # device + surroundings both materialized
    assert udb.device.sim_card_status == SimStatus.ACTIVE
    assert udb.surroundings.is_abroad is False


# ---------------------------------------------------------------------------
# 2. Pydantic-default materialization (the load-bearing prepare_telecom step)
# ---------------------------------------------------------------------------


def test_user_db_json_has_surroundings_block():
    """Upstream's ``user_db.toml`` only carries the ``device`` block; the
    ``surroundings`` block is a Pydantic-default-factory construct that
    materializes only through model round-trip. Predicate / init function
    code reads ``surroundings.*`` fields directly via dict access, so this
    block MUST be present in the imported JSON — otherwise the entire
    predicate evaluation path KeyErrors at runtime."""
    with (_telecom_data_dir() / "user_db.json").open() as f:
        raw = json.load(f)
    assert "surroundings" in raw, (
        "user_db.json is missing the 'surroundings' block — the prepare_telecom "
        "script's Pydantic round-trip is broken or was bypassed."
    )
    surroundings = raw["surroundings"]
    # Defaults that downstream code depends on:
    assert surroundings["is_abroad"] is False
    assert surroundings["roaming_allowed"] is False
    assert surroundings["line_active"] is True
    # Per-network signal_strength default table (used by _run_speed_test):
    assert surroundings["signal_strength"] == {
        "2G": "poor",
        "3G": "fair",
        "4G": "good",
        "5G": "excellent",
    }


def test_user_db_json_has_app_statuses_with_messaging_and_browser():
    """``MockPhoneAttributes.app_statuses`` has a default_factory that seeds
    ``messaging`` + ``browser`` entries. The MMS-related predicates
    (``assert_can_send_mms``) depend on the ``messaging`` app's SMS
    permission being present."""
    with (_telecom_data_dir() / "user_db.json").open() as f:
        raw = json.load(f)
    app_statuses = raw["device"]["app_statuses"]
    assert "messaging" in app_statuses
    assert "browser" in app_statuses
    # SMS permission default for messaging app (used by assert_can_send_mms).
    assert app_statuses["messaging"]["permissions"]["sms"] is True
    assert app_statuses["messaging"]["permissions"]["storage"] is True


# ---------------------------------------------------------------------------
# 3. extra="forbid" semantics
# ---------------------------------------------------------------------------


def test_TelecomUserDB_rejects_unknown_field():
    """``BaseModelNoExtra`` sets ``extra="forbid"`` — a typo'd field name in
    upstream data should raise at validate time, NOT silently shadow a real
    field. Catches accidental shape drift on future re-imports."""
    with pytest.raises(ValidationError):
        TelecomUserDB.model_validate({"device": {}, "surroundings": {}, "totally_made_up_field": 42})


def test_UserSurroundings_rejects_unknown_field():
    with pytest.raises(ValidationError):
        UserSurroundings.model_validate({"is_abroad": True, "not_a_real_field": "x"})


# ---------------------------------------------------------------------------
# 4. Enum values the predicate code depends on
# ---------------------------------------------------------------------------


def test_signal_strength_enum_values_match_upstream():
    """assert_internet_speed downstream checks expected_desc against
    SignalStrength values via case-insensitive string compare. These exact
    values are part of the upstream contract — a rename would break the
    upstream comparison."""
    assert SignalStrength.NONE.value == "none"
    assert SignalStrength.POOR.value == "poor"
    assert SignalStrength.FAIR.value == "fair"
    assert SignalStrength.GOOD.value == "good"
    assert SignalStrength.EXCELLENT.value == "excellent"


def test_network_status_enum_values_match_upstream():
    """_get_mobile_data_working downstream checks
    network_connection_status against NetworkStatus.NO_SERVICE explicitly."""
    assert NetworkStatus.NO_SERVICE.value == "no_service"
    assert NetworkStatus.CONNECTED.value == "connected"


def test_network_technology_enum_values_match_upstream():
    """_run_speed_test's network-technology-to-speed-range mapping keys
    off these exact string values (used as dict keys in signal_strength)."""
    assert NetworkTechnology.TWO_G.value == "2G"
    assert NetworkTechnology.FIVE_G.value == "5G"
    assert NetworkTechnology.NONE.value == "none"


# ---------------------------------------------------------------------------
# Phone numbers as speech recognition renders them
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("555-123-2002", "555-123-2002"),
        ("5551232002", "555-123-2002"),
        ("555 123 2002", "555-123-2002"),
        ("(555) 123-2002", "555-123-2002"),
        ("555.123.2002", "555-123-2002"),
        (" 555-123-2002 ", "555-123-2002"),
        ("+1 555 123 2002", "555-123-2002"),
        ("15551232002", "555-123-2002"),
        # Not a 10-digit US number: left unchanged so the lookup fails loudly.
        ("555123200", "555123200"),
        ("55512320021", "55512320021"),
        ("555-123-20O2", "555-123-20O2"),
        ("555-123-2002 x12", "555-123-2002 x12"),
        ("+55 512 320 02", "+55 512 320 02"),
        (
            "\uff15\uff15\uff15\uff11\uff12\uff13\uff12\uff10\uff10\uff12",
            "\uff15\uff15\uff15\uff11\uff12\uff13\uff12\uff10\uff10\uff12",
        ),
        (5551232002, "555-123-2002"),
        ("", ""),
    ],
)
def test_to_dashed_us_phone(given, expected):
    assert to_dashed_us_phone(given) == expected


def test_get_customer_by_phone_accepts_undashed_numbers():
    """Regression: the lookup compared strings exactly, so "5551232002" missed 555-123-2002.

    In a live voice evaluation the agent searched for the caller's number without the
    dashes in most telecom calls where it never found the account.
    """
    import nemo_voice_agent.evaluation.scenarios.data.tau2_telecom  # noqa: F401  (registers scenarios)
    from nemo_voice_agent.evaluation.scenarios import ALL_EVAL_SCENARIOS

    name = next(n for n in sorted(ALL_EVAL_SCENARIOS) if n.startswith("tau2_telecom__"))
    scenario = ALL_EVAL_SCENARIOS[name]()
    state = {"actions": [], "db": copy.deepcopy(scenario.db)}
    tool = scenario._build_tool_map(state)["get_customer_by_phone"]
    for spoken in ("5551232002", "555 123 2002", "(555) 123-2002", "+1 555 123 2002"):
        result = tool.invoke(phone_number=spoken)
        assert result.get("customer_id") == "C1001", (spoken, result)
    # A line number (not the customer's primary contact) also resolves.
    result = tool.invoke(phone_number="5551232003")
    assert result.get("customer_id") == "C1001", result
    result = tool.invoke(phone_number="555123200")
    assert result.get("status") == "error", result
    assert state["actions"] == []  # a read never records an action

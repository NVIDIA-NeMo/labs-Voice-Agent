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
        TelecomUserDB.model_validate(
            {"device": {}, "surroundings": {}, "totally_made_up_field": 42}
        )


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

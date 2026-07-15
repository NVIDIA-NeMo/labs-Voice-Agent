# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for lightweight logging utilities."""

from pathlib import Path

import nemo_voice_agent.utils.misc as misc
from nemo_voice_agent.utils.misc import FileLogger


def test_filelogger_level_helpers_and_callable_write_expected_prefixes(tmp_path):
    """FileLogger helper methods write level prefixes and callable logs to the file."""
    log_path = tmp_path / "run.log"
    logger = FileLogger(str(log_path))

    logger.info("info message", include_caller=False)
    logger.warning("warn message", include_caller=False)
    logger.error("error message", include_caller=False)
    logger.debug("debug message", include_caller=False)
    logger("call message", include_caller=False)

    content = log_path.read_text()
    assert "[INFO]: info message" in content
    assert "[WARNING]: warn message" in content
    assert "[ERROR]: error message" in content
    assert "[DEBUG]: debug message" in content
    assert "call message" in content


def test_filelogger_includes_caller_location_by_default(tmp_path):
    """Default FileLogger output includes the calling filename and function name."""
    log_path = tmp_path / "run.log"
    logger = FileLogger(str(log_path))

    logger.info("with caller")

    content = log_path.read_text()
    assert Path(__file__).name in content
    assert "test_filelogger_includes_caller_location_by_default" in content


def test_setup_rotating_log_removes_existing_file_when_overwrite(monkeypatch, tmp_path):
    """setup_rotating_log removes an existing log before configuring logging when overwrite is set."""
    log_path = tmp_path / "server.log"
    log_path.write_text("old")
    calls = []

    def _fake_setup_logging(**kwargs):
        """Capture setup_logging arguments without touching loguru global handlers."""
        calls.append(kwargs)

    monkeypatch.setattr(misc, "setup_logging", _fake_setup_logging)

    misc.setup_rotating_log(str(log_path), create_new_log=True, overwrite_existing=True, log_level="INFO")

    assert not log_path.exists()
    assert calls == [{"log_file": str(log_path), "log_level": "INFO", "rotation": "1 day"}]


def test_setup_rotating_log_renames_existing_file_when_not_overwriting(monkeypatch, tmp_path):
    """setup_rotating_log renames an existing log when create_new_log is true without overwrite."""
    log_path = tmp_path / "server.log"
    log_path.write_text("old")
    calls = []

    class _FixedDatetime:
        """Small datetime stand-in with deterministic strftime output."""

        @staticmethod
        def now():
            """Return an object that formats as the expected timestamp."""
            return type("T", (), {"strftime": lambda self, fmt: "20260708_120000"})()

    def _fake_setup_logging(**kwargs):
        """Capture setup_logging arguments without touching loguru global handlers."""
        calls.append(kwargs)

    monkeypatch.setattr(misc, "setup_logging", _fake_setup_logging)
    monkeypatch.setattr(misc, "datetime", _FixedDatetime)

    misc.setup_rotating_log(str(log_path), create_new_log=True, overwrite_existing=False, rotation="10 MB")

    assert not log_path.exists()
    assert (tmp_path / "server.20260708_120000.log").read_text() == "old"
    assert calls == [{"log_file": str(log_path), "log_level": "DEBUG", "rotation": "10 MB"}]

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

from omegaconf import OmegaConf

import nemo_voice_agent.utils.misc as misc
from nemo_voice_agent.utils.misc import (
    REDACTED,
    FileLogger,
    cli_api_keys,
    redact_cli_secrets,
    redact_literals,
    redact_secrets,
    redact_value,
)


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


def test_redact_secrets_masks_credential_keys_and_leaves_interpolations_unresolved():
    """Credential values are masked at any depth; other values and interpolations are kept as written."""
    config = {
        "api_key": "fake-api-key",
        "model": "m",
        "nested": {"api_key": "fake-nested-key", "judge_API_KEY": "fake-judge-key", "base_url": "http://x"},
        "api_key_env_var": "NVIDIA_API_KEY",
        "empty": {"api_key": "", "other_api_key": None},
    }
    assert redact_secrets(config) == {
        "api_key": REDACTED,
        "model": "m",
        "nested": {"api_key": REDACTED, "judge_API_KEY": REDACTED, "base_url": "http://x"},
        "api_key_env_var": "NVIDIA_API_KEY",
        "empty": {"api_key": "", "other_api_key": None},
    }

    dict_config = OmegaConf.create({"llm": {"api_key": "fake-api-key", "top_p": "${missing.key}"}})
    assert redact_secrets(dict_config) == {"llm": {"api_key": REDACTED, "top_p": "${missing.key}"}}


def test_redact_cli_secrets_masks_every_api_key_flag_form():
    """``--api-key`` values are masked whether separated by spaces or ``=``, and in either spelling."""
    params = "--dtype bfloat16 --api-key sk-one --port 8000 --api_key=sk-two --api-key   sk-three"
    assert cli_api_keys(params) == ["sk-one", "sk-two", "sk-three"]
    assert redact_cli_secrets(params) == (
        f"--dtype bfloat16 --api-key {REDACTED} --port 8000 --api_key={REDACTED} --api-key   {REDACTED}"
    )
    assert redact_cli_secrets("--dtype bfloat16 --api-key-file keys.txt") == "--dtype bfloat16 --api-key-file keys.txt"


def test_redact_literals_masks_each_non_empty_secret():
    assert redact_literals("args: {'api_key': ['sk-one']} sk-one", ["sk-one", "", None]) == (
        f"args: {{'api_key': ['{REDACTED}']}} {REDACTED}"
    )


def test_redact_secrets_masks_api_key_flags_inside_string_values():
    """``vllm_server_params`` is a flag string, so its ``--api-key`` is masked by value, not by key name."""
    config = OmegaConf.create(
        {"llm": {"vllm_server_params": "--api-key sk-secret --port 8000", "args": ["--api-key=x"]}}
    )
    assert redact_secrets(config) == {
        "llm": {"vllm_server_params": f"--api-key {REDACTED} --port 8000", "args": [f"--api-key={REDACTED}"]}
    }


def test_redact_value_recurses_into_a_non_credential_mapping():
    """Override log lines pass a whole section as the value, so nested credentials must still be masked."""
    value = OmegaConf.create({"api_key": "sk-nested", "vllm_server_params": "--api-key sk-flag"})
    assert redact_value("params", value) == {"api_key": REDACTED, "vllm_server_params": f"--api-key {REDACTED}"}
    assert redact_value("judge_api_key", value) == REDACTED

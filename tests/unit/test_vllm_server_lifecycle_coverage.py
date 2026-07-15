# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fully mocked coverage for the local vLLM server lifecycle."""

from types import SimpleNamespace

import pytest

from nemo_voice_agent.pipecat.services.nemo import llm as llm_module
from nemo_voice_agent.pipecat.services.nemo.llm import VLLMService


class _Response:
    def __init__(self, status_code, model=None):
        self.status_code = status_code
        self.model = model

    def json(self):
        return {"data": [] if self.model is None else [{"id": self.model}]}


class _Socket:
    attempts = 0
    fail_first = False

    def __init__(self, *args):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def bind(self, address):
        type(self).attempts += 1
        if self.fail_first and self.attempts == 1:
            raise OSError("busy")


class _Process:
    pid = 123

    def __init__(self, *, poll_result=None):
        self.poll_result = poll_result
        self.terminated = 0

    def poll(self):
        return self.poll_result

    def communicate(self):
        return ("stdout", "stderr")

    def terminate(self):
        self.terminated += 1


def _service(device="cpu", max_wait=2):
    service = VLLMService.__new__(VLLMService)
    service._device = device
    service._vllm_server_max_wait_time = max_wait
    service._vllm_server_check_interval = 1
    return service


def test_existing_server_with_requested_model_is_reused(monkeypatch):
    service = _service()
    monkeypatch.setattr(llm_module.requests, "get", lambda *args, **kwargs: _Response(200, "model"))
    monkeypatch.setattr(
        llm_module.psutil,
        "net_connections",
        lambda **kwargs: [
            SimpleNamespace(laddr=SimpleNamespace(port=8123), status=llm_module.psutil.CONN_LISTEN, pid=77)
        ],
    )
    assert service._start_vllm_server("model", base_url="http://localhost:8123/v1") == (
        "http://localhost:8123/v1"
    )


def test_server_launch_builds_command_sets_device_and_waits_until_ready(monkeypatch):
    service = _service(device="cuda:2")
    responses = iter([_Response(503), _Response(200, "model")])
    monkeypatch.setattr(llm_module.requests, "get", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(llm_module.socket, "socket", _Socket)
    monkeypatch.setattr(llm_module.psutil, "process_iter", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm_module.psutil, "net_connections", lambda **kwargs: [])
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: None)
    process = _Process()
    popen = {}

    def launch(command, **kwargs):
        popen.update(command=command, kwargs=kwargs)
        return process

    monkeypatch.setattr(llm_module.subprocess, "Popen", launch)
    url = service._start_vllm_server(
        "model", "--port 8124 --tensor-parallel-size 1", "http://localhost:8123/v1"
    )
    assert url == "http://localhost:8124/v1"
    assert popen["command"] == ["vllm", "serve", "model", "--port", "8124", "--tensor-parallel-size", "1"]
    assert popen["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == "2"


def test_launch_handles_busy_port_invalid_port_timeout_and_cleanup(monkeypatch):
    service = _service(device="cpu", max_wait=0)
    _Socket.attempts, _Socket.fail_first = 0, True
    monkeypatch.setattr(llm_module.requests, "get", lambda *args, **kwargs: _Response(503))
    monkeypatch.setattr(llm_module.socket, "socket", _Socket)
    monkeypatch.setattr(llm_module.psutil, "process_iter", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm_module.psutil, "net_connections", lambda **kwargs: [])
    process = _Process()
    monkeypatch.setattr(llm_module.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(RuntimeError, match="failed to start within"):
        service._start_vllm_server("model", "--port invalid")
    assert _Socket.attempts == 2 and process.terminated == 2
    assert service._vllm_process is process
    service._stop_vllm_server()
    assert process.terminated == 3
    _Socket.fail_first = False


def test_launch_reports_early_process_exit_and_missing_executable(monkeypatch):
    service = _service(max_wait=1)
    monkeypatch.setattr(llm_module.requests, "get", lambda *args, **kwargs: _Response(503))
    monkeypatch.setattr(llm_module.socket, "socket", _Socket)
    monkeypatch.setattr(llm_module.psutil, "process_iter", lambda *args, **kwargs: [])
    monkeypatch.setattr(llm_module.psutil, "net_connections", lambda **kwargs: [])
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: None)
    process = _Process(poll_result=1)
    monkeypatch.setattr(llm_module.subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(RuntimeError, match="Failed to start vLLM server"):
        service._start_vllm_server("model")

    def missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(llm_module.subprocess, "Popen", missing)
    service._vllm_process = None
    with pytest.raises(RuntimeError, match="vLLM not found"):
        service._start_vllm_server("model")

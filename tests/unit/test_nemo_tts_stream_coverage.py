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

"""Unit coverage for BaseNemoTTSService request and audio streaming."""

import asyncio
from types import SimpleNamespace

import numpy as np
from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame

from nemo_voice_agent.pipecat.services.nemo.tts import BaseNemoTTSService


async def _collect(generator):
    return [frame async for frame in generator]


class _Queue:
    def __init__(self, service, outcome=None, error=None):
        self.service, self.outcome, self.error = service, outcome, error

    async def put(self, request):
        if self.error:
            raise self.error
        if request is None:
            return
        _, request_id = request
        await self.service._pending_requests[request_id].put(self.outcome)


def _service(outcome, **kwargs):
    service = BaseNemoTTSService.__new__(BaseNemoTTSService)
    service._name = "test-tts"
    service._think_tokens = kwargs.get("think_tokens")
    service._have_seen_think_tokens = False
    service._ignore_strings = kwargs.get("ignore_strings")
    service._pending_requests = {}
    service._audio_logger = kwargs.get("audio_logger")
    service._model_name = "fake"
    service._sample_rate = 16000
    service._chunk_size = 4
    service._tts_queue = _Queue(service, outcome=outcome, error=kwargs.get("queue_error"))

    async def noop(*args, **kw):
        return None

    service.start_ttfb_metrics = service.stop_ttfb_metrics = service.start_tts_usage_metrics = noop
    return service


def test_run_tts_streams_list_chunks_and_logs_audio():
    logger = SimpleNamespace(first_audio_timestamp=None, turns=[], logs=[])
    logger.increment_turn_index = lambda speaker: logger.turns.append(speaker)
    logger.get_time_from_start_of_session = lambda: 0.5
    logger.log_agent_audio = lambda **kwargs: logger.logs.append(kwargs)
    chunks = [np.array([0.5, -0.5], dtype=np.float32), b"\x01\x00" * 3, None]
    service = _service(("success", chunks), audio_logger=logger)
    frames = asyncio.run(_collect(service.run_tts("hello", "ctx")))
    assert isinstance(frames[0], TTSStartedFrame)
    assert isinstance(frames[-1], TTSStoppedFrame)
    assert sum(isinstance(frame, TTSAudioRawFrame) for frame in frames) >= 2
    assert logger.turns == ["agent"] and logger.logs[0]["text"] == "hello"
    assert service._pending_requests == {}


def test_run_tts_single_audio_errors_and_empty_results():
    single = _service(("success", np.array([1, 2], dtype=np.int32)))
    frames = asyncio.run(_collect(single.run_tts("single", "ctx")))
    assert any(isinstance(frame, TTSAudioRawFrame) for frame in frames)
    failed = _service(("error", RuntimeError("generation failed")))
    assert any(isinstance(frame, ErrorFrame) for frame in asyncio.run(_collect(failed.run_tts("bad", "ctx"))))
    empty = _service(("success", None))
    assert any(isinstance(frame, ErrorFrame) for frame in asyncio.run(_collect(empty.run_tts("empty", "ctx"))))
    broken = _service(None, queue_error=RuntimeError("queue failed"))
    assert any(isinstance(frame, ErrorFrame) for frame in asyncio.run(_collect(broken.run_tts("broken", "ctx"))))


def test_think_tokens_special_strings_and_byte_conversion():
    service = _service(("success", b"\x00\x00"), think_tokens=["<think>", "</think>"], ignore_strings={"DROP"})
    assert service._handle_think_tokens("before<think>secret") == "before"
    assert service._handle_think_tokens("still thinking") is None
    assert service._handle_think_tokens("done</think>answer") == "answer"
    assert service._handle_think_tokens("<think>x</think>visible") == "visible"
    assert service._handle_think_tokens("plain") == "plain"
    assert service._drop_special_tokens("aDROPb") == "ab"
    assert service._convert_to_bytes(bytearray(b"x")) == b"x"
    assert service._convert_to_bytes(np.array([1.0, -1.0]))
    assert service._convert_to_bytes(np.array([1, 2], dtype=np.int32))
    assert service._convert_to_bytes(memoryview(b"z")) == b"z"
    frames = asyncio.run(_collect(service.run_tts("<think>hidden", "ctx")))
    assert frames == [None]

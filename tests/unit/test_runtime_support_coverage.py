# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for model-independent runtime support utilities."""

import asyncio
import base64
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from nemo_voice_agent.pipecat.services.nemo import _espeak_gpl_shim as espeak_shim
from nemo_voice_agent.pipecat.services.nemo import _g2p_fallback as g2p
from nemo_voice_agent.pipecat.services.nemo.audio_logger import AudioLogger
from nemo_voice_agent.pipecat.services.nvidia_llm_traced import NvidiaLLMService


@pytest.mark.parametrize(
    "phone,expected",
    [("AH0", "ə"), ("AH1", "ˈʌ"), ("ER0", "əɹ"), ("ER2", "ˌɜɹ"), ("OW1", "ˈO"), ("B", "b"), ("???", "")],
)
def test_arpabet_mapping(phone, expected):
    assert g2p._arpabet_phone_to_misaki(phone) == expected


def test_g2p_fallback_and_singleton(monkeypatch):
    fallback = g2p.ApacheG2PFallback.__new__(g2p.ApacheG2PFallback)
    token = SimpleNamespace(text="hello")
    fallback._g2p = lambda text: ["HH", "AH0", "L", "OW1", "!"]
    assert fallback(token) == ("həlˈO", 2)
    fallback._g2p = lambda text: ["!", " "]
    assert fallback(token) == (None, None)
    fallback._g2p = lambda text: (_ for _ in ()).throw(RuntimeError("failed"))
    assert fallback(token) == (None, None)
    sentinel = object()
    monkeypatch.setattr(g2p, "_shared_fallback", None)
    monkeypatch.setattr(g2p, "ApacheG2PFallback", lambda: sentinel)
    assert g2p.get_shared_fallback() is sentinel
    assert g2p.get_shared_fallback() is sentinel


def test_espeak_shim_stubs():
    names = [
        "phonemizer",
        "phonemizer.backend",
        "phonemizer.backend.espeak",
        "phonemizer.backend.espeak.wrapper",
        "espeakng_loader",
    ]
    saved = {name: sys.modules.get(name) for name in names}
    for name in names:
        sys.modules.pop(name, None)
    try:
        espeak_shim.install()
        loader = sys.modules["espeakng_loader"]
        assert loader.get_library_path() is loader.get_data_path() is None
        wrapper = sys.modules["phonemizer.backend.espeak.wrapper"].EspeakWrapper
        assert wrapper.set_library(None) is wrapper.set_data_path(None) is None
        with pytest.raises(RuntimeError, match="intentionally excluded"):
            sys.modules["phonemizer.backend"].EspeakBackend("en")
        existing = sys.modules["phonemizer"]
        espeak_shim.install()
        assert sys.modules["phonemizer"] is existing
    finally:
        for name in names:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]


class _Stream:
    def __init__(self, values, error=None):
        self.values, self.error = values, error

    def __aiter__(self):
        async def iterate():
            for value in self.values:
                yield value
            if self.error:
                raise self.error

        return iterate()


async def _collect(stream):
    return [chunk async for chunk in stream]


def _traced(stream=None, error=None):
    service = NvidiaLLMService.__new__(NvidiaLLMService)
    service.build_chat_completion_params = lambda params: {"model": "m", "messages": [{"role": "user"}], "tools": [{}]}

    async def create(**kwargs):
        if error:
            raise error
        return stream

    service._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return service


def test_traced_llm_success_and_errors():
    traced = asyncio.run(_traced(_Stream(["one", "two"])).get_chat_completions(SimpleNamespace()))
    assert asyncio.run(_collect(traced)) == ["one", "two"]
    traced = asyncio.run(
        _traced(_Stream(["one"], RuntimeError("stream failed"))).get_chat_completions(SimpleNamespace())
    )
    with pytest.raises(RuntimeError, match="stream failed"):
        asyncio.run(_collect(traced))
    error = RuntimeError("dispatch failed")
    error.response = SimpleNamespace(status_code=503, text="down")
    with pytest.raises(RuntimeError, match="dispatch failed"):
        asyncio.run(_traced(error=error).get_chat_completions(SimpleNamespace()))


def test_audio_logger_full_session(tmp_path, monkeypatch):
    logger = AudioLogger(log_dir=tmp_path, session_id="session", user_audio_sample_rate=100, pre_roll_time_sec=0)
    logger.first_audio_timestamp = datetime.now() - timedelta(seconds=1)
    pcm = np.arange(200, dtype=np.int16).tobytes()
    logger.append_continuous_user_audio(pcm)
    assert logger._resample_audio(pcm, 100, 100).dtype == np.float32
    assert logger._resample_audio(np.array([1, 2], dtype=np.int16), 100, 100).shape == (2,)
    monkeypatch.setattr(
        "nemo_voice_agent.pipecat.services.nemo.audio_logger.librosa.resample",
        lambda audio, orig_sr, target_sr: audio[::2],
    )
    assert logger._resample_audio(np.array([0.1, 0.2]), 200, 100).shape == (1,)
    assert logger.increment_turn_index("user") == logger.increment_turn_index("user") == 1
    logger.turn_audio_buffer, logger.turn_transcription_buffer = [pcm], ["hello", " world"]
    logger.stage_turn_audio_and_transcription(datetime.now(), is_first_frame=True, additional_metadata={"model": "x"})
    logger.save_user_audio()
    assert logger.session_metadata["user_entries"][0]["model"] == "x"
    assert logger.increment_turn_index("agent") == 2
    logger.set_agent_turn_start_time()
    assert logger.log_agent_audio(np.array([0.1, -0.1], dtype=np.float32), "first", sample_rate=100)
    assert logger.log_agent_audio(b"\x01\x00" * 4, "second", sample_rate=100)
    logger.set_agent_cutoff_time(0.5)
    assert all(segment["cutoff_time"] == 0.5 for segment in logger.session_metadata["agent_entries"][0])
    logger._append_to_stereo_conversation(np.array([0.8]), "left", 0.0, 100)
    logger.finalize_session()
    assert logger._stereo_conversation_file.exists()


def test_audio_logger_disabled_and_empty(tmp_path):
    disabled = AudioLogger(log_dir=tmp_path, enabled=False)
    assert disabled.append_continuous_user_audio(b"x") is None
    assert disabled.stage_user_audio(datetime.now(), "x") is None
    assert disabled.log_agent_audio(b"x", "x") is None
    assert (
        disabled.set_agent_turn_start_time() is disabled.set_agent_cutoff_time() is disabled.finalize_session() is None
    )
    logger = AudioLogger(log_dir=tmp_path, session_id="empty")
    with pytest.raises(ValueError):
        logger.get_time_from_start_of_session()
    assert logger.save_user_audio() is logger.save_stereo_conversation() is None
    logger.turn_audio_buffer, logger.turn_transcription_buffer = [b"x"], ["x"]
    logger.clear_user_audio_buffer()
    logger.set_agent_cutoff_time(0.0)

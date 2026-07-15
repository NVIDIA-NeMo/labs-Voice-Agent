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

"""Unit coverage for config-driven STT, TTS, and LLM factories."""

import pytest
from omegaconf import OmegaConf

from nemo_voice_agent.pipecat.services.nemo import llm, stt, tts


def _fake(name):
    return lambda **kwargs: (name, kwargs)


def test_stt_factory_routes_all_backends(monkeypatch):
    monkeypatch.setattr(stt, "NemoSTTService", _fake("nemo"))
    monkeypatch.setattr(stt, "ResilientNvidiaSTTService", _fake("nvidia"))
    monkeypatch.setattr(stt, "NemotronASRService", _fake("nemotron"))
    nemo = stt.get_stt_service_from_config(OmegaConf.create({"type": "nemo", "model": "m", "device": "cpu"}), "log")
    assert nemo[0] == "nemo" and nemo[1]["audio_logger"] == "log"
    nvidia = stt.get_stt_service_from_config(OmegaConf.create({"type": "nvidia", "model": "m", "sample_rate": 8000}))
    assert nvidia[0] == "nvidia" and nvidia[1]["sample_rate"] == 8000
    nemotron = stt.get_stt_service_from_config(OmegaConf.create({"type": "nemotron", "generate_interruptions": True}))
    assert nemotron[0] == "nemotron" and nemotron[1]["generate_interruptions"] is True
    with pytest.raises(AssertionError, match="Invalid STT backend"):
        stt.get_stt_service_from_config(OmegaConf.create({"type": "bad"}))


@pytest.mark.parametrize(
    "model, expected",
    [("fastpitch-hifigan", "fastpitch"), ("magpie", "magpie"), ("kokoro", "kokoro")],
)
def test_nemo_tts_factory_models(monkeypatch, model, expected):
    monkeypatch.setattr(tts, "SimpleSegmentedTextAggregator", _fake("aggregator"))
    monkeypatch.setattr(tts, "NeMoFastPitchHiFiGANTTSService", _fake("fastpitch"))
    monkeypatch.setattr(tts, "MagpieTTSService", _fake("magpie"))
    monkeypatch.setattr(tts, "KokoroTTSService", _fake("kokoro"))
    result = tts.get_tts_service_from_config(
        OmegaConf.create({"type": "nemo", "model": model, "main_model_id": "main", "sub_model_id": "sub"}), "log"
    )
    assert result[0] == expected and result[1]["audio_logger"] == "log"


def test_remote_tts_factories_and_errors(monkeypatch):
    monkeypatch.setattr(tts, "SimpleSegmentedTextAggregator", _fake("aggregator"))
    monkeypatch.setattr(tts, "NvidiaTTSService", _fake("nvidia"))
    monkeypatch.setattr(tts, "ResilientNemotronTTSService", _fake("nemotron"))
    assert tts.get_tts_service_from_config({"type": "nvidia"})[0] == "nvidia"
    assert tts.get_tts_service_from_config({"type": "nemotron", "use_text_aggregator": False})[0] == "nemotron"
    with pytest.raises(ValueError, match="Model is required"):
        tts.get_tts_service_from_config({"type": "nemo"})
    with pytest.raises(ValueError, match="Invalid model"):
        tts.get_tts_service_from_config({"type": "nemo", "model": "bad"})
    with pytest.raises(AssertionError, match="Invalid TTS type"):
        tts.get_tts_service_from_config({"type": "bad"})


def test_llm_hf_vllm_and_nvidia_factories(monkeypatch):
    monkeypatch.setattr(llm, "HuggingFaceLLMService", _fake("hf"))
    monkeypatch.setattr(llm, "VLLMService", _fake("vllm"))
    monkeypatch.setattr(llm, "NvidiaLLMService", _fake("nvidia"))
    hf = llm.get_llm_service_from_config(
        OmegaConf.create(
            {"type": "hf", "model": "m", "device": "cpu", "dtype": "float32", "generation_kwargs": {"x": 1}}
        )
    )
    assert hf[0] == "hf" and hf[1]["generation_kwargs"] == {"x": 1}
    vllm_result = llm.get_llm_service_from_config(
        OmegaConf.create(
            {
                "type": "vllm",
                "model": "m",
                "dtype": "float16",
                "vllm_server_params": "--port 1",
                "vllm_generation_params": {"temperature": 0.1},
            }
        )
    )
    assert vllm_result[0] == "vllm" and vllm_result[1]["vllm_server_params"].startswith("--dtype float16")
    monkeypatch.setenv("NVIDIA_API_KEY", "key")
    nvidia = llm.get_llm_service_from_config(
        OmegaConf.create(
            {
                "type": "nvidia",
                "model": "m",
                "nvidia_generation_params": {"temperature": 0.2},
                "default_headers": {"x": "y"},
            }
        )
    )
    assert nvidia[0] == "nvidia" and nvidia[1]["api_key"] == "key"


def test_llm_auto_detection_and_validation(monkeypatch):
    monkeypatch.setattr(llm, "HuggingFaceLLMService", _fake("hf"))
    monkeypatch.setattr(llm, "VLLMService", _fake("vllm"))
    monkeypatch.setattr(llm, "vllmModelConfig", lambda *args, **kwargs: object())
    cfg = OmegaConf.create({"type": "auto", "model": "m", "dtype": "float16"})
    assert llm.get_llm_service_from_config(cfg)[0] == "vllm"
    monkeypatch.setattr(
        llm, "vllmModelConfig", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unsupported"))
    )
    cfg = OmegaConf.create({"type": "auto", "model": "m", "device": "cpu", "dtype": "float32"})
    assert llm.get_llm_service_from_config(cfg)[0] == "hf"
    with pytest.raises(ValueError, match="Model name is required"):
        llm.get_llm_service_from_config(OmegaConf.create({"type": "auto"}))
    with pytest.raises(ValueError, match="Invalid LLM backend"):
        llm.get_llm_service_from_config(OmegaConf.create({"type": "bad"}))
    with pytest.raises(ValueError, match="extra must be a dictionary"):
        llm.get_llm_service_from_config(
            OmegaConf.create({"type": "vllm", "dtype": "float16", "vllm_generation_params": {"extra": "bad"}})
        )
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="NVIDIA_API_KEY is required"):
        llm.get_llm_service_from_config(OmegaConf.create({"type": "nvidia"}))

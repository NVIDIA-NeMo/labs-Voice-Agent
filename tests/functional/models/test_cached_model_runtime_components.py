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

"""Functional coverage for runtime components backed by cached local models."""

import asyncio
import gc
import os
from pathlib import Path

import numpy as np
import pytest


pytestmark = [pytest.mark.functional, pytest.mark.gpu]

REALTIME_EOU_ASR_MODEL = "nvidia/parakeet_realtime_eou_120m-v1"
SMALL_PARAKEET_ASR_MODEL = "nvidia/parakeet-tdt_ctc-110m"
DIAR_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"
KOKORO_MODEL = "hexgrad/Kokoro-82M"
MAGPIE_MODEL = "nvidia/magpie_tts_multilingual_357m"
FASTPITCH_MODEL = "nvidia/tts_en_fastpitch"
HIFIGAN_MODEL = "nvidia/tts_hifigan"
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _require_cached_model(model_id: str) -> Path:
    """Return a cached model directory or skip when it is unavailable offline."""
    hf_cache_name = f"models--{model_id.replace('/', '--')}"
    nemo_cache_names = [
        model_id.replace("/", "--"),
        model_id.replace("/", "_"),
        model_id.rsplit("/", maxsplit=1)[-1],
    ]
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    nemo_home = Path(os.environ.get("NEMO_HOME", Path.home() / ".cache" / "torch" / "NeMo"))
    candidates = [hf_home / hf_cache_name, hf_home / "hub" / hf_cache_name]
    candidates.extend(nemo_home / cache_name for cache_name in nemo_cache_names)

    for candidate in candidates:
        snapshots = candidate / "snapshots"
        if snapshots.is_dir() and any(snapshots.iterdir()):
            return candidate
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate

    pytest.skip(f"{model_id} is not available in the offline model caches under {hf_home} or {nemo_home}")


def _require_cached_model_weights(model_id: str) -> Path:
    """Require cached Hugging Face checkpoint weights before loading a local CausalLM."""
    model_root = _require_cached_model(model_id)
    snapshots = model_root / "snapshots"
    search_roots = [snapshot for snapshot in snapshots.iterdir() if snapshot.is_dir()] if snapshots.is_dir() else []
    if not search_roots:
        search_roots = [model_root]

    weight_patterns = (
        "model.safetensors",
        "model.safetensors.index.json",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "pytorch_model-*.bin",
    )
    for search_root in search_roots:
        for pattern in weight_patterns:
            if any(search_root.rglob(pattern)):
                return model_root

    pytest.skip(f"{model_id} is cached under {model_root}, but no checkpoint weight files are present")


def _require_cuda() -> None:
    """Skip model-backed functional tests when no CUDA device is visible."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for cached model functional tests")


def _cleanup_cuda() -> None:
    """Release model references and cached CUDA memory between functional tests."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return


def _pcm_silence(seconds: float, sample_rate: int = 16000) -> bytes:
    """Build a short mono int16 PCM silence buffer."""
    return np.zeros(int(seconds * sample_rate), dtype=np.int16).tobytes()


def test_realtime_eou_asr_model_streams_silence_from_cache():
    """NemoStreamingASRService loads the repo's realtime EOU ASR model and runs one streaming step."""
    _require_cuda()
    _require_cached_model(REALTIME_EOU_ASR_MODEL)

    from nemo_voice_agent.pipecat.services.nemo.streaming_asr import ASRResult, NemoStreamingASRService

    service = NemoStreamingASRService(
        model=REALTIME_EOU_ASR_MODEL,
        device="cuda:0",
        decoder_type="rnnt",
        ignore_eou_eob=True,
    )
    try:
        result = service.transcribe(_pcm_silence(1.0), stream_id="functional-asr")

        assert isinstance(result, ASRResult)
        assert isinstance(result.text, str)
        assert result.processing_time is not None
        assert result.is_final is False
    finally:
        del service
        _cleanup_cuda()


def test_small_parakeet_asr_model_loads_through_streaming_wrapper_when_compatible():
    """A small cached Parakeet ASR fallback can exercise the streaming ASR wrapper when compatible."""
    _require_cuda()
    _require_cached_model(SMALL_PARAKEET_ASR_MODEL)

    from nemo_voice_agent.pipecat.services.nemo.streaming_asr import NemoStreamingASRService

    try:
        service = NemoStreamingASRService(
            model=SMALL_PARAKEET_ASR_MODEL,
            device="cuda:0",
            decoder_type=None,
            ignore_eou_eob=True,
        )
    except (AttributeError, ValueError) as exc:
        pytest.skip(f"{SMALL_PARAKEET_ASR_MODEL} is cached but not compatible with the streaming wrapper: {exc}")

    try:
        result = service.transcribe(_pcm_silence(1.0), stream_id="functional-small-asr")

        assert isinstance(result.text, str)
        assert result.eou_prob is None
        assert result.eob_prob is None
    finally:
        del service
        _cleanup_cuda()


def test_streaming_diarization_model_returns_speaker_probabilities_from_cache():
    """NeMoStreamingDiarService loads Sortformer diarization and returns frame speaker probabilities."""
    _require_cuda()
    _require_cached_model(DIAR_MODEL)

    from nemo_voice_agent.pipecat.services.nemo.streaming_diar import DiarizationConfig, NeMoStreamingDiarService

    service = NeMoStreamingDiarService(
        cfg=DiarizationConfig(device="cuda"),
        model=DIAR_MODEL,
        sample_rate=16000,
    )
    try:
        diar_result = service.diarize(_pcm_silence(1.0), stream_id="functional-diar")

        assert diar_result.shape == (service.chunk_size, service.max_num_speakers)
        assert np.isfinite(diar_result).all()
    finally:
        del service
        _cleanup_cuda()


def test_kokoro_tts_model_generates_audio_from_cache():
    """KokoroTTSService loads the cached Kokoro model and generates at least one audio chunk."""
    _require_cuda()
    _require_cached_model(KOKORO_MODEL)
    pytest.importorskip("kokoro")

    from nemo_voice_agent.pipecat.services.nemo.tts import KokoroTTSService

    service = KokoroTTSService(
        model=KOKORO_MODEL,
        device="cuda",
        download_all=False,
        cache_models=False,
    )
    try:
        chunks = list(service._generate_audio("Hello from the functional test."))

        assert chunks
        assert all(isinstance(chunk, np.ndarray) for chunk in chunks)
        assert sum(chunk.size for chunk in chunks) > 0
    finally:
        del service
        _cleanup_cuda()


def test_fastpitch_hifigan_tts_models_generate_audio_from_cache():
    """FastPitch and HiFiGAN cached NeMo TTS models generate PCM-ready audio."""
    _require_cuda()
    _require_cached_model(FASTPITCH_MODEL)
    _require_cached_model(HIFIGAN_MODEL)

    from nemo_voice_agent.pipecat.services.nemo.tts import NeMoFastPitchHiFiGANTTSService

    service = NeMoFastPitchHiFiGANTTSService(
        fastpitch_model=FASTPITCH_MODEL,
        hifigan_model=HIFIGAN_MODEL,
        device="cuda",
    )
    try:
        chunks = list(service._generate_audio("Hello from FastPitch."))

        assert len(chunks) == 1
        assert chunks[0].ndim == 1
        assert chunks[0].size > 0
    finally:
        del service
        _cleanup_cuda()


def test_magpie_tts_model_generates_audio_from_cache():
    """MagpieTTSService loads the cached Magpie model and synthesizes a short utterance."""
    _require_cuda()
    _require_cached_model(MAGPIE_MODEL)

    from nemo_voice_agent.pipecat.services.nemo.tts import MagpieTTSService

    service = MagpieTTSService(model=MAGPIE_MODEL, device="cuda")
    try:
        chunks = list(service._generate_audio("Hello from Magpie."))

        assert len(chunks) == 1
        assert chunks[0].ndim == 1
        assert chunks[0].size > 0
    finally:
        del service
        _cleanup_cuda()


def test_huggingface_llm_local_service_streams_from_cached_qwen_model():
    """HuggingFaceLLMLocalService loads cached Qwen and streams a short chat completion."""
    _require_cuda()
    _require_cached_model_weights(QWEN_MODEL)

    from nemo_voice_agent.pipecat.services.nemo.llm import HuggingFaceLLMLocalService

    service = HuggingFaceLLMLocalService(
        model=QWEN_MODEL,
        device="cuda:0",
        dtype="bfloat16",
        generation_kwargs={"max_new_tokens": 4, "do_sample": False},
    )

    async def _collect_chunks(local_service) -> list[str]:
        """Collect a bounded number of streamed text chunks."""
        chunks = []
        messages = [{"role": "user", "content": "Reply with one word."}]
        async for chunk in local_service.generate_stream(messages):
            text = chunk.choices[0].delta.content
            if text:
                chunks.append(text)
        return chunks

    try:
        chunks = asyncio.run(_collect_chunks(service))

        assert chunks
        assert all(isinstance(chunk, str) for chunk in chunks)
    finally:
        del service
        _cleanup_cuda()

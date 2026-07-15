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
from huggingface_hub import snapshot_download


pytestmark = [pytest.mark.functional, pytest.mark.gpu]

REALTIME_EOU_ASR_MODEL = "nvidia/parakeet_realtime_eou_120m-v1"
SMALL_PARAKEET_ASR_MODEL = "nvidia/parakeet-tdt_ctc-110m"
DIAR_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"
KOKORO_MODEL = "hexgrad/Kokoro-82M"
MAGPIE_MODEL = "nvidia/magpie_tts_multilingual_357m"
FASTPITCH_MODEL = "nvidia/tts_en_fastpitch"
HIFIGAN_MODEL = "nvidia/tts_hifigan"
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ALL_MODEL_IDS = (
    REALTIME_EOU_ASR_MODEL,
    SMALL_PARAKEET_ASR_MODEL,
    DIAR_MODEL,
    KOKORO_MODEL,
    FASTPITCH_MODEL,
    HIFIGAN_MODEL,
    MAGPIE_MODEL,
    QWEN_MODEL,
)


def _cached_snapshot(model_id: str) -> Path:
    """Resolve a Hugging Face model from the mounted offline cache."""
    snapshot = Path(snapshot_download(model_id, local_files_only=True))
    print(f"[model-cache] {model_id} snapshot={snapshot}")
    return snapshot


def _cached_nemo_model_arg(model_id: str) -> str:
    """Return a local .nemo artifact when present, otherwise the cached snapshot directory."""
    snapshot = _cached_snapshot(model_id)
    nemo_files = sorted(snapshot.glob("*.nemo"))
    if nemo_files:
        print(f"[model-cache] {model_id} using nemo artifact={nemo_files[0]}")
        return str(nemo_files[0])
    print(f"[model-cache] {model_id} using snapshot directory={snapshot}")
    return str(snapshot)


def _snapshot_file_summary(snapshot: Path) -> dict[str, object]:
    """Return a compact summary of files relevant to HF/NeMo model loading."""
    patterns = {
        "nemo": "*.nemo",
        "config": "config.json",
        "safetensors_index": "model.safetensors.index.json",
        "safetensors": "model*.safetensors",
        "pytorch_bin": "pytorch_model*.bin",
        "tokenizer": "tokenizer*",
    }
    summary = {}
    for name, pattern in patterns.items():
        matches = sorted(snapshot.glob(pattern))
        summary[name] = {
            "count": len(matches),
            "sample": [match.name for match in matches[:3]],
        }
    return summary


def test_cached_hf_snapshots_are_resolvable_and_report_contents():
    """HF cache diagnostics show each model's resolved snapshot and key artifact files."""
    print(f"[model-cache] HF_HOME={os.environ.get('HF_HOME')}")
    print(f"[model-cache] HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}")
    print(f"[model-cache] NEMO_HOME={os.environ.get('NEMO_HOME')}")
    print(f"[model-cache] TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')}")
    print(f"[model-cache] HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}")

    for model_id in ALL_MODEL_IDS:
        snapshot = _cached_snapshot(model_id)
        summary = _snapshot_file_summary(snapshot)
        print(f"[model-cache] {model_id} files={summary}")
        assert snapshot.exists()


def test_cached_qwen_snapshot_loads_with_transformers_auto_classes():
    """Transformers can instantiate the cached Qwen tokenizer and causal LM directly from the snapshot."""
    _require_cuda()
    transformers = pytest.importorskip("transformers")
    snapshot = str(_cached_snapshot(QWEN_MODEL))

    tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        snapshot,
        device_map="cuda:0",
        dtype="bfloat16",
        trust_remote_code=True,
        local_files_only=True,
    )
    try:
        assert tokenizer is not None
        assert model is not None
        assert getattr(model.config, "_name_or_path", None)
    finally:
        del model
        del tokenizer
        _cleanup_cuda()


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

    from nemo_voice_agent.pipecat.services.nemo.streaming_asr import ASRResult, NemoStreamingASRService

    service = NemoStreamingASRService(
        model=_cached_nemo_model_arg(REALTIME_EOU_ASR_MODEL),
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

    from nemo_voice_agent.pipecat.services.nemo.streaming_asr import NemoStreamingASRService

    try:
        service = NemoStreamingASRService(
            model=_cached_nemo_model_arg(SMALL_PARAKEET_ASR_MODEL),
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

    from nemo_voice_agent.pipecat.services.nemo.streaming_diar import DiarizationConfig, NeMoStreamingDiarService

    service = NeMoStreamingDiarService(
        cfg=DiarizationConfig(device="cuda"),
        model=_cached_nemo_model_arg(DIAR_MODEL),
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
    pytest.importorskip("kokoro")

    from nemo_voice_agent.pipecat.services.nemo.tts import KokoroTTSService

    service = KokoroTTSService(
        model=str(_cached_snapshot(KOKORO_MODEL)),
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

    from nemo_voice_agent.pipecat.services.nemo.tts import NeMoFastPitchHiFiGANTTSService

    service = NeMoFastPitchHiFiGANTTSService(
        fastpitch_model=_cached_nemo_model_arg(FASTPITCH_MODEL),
        hifigan_model=_cached_nemo_model_arg(HIFIGAN_MODEL),
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

    from nemo_voice_agent.pipecat.services.nemo.tts import MagpieTTSService

    service = MagpieTTSService(model=_cached_nemo_model_arg(MAGPIE_MODEL), device="cuda")
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

    from nemo_voice_agent.pipecat.services.nemo.llm import HuggingFaceLLMLocalService

    service = HuggingFaceLLMLocalService(
        model=str(_cached_snapshot(QWEN_MODEL)),
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

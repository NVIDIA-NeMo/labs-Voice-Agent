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

"""Unit tests for audio conversion, noise chunking, and stream padding helpers."""

import asyncio

import numpy as np

from nemo_voice_agent.utils.audio import (
    AudioStream,
    NoiseConfig,
    NoiseGenerator,
    SOXRAudioResampler,
    SOXRAudioStreamResampler,
    audio_bytes_to_float32,
    audio_float32_to_bytes,
)


def test_pcm16_bytes_round_trip_clips_float32_audio():
    """Float32 samples are clipped to PCM-16 range and decoded back as normalized audio."""
    samples = np.array([-2.0, -1.0, 0.0, 0.5, 2.0], dtype=np.float32)

    encoded = audio_float32_to_bytes(samples)
    decoded = audio_bytes_to_float32(encoded)

    assert decoded[0] == decoded[1]
    assert decoded[2] == 0.0
    assert np.isclose(decoded[3], 0.5, atol=1e-4)
    assert decoded[4] <= 1.0


def test_noise_config_serializes_all_public_fields():
    """NoiseConfig.to_dict exposes the exact values passed to the stream layer."""
    config = NoiseConfig(
        noise_files=["noise.wav"],
        gain_db=-12.0,
        max_noise_duration=1.5,
        random_offset=False,
        random_white_noise=True,
        white_noise_db=-40.0,
    )

    assert config.to_dict() == {
        "noise_files": ["noise.wav"],
        "gain_db": -12.0,
        "max_noise_duration": 1.5,
        "random_offset": False,
        "random_white_noise": True,
        "white_noise_db": -40.0,
    }


def test_noise_generator_white_noise_respects_duration_and_db_scale(monkeypatch):
    """White-noise generation uses max_duration * sample_rate and applies dB attenuation."""
    monkeypatch.setattr(np.random, "uniform", lambda low, high, size: np.ones(size, dtype=np.float32))
    generator = NoiseGenerator(
        noise_audio_files=[],
        sample_rate=10,
        max_duration=0.5,
        random_offset=False,
        random_white_noise=True,
        white_noise_db=-20.0,
    )

    assert generator.noise_audio_data.dtype == np.float32
    assert generator.noise_audio_data.shape == (5,)
    assert np.allclose(generator.noise_audio_data, 0.1)


def test_noise_generator_chunks_wrap_and_repeat_without_file_io():
    """Chunk extraction wraps within the noise buffer and repeats when the request is larger."""
    generator = object.__new__(NoiseGenerator)
    generator.sample_rate = 4
    generator.noise_audio_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    generator.current_position = 2

    wrapped = generator.get_noise_chunk(0.5)
    repeated = generator.get_noise_chunk(1.0)

    assert np.allclose(wrapped, [0.3, 0.1])
    assert generator.current_position == 1
    assert np.allclose(repeated, [0.1, 0.2, 0.3, 0.1])
    assert generator.current_position == 1


def test_soxr_resampler_returns_original_bytes_when_rates_match():
    """The stateless resampler bypasses SOXR when no sample-rate conversion is needed."""
    audio = np.array([1, 2, 3], dtype=np.int16).tobytes()
    resampler = SOXRAudioResampler(16000, 16000)

    assert resampler.resample(audio) is audio


def test_stream_resampler_flushes_and_resets_after_timeout(monkeypatch):
    """The streaming resampler marks stale chunks as final and clears state after flushing."""
    calls = []

    class _FakeResampleStream:
        """Fake soxr stream that records resample and clear calls."""

        def resample_chunk(self, audio_data, last=False):
            """Capture the last flag and return the input data."""
            calls.append(("resample", audio_data.tolist(), last))
            return audio_data

        def clear(self):
            """Record a clear call."""
            calls.append(("clear",))

    resampler = object.__new__(SOXRAudioStreamResampler)
    resampler.in_sample_rate = 16000
    resampler.out_sample_rate = 24000
    resampler.quality = "VHQ"
    resampler.resampler = _FakeResampleStream()
    resampler._last_resample_time = 100.0
    monkeypatch.setattr("nemo_voice_agent.utils.audio.time.time", lambda: 101.0)

    output = resampler.resample(np.array([1, 2], dtype=np.int16).tobytes())

    assert output == np.array([1, 2], dtype=np.int16).tobytes()
    assert calls == [("resample", [1, 2], True), ("clear",)]
    assert resampler._last_resample_time is None


def test_audio_stream_output_chunk_pads_trims_and_mixes_noise():
    """Output chunks are normalized to the fixed byte length and can be noise-augmented."""
    stream = AudioStream(
        chunk_size_in_seconds=0.001,
        input_sample_rate=1000,
        output_sample_rate=1000,
        stream_resampler=False,
        noise_config=None,
    )
    stream.gain_db = 0.0
    short_audio = np.array([1000], dtype=np.int16).tobytes()
    long_noise = np.array([1000, 2000, 3000], dtype=np.int16).tobytes()

    output = stream.get_output_chunk(short_audio, noise_chunk=long_noise)

    assert len(output) == stream.output_chunk_bytes
    assert np.frombuffer(output, dtype=np.int16).tolist() == [1999]


def test_audio_stream_get_nowait_returns_silence_until_buffer_is_ready():
    """A no-wait read before enough buffered audio returns a silence chunk and has_speech False."""
    stream = AudioStream(
        chunk_size_in_seconds=0.001,
        input_sample_rate=1000,
        output_sample_rate=1000,
        stream_resampler=False,
        min_buffer_chunks=2,
        noise_config=None,
    )

    chunk, has_speech = asyncio.run(stream.get_nowait())

    assert chunk == b"\x00\x00"
    assert has_speech is False


def test_audio_stream_get_wait_returns_ready_buffered_audio():
    """When enough audio is buffered, get_wait returns a speech chunk and preserves overflow."""
    stream = AudioStream(
        chunk_size_in_seconds=0.001,
        input_sample_rate=1000,
        output_sample_rate=1000,
        stream_resampler=False,
        min_buffer_chunks=1,
        noise_config=None,
    )

    async def _read():
        """Buffer two chunks and read one output chunk."""
        await stream.put(np.array([10, 20], dtype=np.int16).tobytes())
        return await stream.get_wait(timeout=0.01)

    chunk, has_speech = asyncio.run(_read())

    assert np.frombuffer(chunk, dtype=np.int16).tolist() == [10]
    assert has_speech is True
    assert np.frombuffer(stream.output_buffer, dtype=np.int16).tolist() == [20]

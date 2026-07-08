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

"""Tests for ``sanitize_context_for_transport``.

Omni-model LLM contexts carry raw audio/image bytes inline; serializing them
overflows pipecat's WebSocket frame cap and drops the get_context_history
response. The sanitizer strips those blobs to dummy tags before transport
without mutating the live context.
"""

from nemo_voice_agent.pipecat.processors.frameworks.rtvi_actions import sanitize_context_for_transport


def test_audio_and_image_bytes_replaced_by_tags():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "audio", "audio": b"\x00\x01\x02large-blob"},
            ],
        },
        {"role": "assistant", "content": "hi there", "image": bytearray(b"\xff\xd8\xff")},
    ]
    out = sanitize_context_for_transport(messages)
    assert out[0]["content"][0] == {"type": "text", "text": "hello"}
    assert out[0]["content"][1]["audio"] == "<audio>"
    assert out[1]["content"] == "hi there"
    assert out[1]["image"] == "<image>"


def test_generic_bytes_get_binary_tag():
    out = sanitize_context_for_transport({"data": b"raw", "n": 5})
    assert out == {"data": "<binary>", "n": 5}


def test_input_audio_data_leaf_dropped_metadata_kept():
    # The real shape built by OpenAILLMContext.create_audio_message: the audio is
    # a base64 STRING under input_audio.data. Only that leaf is dropped; the
    # sibling "format" metadata is kept.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Audio follows"},
                {"type": "input_audio", "input_audio": {"data": "QUJD" * 100000, "format": "wav"}},
            ],
        }
    ]
    out = sanitize_context_for_transport(messages)
    part = out[0]["content"][1]
    assert part == {"type": "input_audio", "input_audio": {"data": "<audio>", "format": "wav"}}


def test_data_uri_stripped_but_real_url_kept():
    # A base64 data: URI is dropped; a plain http URL is small and kept.
    assert sanitize_context_for_transport({"image_url": {"url": "data:image/png;base64,AAAA" * 9999}}) == {
        "image_url": {"url": "<image>"}
    }
    kept = {"image_url": {"url": "https://example.com/cat.jpg", "detail": "auto"}}
    assert sanitize_context_for_transport(kept) == kept


def test_generic_data_key_outside_media_kept():
    # A "data" key NOT inside a media container is left alone (e.g. tool payload).
    payload = {"role": "tool", "data": {"rows": [1, 2, 3]}}
    assert sanitize_context_for_transport(payload) == payload


def test_all_openai_chat_content_parts():
    # Full OpenAI Chat Completions content-part surface: text/refusal + all small
    # metadata survive; only raw base64 leaves are dropped.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + "A" * 50000}},
                {"type": "input_audio", "input_audio": {"data": "B" * 50000, "format": "wav"}},
                {"type": "file", "file": {"filename": "r.pdf", "file_id": "file_9", "file_data": "C" * 50000}},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}, {"type": "refusal", "refusal": "I can't help"}],
            "audio": {"id": "audio_123", "data": "D" * 50000, "transcript": "hello"},
        },
    ]
    out = sanitize_context_for_transport(messages)
    parts = out[0]["content"]
    assert parts[0] == {"type": "text", "text": "look at this"}
    assert parts[1] == {"type": "image_url", "image_url": {"url": "<image>"}}
    assert parts[2] == {"type": "input_audio", "input_audio": {"data": "<audio>", "format": "wav"}}
    # filename + file_id kept; only file_data dropped.
    assert parts[3] == {"type": "file", "file": {"filename": "r.pdf", "file_id": "file_9", "file_data": "<file>"}}
    # Assistant text + refusal preserved; audio id + transcript kept, data dropped.
    assert out[1]["content"] == [{"type": "text", "text": "ok"}, {"type": "refusal", "refusal": "I can't help"}]
    assert out[1]["audio"] == {"id": "audio_123", "data": "<audio>", "transcript": "hello"}


def test_non_binary_payload_is_unchanged():
    messages = [{"role": "user", "content": "plain text"}, {"role": "assistant", "content": "reply"}]
    assert sanitize_context_for_transport(messages) == messages


def test_original_context_not_mutated():
    blob = b"\x00\x01\x02"
    messages = [{"role": "user", "audio": blob}]
    out = sanitize_context_for_transport(messages)
    # Live context keeps its bytes; only the returned copy is stripped.
    assert messages[0]["audio"] is blob
    assert out[0]["audio"] == "<audio>"


def test_result_is_json_serializable():
    import json

    messages = [{"role": "user", "content": [{"type": "audio", "audio": b"x" * 1000}]}]
    out = sanitize_context_for_transport(messages)
    # Must round-trip through JSON (no residual bytes) — this is the whole point.
    assert "<audio>" in json.dumps(out)

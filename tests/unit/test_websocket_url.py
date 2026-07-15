# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import pytest

from nemo_voice_agent.utils.websocket_url import (
    _normalize_websocket_host,
    _normalize_websocket_scheme,
    build_websocket_url,
)


class TestNormalizeScheme:
    def test_ws_accepted(self):
        assert _normalize_websocket_scheme("ws") == "ws"

    def test_wss_accepted(self):
        assert _normalize_websocket_scheme("wss") == "wss"

    def test_uppercase_normalized(self):
        assert _normalize_websocket_scheme("WS") == "ws"
        assert _normalize_websocket_scheme("WSS") == "wss"

    def test_whitespace_stripped(self):
        assert _normalize_websocket_scheme("  ws  ") == "ws"

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError, match="WEBSOCKET_SCHEME"):
            _normalize_websocket_scheme("http")

    def test_empty_scheme_raises(self):
        with pytest.raises(ValueError):
            _normalize_websocket_scheme("")


class TestNormalizeHost:
    def test_plain_hostname(self):
        assert _normalize_websocket_host("example.com") == "example.com"

    def test_plain_ip(self):
        assert _normalize_websocket_host("127.0.0.1") == "127.0.0.1"

    def test_whitespace_stripped(self):
        assert _normalize_websocket_host("  localhost  ") == "localhost"

    def test_full_url_stripped_to_hostname(self):
        assert _normalize_websocket_host("http://example.com") == "example.com"
        assert _normalize_websocket_host("https://example.com:8080/path") == "example.com"

    def test_empty_host_raises(self):
        with pytest.raises(ValueError, match="SERVER_PUBLIC_HOST"):
            _normalize_websocket_host("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            _normalize_websocket_host("   ")

    def test_url_without_hostname_raises(self):
        with pytest.raises(ValueError, match="SERVER_PUBLIC_HOST"):
            _normalize_websocket_host("http://")


class TestBuildWebsocketUrl:
    def test_basic_ipv4(self):
        assert build_websocket_url("127.0.0.1", 8765) == "ws://127.0.0.1:8765"

    def test_hostname(self):
        assert build_websocket_url("example.com", 8765) == "ws://example.com:8765"

    def test_wss_scheme(self):
        assert build_websocket_url("example.com", 8765, "wss") == "wss://example.com:8765"

    def test_ipv6_gets_brackets(self):
        assert build_websocket_url("::1", 8765) == "ws://[::1]:8765"

    def test_ipv6_already_bracketed(self):
        assert build_websocket_url("[::1]", 8765) == "ws://[::1]:8765"

    def test_full_url_as_host_stripped(self):
        assert build_websocket_url("http://myserver.example.com", 9000) == "ws://myserver.example.com:9000"

    def test_invalid_scheme_propagates(self):
        with pytest.raises(ValueError):
            build_websocket_url("localhost", 8765, "ftp")

    def test_empty_host_propagates(self):
        with pytest.raises(ValueError):
            build_websocket_url("", 8765)

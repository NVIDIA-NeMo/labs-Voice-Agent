# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from urllib.parse import urlsplit


def _normalize_websocket_scheme(scheme: str) -> str:
    scheme = scheme.strip().lower()
    if scheme not in {"ws", "wss"}:
        raise ValueError("WEBSOCKET_SCHEME must be either 'ws' or 'wss'")
    return scheme


def _normalize_websocket_host(host: str) -> str:
    host = host.strip()
    if not host:
        raise ValueError("SERVER_PUBLIC_HOST must not be empty")
    if "://" in host:
        parsed_host = urlsplit(host).hostname
        if not parsed_host:
            raise ValueError("SERVER_PUBLIC_HOST must include a host name")
        host = parsed_host
    return host


def build_websocket_url(host: str, port: int, scheme: str = "ws") -> str:
    """Build the client-facing WebSocket URL from trusted server configuration."""
    scheme = _normalize_websocket_scheme(scheme)
    host = _normalize_websocket_host(host)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{scheme}://{host}:{port}"

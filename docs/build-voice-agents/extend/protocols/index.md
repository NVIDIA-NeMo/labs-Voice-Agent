{/*
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/}

# Protocols

The browser client and evaluation bridge control a running pipeline through the WebSocket transport and
real-time voice inference (RTVI) messages.

## Choose an Integration Guide

Choose the client protocol for connection behavior or the RTVI control plane for actions sent after a client
connects.

- [Client protocol](client-protocol.md) explains the `/connect` handshake, WebSocket URL, and browser flow.
- [RTVI control plane](rtvi-actions.md) explains the actions used to reset services, update prompts, initialize
  evaluation state, and retrieve results.

## Related Reference

For message shapes and fields, use the [RTVI reference](../../../reference/runtime/rtvi-messages.md).

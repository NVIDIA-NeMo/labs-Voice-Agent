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

# Tools

Tools let supported language-model backends invoke application functions during a conversation. NeMo Voice
Agent supports direct functions registered by the server and tools owned by a pipeline component.

## Choose a guide

- [Tool calling](tool-calling.md) explains backend requirements, parser configuration, registration, and the
  runtime call lifecycle.
- [Write your own tools](custom-tools.md) walks through direct functions and component-owned tools.

Start with a direct function when the action is independent of a service. Use a component-owned tool when the
action changes service state, such as text-to-speech speed or voice selection.

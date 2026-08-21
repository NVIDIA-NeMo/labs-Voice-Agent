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

# Create Evaluations

Extend the harness by defining scenarios, exposing tools, or adding a complete domain. Keep scenario fixture
state, live tool behavior, and gold replay behavior aligned so deterministic scoring remains meaningful.

## Authoring Guides

Choose the guide that matches the evaluation surface you need to extend.

| Guide | Scope |
| --- | --- |
| [Authoring Scenarios](authoring-scenarios.md) | Define personas, tasks, prompts, expected state, and success signals. |
| [Authoring Tools](authoring-tools.md) | Implement read and write tools, action records, and domain registration. |
| [Adding a Domain](authoring-domains.md) | Package fixtures, register scenarios, and connect domain-specific scoring. |

Use the existing [domain guides](../domain-guides/index.md) as implementation examples, then validate the new
content with the documentation-consistency tests.

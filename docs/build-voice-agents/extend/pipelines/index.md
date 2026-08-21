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

# Pipelines

Pipeline extensions change how services are constructed, transform frames between stages, or replace the
shipped assembly.

## Choose an Extension Point

Use the narrowest guide that gives you control over the pipeline behavior you need.

| Guide | Scope |
| --- | --- |
| [Builder API](builders.md) | Component construction and the configuration consumed by each builder. |
| [Custom frame processors](custom-processor.md) | Add a focused transformation between existing stages. |
| [Custom pipelines](custom-pipeline.md) | Assemble a different `run_bot_websocket` pipeline. |

## Decision Guidance

Start with a processor for one transformation. Replace the complete pipeline only when stage ordering or
ownership must differ from the shipped server.

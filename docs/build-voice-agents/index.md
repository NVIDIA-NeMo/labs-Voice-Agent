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

# Build Voice Agents

Start with configuration when the shipped pipeline already supports the behavior you need. Add tools or
custom pipeline code only when configuration is not enough.

| Area | Use it to |
| --- | --- |
| [Configure](configure/index.md) | Select component configs, prompts, model-registry entries, and audio logging. |
| [Model serving](model-serving/index.md) | Run a vLLM server, configure plugins, or use hosted NVIDIA endpoints. |
| [Tools](tools/index.md) | Enable function calling and add direct or component-owned tools. |
| [Extend](extend/index.md) | Add processors, assemble a custom pipeline, or integrate through RTVI. |

## Recommended order

1. Confirm the current behavior in the [Quickstart](../get-started/quickstart.md).
2. Make the smallest configuration change that satisfies the requirement.
3. Add a tool for a discrete action or a processor for a new pipeline transformation.
4. Build a custom pipeline only when the shipped assembly cannot express the design.

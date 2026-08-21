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

## Choose an Area

Choose the smallest surface that supports the behavior you want to add.

| Area | Use It to |
| --- | --- |
| [Configure](configure/index.md) | Select component configs, prompts, model-registry entries, and audio logging. |
| [Model serving](model-serving/index.md) | Run a vLLM server, configure plugins, or use hosted NVIDIA endpoints. |
| [Tools](tools/index.md) | Enable function calling and add direct or component-owned tools. |
| [Extend](extend/index.md) | Add processors, assemble a custom pipeline, or integrate through RTVI. |

## Recommended Order

Use this order to move from a working example to the smallest maintainable customization.

1. Confirm the current behavior in the [Quickstart](../get-started/quickstart.md).
2. Make the smallest configuration change that satisfies the requirement.
3. Add a tool for a discrete action or a processor for a new pipeline transformation.
4. Build a custom pipeline only when the shipped assembly cannot express the design.

## Key Concepts

The extension surface you choose determines where the change belongs and how much pipeline behavior you
own:

- **Configuration** selects existing components and behavior without changing Python code.
- **Tools** let supported language models invoke application functions during a conversation.
- **Processors** transform frames at a focused point in the shipped pipeline.
- **Custom pipelines** replace the shipped assembly when stage ordering or ownership must change.

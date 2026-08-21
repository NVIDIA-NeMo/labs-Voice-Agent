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

# Choose Your Path

Use this page to select the documentation path that matches your goal. Each path assumes a source checkout
of the repository. The installation script creates the project virtual environment.

Before you begin, review [Prerequisites](prerequisites.md) for the system, software, and audio requirements.

## Choose by Goal

Match your goal to its starting page, then follow the related guide for the next part of the workflow:

| Goal | Start Here | Continue With |
| --- | --- | --- |
| Talk to the default agent | [Prerequisites](prerequisites.md) | [Installation](installation.md), then [Quickstart](quickstart.md) |
| Understand the pipeline first | [Overview](../about/overview.md) | [Architecture](../about/architecture.md) |
| Change models or behavior | [Configure](../build-voice-agents/configure/index.md) | [Model serving](../build-voice-agents/model-serving/index.md) or [Tools](../build-voice-agents/tools/index.md) |
| Add pipeline code | [Extend](../build-voice-agents/extend/index.md) | [Pipelines](../build-voice-agents/extend/pipelines/index.md) or [Protocols](../build-voice-agents/extend/protocols/index.md) |
| Measure an agent | [Evaluation overview](../evaluate/index.md) | [Run evaluations](../evaluate/run-evaluations/index.md) |

## First-Run Sequence

For the shortest path to a working browser session:

1. Verify the [prerequisites](prerequisites.md).
2. [Install](installation.md) the Python environment and browser dependencies.
3. Complete the [quickstart](quickstart.md) to launch vLLM, the agent server, and the client.

If a step fails, use [Troubleshooting](../troubleshooting/index.md).

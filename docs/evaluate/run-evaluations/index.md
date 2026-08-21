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

# About Running Evaluations

Run the simulated-user bot, the agent under test, and the bridge as three processes. The bridge exchanges
audio, captures artifacts, and passes each completed scenario to the scoring runner.

## Workflow Overview

Start with the quickstart, inspect the generated evidence, and then use the specialized guides for interrupted
runs or agents hosted outside this repository.

## Key Concepts

An evaluation run coordinates three independent processes: a simulated-user bot, the agent under test, and
the bridge. The bridge saves both session-level aggregates and per-scenario evidence, so reading results is
part of the workflow rather than a separate scoring step.

| Guide | Use It to |
| --- | --- |
| [Quickstart](quickstart.md) | Complete a first local evaluation from startup through saved results. |
| [Reading Results](results.md) | Interpret the run summary, scenario metrics, transcript, audio, and logs. |
| [Resume and Manage Long Runs](resume.md) | Continue an interrupted evaluation without rerunning completed scenarios. |
| [Evaluating an External Agent](external-agents.md) | Point the bridge at an agent outside this repository. |

Before comparing runs, review [Understand scoring](../understand-scoring/index.md) so the composite success
rate and per-signal denominators are interpreted correctly.

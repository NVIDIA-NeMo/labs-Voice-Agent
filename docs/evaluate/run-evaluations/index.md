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

# Run Evaluations

Run the simulated-user bot, the agent under test, and the bridge as three processes. The bridge exchanges
audio, captures artifacts, and passes each completed scenario to the scoring runner.

| Guide | Use it to |
| --- | --- |
| [Quickstart](quickstart.md) | Complete a first local evaluation from startup through saved results. |
| [Read results](results.md) | Interpret the run summary, scenario metrics, transcript, audio, and logs. |
| [Resume long runs](resume.md) | Continue an interrupted evaluation without rerunning completed scenarios. |
| [Evaluate an external agent](external-agents.md) | Point the bridge at an agent outside this repository. |

Before comparing runs, review [Understand scoring](../understand-scoring/index.md) so the composite success
rate and per-signal denominators are interpreted correctly.

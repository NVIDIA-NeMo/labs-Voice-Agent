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

# Understand Scoring

Evaluation results combine deterministic outcome checks, action records, an LLM judge, natural-language
assertions, and clean conversation exit. Each domain chooses which supported signals gate its composite
success verdict.

## Scoring Guides

Use these pages to identify the available scenario collections, interpret their success signals, and look up
the exact metrics fields written by the runner.

| Guide | Scope |
| --- | --- |
| [Benchmarks & Domains](benchmarks.md) | Explore the scenario collections and what each domain measures. |
| [Scoring Model](scoring.md) | Understand the six signals, domain whitelists, and aggregation rules. |
| [Metrics Dictionary](../../reference/evaluation/metrics.md) | Look up the exact fields written to evaluation result files. |

Read the scoring model before treating one percentage as a complete quality measure. A run can include useful
per-signal evidence even when a short or incomplete conversation fails the composite result.

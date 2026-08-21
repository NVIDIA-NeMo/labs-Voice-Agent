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

# Model Serving

Choose how the voice-agent server reaches its language and speech models. The shipped default expects a
separate local vLLM server, while the NVIDIA configuration uses hosted endpoints.

## Choose a serving path

Select the guide that matches where inference runs and whether your deployment needs vLLM plugins.

| Guide | Use it to |
| --- | --- |
| [Serve with vLLM](vllm.md) | Start and connect to an OpenAI-compatible local model server. |
| [vLLM plugins](vllm-plugins.md) | Configure tool-call, reasoning, and logits-processor plugins. |
| [NVIDIA NIM and Riva](nvidia-nim.md) | Connect speech and language stages to hosted NVIDIA services. |

## Related concept

Model serving describes deployment and connection choices. For backend behavior rather than deployment,
see [Language models](../../about/core-concepts/language-models/overview.md).

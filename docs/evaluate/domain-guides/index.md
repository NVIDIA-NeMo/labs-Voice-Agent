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

# Domain Guides

Each benchmark domain defines its own scenario data, tool surface, state model, and supported success signals.
Use these guides to understand domain-specific behavior before interpreting results or extending a fixture.

| Domain | Focus |
| --- | --- |
| [eva_airline](eva-airline.md) | Airline-service scenarios derived from the EVA dataset. |
| [tau2_airline](tau2-airline.md) | Reservation and flight-service tasks from tau2-bench. |
| [tau2_retail](tau2-retail.md) | Order, return, exchange, and account tasks from tau2-bench. |
| [tau2_telecom](tau2-telecom.md) | Dual-side telecom support with synchronized agent and user state. |

The [data provenance](data-provenance.md) page records the source and license of packaged evaluation fixtures.

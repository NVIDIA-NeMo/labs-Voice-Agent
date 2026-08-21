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

# Benchmarks and Domains

NeMo Labs Voice Agent ships a scenario catalog for the two-bot evaluation harness. Every scenario is a
registered Python class under `nemo_voice_agent/evaluation/scenarios/data/`; the runner selects them by name
or by domain. This page lists what is available, where each domain came from, and how to run it.

## Catalog

The catalog distinguishes benchmark-derived domains from smaller in-repository verification sets.

| Domain | Scenarios | Upstream | License | Gating Signals |
| --- | --- | --- | --- | --- |
| `eva_airline` | 50 | [ServiceNow/eva](https://github.com/ServiceNow/eva) 0.1.3 | MIT | `DB_STATE_MATCH`, `CLEAN_EXIT` |
| `tau2_airline` | 50 | [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) | MIT | `DB_STATE_MATCH`, `CLEAN_EXIT` |
| `tau2_retail` | 114 | tau2-bench | MIT | `DB_STATE_MATCH`, `CLEAN_EXIT`, plus `NL_ASSERTION` on the 40 tasks that carry one |
| `tau2_telecom` | 114 | tau2-bench | MIT | `DB_STATE_ASSERTION`, `CLEAN_EXIT` |
| `tau2_telecom_workflow` | 114 | tau2-bench | MIT | same as `tau2_telecom` |
| `restaurant` | 11 | in-repo | Apache-2.0 | `ACTION_MATCH`, `CLEAN_EXIT` |
| `customer_service` | 10 | in-repo | Apache-2.0 | `ACTION_MATCH`, `CLEAN_EXIT` |
| `qa` | 10 | in-repo | Apache-2.0 | `JUDGE_PASSED`, `CLEAN_EXIT` |
| Legacy (`fastbite`, `simple_qa_1`–`simple_qa_3`) | 4 | in-repository | Apache-2.0 | `ACTION_MATCH` or `JUDGE_PASSED`, plus `CLEAN_EXIT` |

Counts come from the live registry (`ALL_EVAL_SCENARIOS` in
`nemo_voice_agent/evaluation/scenarios/__init__.py`); reproduce them with `--list-domains` below. Signals not
in a scenario's whitelist are still computed and saved for diagnostics — they just do not gate the verdict.
Refer to [Scoring Signals](scoring.md) for how the six `SuccessSignal` members combine.

## List and Select Scenarios

`--domain` filters by the `<domain>__` name prefix, so domain names are exactly the prefixes printed by
`--list-domains`. Run these from the `evaluation/` directory.

```bash
cd evaluation

# Domains and their scenario counts
python run_evaluation.py --list-domains

# Every registered scenario name, grouped by domain
python run_evaluation.py --list

# One whole domain
python run_evaluation.py --domain tau2_retail

# Individual scenarios by name
python run_evaluation.py --scenarios eva_airline__1_1_3 tau2_airline__0
```

Neither `--list` nor `--list-domains` requires running bots. A real run requires both bot servers. Refer to
the [Evaluation Quickstart](../run-evaluations/quickstart.md) and
[Evaluation Command-Line Interface (CLI) Reference](../../reference/evaluation/eval-cli.md).

Naming conventions per domain:

| Domain | Name Pattern | Example |
| --- | --- | --- |
| `eva_airline` | EVA scenario ID with dots replaced by underscores | `eva_airline__1_1_3` |
| `tau2_airline`, `tau2_retail` | zero-based task index | `tau2_retail__0` |
| `tau2_telecom` | issue family plus fault list plus optional difficulty | `tau2_telecom__mobile_data_issue__airplane_mode_on__data_mode_off` |
| Verification sets | Hand-written slug | `restaurant__burger_classic` |

## Ported Benchmarks

Each domain has its own page with the task shape, tool surface, and known caveats.

- [eva_airline](../domain-guides/eva-airline.md) — 50 airline-support scenarios. Every scenario ships a gold
  `expected_scenario_db`, so database (DB)-state hash equality is the primary signal and no large language model (LLM)
  judge is required.
  Scenarios do not define a `reference_answer`. The agent prompt uses eva 0.1.3's complete
  `airline_agent.yaml` policy plus a short NeMo voice/runtime appendix. Agent tool surface: 15 eva tools plus
  `EndConversationTool`.
- [tau2_airline](../domain-guides/tau2-airline.md) — the 50 IDs in tau2-bench's `base` split. The agent prompt is
  the upstream `policy.md` verbatim plus a short voice-realization appendix, which keeps scores comparable to
  the published tau2 voice numbers. Agent tool surface: 14 tools plus `EndConversationTool`.
- [tau2_retail](../domain-guides/tau2-retail.md) — the 114 IDs in the retail `base` split. 40 of them carry
  `nl_assertions` (natural-language claims judged by the LLM judge); the rest are action/DB-only. Agent tool
  surface: 16 tools plus `EndConversationTool`.
- [tau2_telecom](../domain-guides/tau2-telecom.md) — the 114 IDs in the telecom `base` split, and the only dual-side
  domain: the simulated user gets 30 phone-control tools and its own user-side database alongside the agent's
  13 tools plus `EndConversationTool`. All 114 tasks carry `db_state_assertions`, which is why per-predicate
  scoring gates the verdict instead of whole-DB hash equality — telecom has an open solution space where
  several valid action sequences land in different databases.

### Telecom Policy Variants

Telecom is registered twice, mirroring upstream's `telecom` versus `telecom-workflow` split. The two
registrations share the same 114 upstream tasks, databases, reference actions, predicates, initialization
actions, and tool surface. The only difference is which tech-support policy file is concatenated into the
agent prompt:

| Registration | Base Class | `policy_variant` | Policy File Appended |
| --- | --- | --- | --- |
| `tau2_telecom__*` | `Tau2TelecomBaseScenario` | `manual` | `tech_support_manual.md` (long-form prose) |
| `tau2_telecom_workflow__*` | `Tau2TelecomWorkflowBaseScenario` | `workflow` | `tech_support_workflow.md` (procedural steps) |

Both keep `domain = "tau2_telecom"`, so tool-registry lookup, fixture paths, and predicate registries are
identical — only `scenario.name` and the rendered prompt differ. Because filtering is by name prefix,
`--domain tau2_telecom` runs only the manual variant and `--domain tau2_telecom_workflow` only the workflow
variant, which isolates the effect of the two policy phrasings.

## In-Repo Smoke Sets

These hand-authored domains require no external fixtures. Use them to verify a pipeline end to end before
running a ported benchmark.

| Domain | Scenarios | `max_duration` | What It Exercises |
| --- | --- | --- | --- |
| `restaurant` | 11 | 180 s (one at 120 s) | Menu-driven ordering with 3 agent tools; adds white noise at -20 dB to the audio |
| `customer_service` | 10 | 120 s | Ticket lookup and resolution with 4 agent tools |
| `qa` | 10 | 60 s | Single-question knowledge answers, 2 agent tools, judge-only scoring |

Because `qa` and the `simple_qa_*` legacy scenarios gate on `JUDGE_PASSED`, they need a reachable judge.
`--judge-url` defaults to `http://localhost:8000/v1/chat/completions`, so every run constructs a judge. If
you explicitly blank the URL (`--judge-url ""`), those scenarios yield `is_successful="N/A"`.

### Legacy Scenarios

The shorthand `simple_qa_1..3` refers to `simple_qa_1`, `simple_qa_2`, and `simple_qa_3`. These scenarios and
`fastbite` have no `<domain>__` prefix, so they are listed
separately by `--list-domains` and cannot be selected with `--domain`. Pass them to `--scenarios` by name.
They predate the domain convention and are kept as minimal regression cases; `fastbite` is a single
noisy-audio ordering scenario, and the `simple_qa_*` trio are one-shot question/answer checks.

## Fixtures and Provenance

Databases, task definitions, policy markdown, and split files are packaged inside the library at
`nemo_voice_agent/evaluation/data/`, one subdirectory per domain. `get_eval_data_root()` resolves that path
and honors the `EVAL_DATA_ROOT` environment variable as an override, so you can point a run at a modified
fixture tree without touching the package.

Two notes on how fixture size is handled:

- The tau2 airline database is sharded into one file per top-level table because the single upstream file
  exceeds the mirror's per-file cap. `load_db_artifact` reassembles it into a byte-identical dict, so DB
  hashes and gold replays are unchanged.
- Large databases are not inlined over the WebSocket. The bridge sends a `db_path` string in
  `shared_state_init` and the bot resolves it against the data root — inlining a multi-megabyte database
  exceeds Pipecat's frame cap and closes the connection.

Upstream pins, per-file source paths, and license text for every imported artifact live in
[Data provenance](../domain-guides/data-provenance.md), which mirrors the `README.md` kept next to the fixtures.

## Adding Your Own

Subclass the relevant domain base and decorate with `@register_eval_scenario`. Refer to
[Authoring Scenarios](../create-evaluations/authoring-scenarios.md). For a new benchmark with its own tools,
database, and scoring contract, follow [Authoring Domains](../create-evaluations/authoring-domains.md). Record
the upstream source and license in the fixture `README.md` before importing third-party data.

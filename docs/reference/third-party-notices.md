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

# Third-Party Notices

This page summarizes the licenses that apply to NeMo Labs Voice Agent and to the
third-party source code and data fixtures redistributed inside the repository.

**The authoritative document is `THIRD_PARTY_NOTICES.md` at the repository root.** It
reproduces the full license text of every redistributed upstream project. If this page
and that file ever disagree, the root file wins.

## Project license

| Item | Value |
| --- | --- |
| License | Apache License, Version 2.0 |
| Full text | `LICENSE` at the repository root |
| Package metadata | `license = "Apache-2.0"`, `license-files = ["LICENSE"]` in `pyproject.toml` |
| Copyright | NVIDIA CORPORATION & AFFILIATES |

Every `*.py` file in the repository (except `__init__.py`) carries an SPDX/Apache header
in its first lines. A copyright check runs on every pull request and fails without it —
see [Contributing](../contribute/index.md).

## Redistributed third-party code and data

`THIRD_PARTY_NOTICES.md` covers only upstream material whose **source or data is copied
into this repository**. Regular runtime dependencies (Pipecat, NeMo, PyTorch, vLLM, and
the rest of the `pyproject.toml` dependency tree) are installed from PyPI at install
time and keep their own licenses; they are not reproduced there.

| Upstream project | Owner | Version pinned | License | Redistributed as |
| --- | --- | --- | --- | --- |
| [eva](https://github.com/ServiceNow/eva) | ServiceNow | `0.1.3` | MIT | Evaluation scenarios, tools, and fixtures for the `eva_airline` domain |
| [tau2-bench](https://github.com/sierra-research/tau2-bench) | Sierra Research | tag `voice-user-sim-v1.0` (commit `17e07b1`) | MIT | Evaluation scenarios, tools, and fixtures for the `tau2_airline`, `tau2_retail`, and `tau2_telecom` domains |
| [Pipecat](https://github.com/pipecat-ai/pipecat) | Daily | — | BSD 2-Clause | The adapted RTVI browser client in `examples/generic_voice_agent/client/src/app.ts` |

### eva (MIT, ServiceNow)

Used for the [eva_airline](../evaluate/domains/eva-airline.md) evaluation domain (50
scenarios). Adapted code covers the DB-hash canonicalization module
(`nemo_voice_agent/evaluation/db_hash.py`), the airline tool and parameter definitions
(`eva_airline_tools.py`, `eva_airline_params.py`), and the scenario package under
`nemo_voice_agent/evaluation/scenarios/data/eva_airline/`. The scenario databases and
dataset index under `nemo_voice_agent/evaluation/data/eva_airline/` are copied verbatim.

### tau2-bench (MIT, Sierra Research)

Used for the [tau2_airline](../evaluate/domains/tau2-airline.md) (50 tasks),
[tau2_retail](../evaluate/domains/tau2-retail.md) (114 tasks), and
[tau2_telecom](../evaluate/domains/tau2-telecom.md) (114 tasks, plus a parallel
`tau2_telecom_workflow` registration over the same tasks) domains. Adapted code covers
the per-domain tool, parameter, predicate, and initialization-function modules under
`nemo_voice_agent/evaluation/tools/`, the shared write-tool base
(`_write_tool_base.py`), and the scenario packages under
`nemo_voice_agent/evaluation/scenarios/data/`. Databases, task definitions, split lists,
and agent policy markdown under `nemo_voice_agent/evaluation/data/tau2_*/` are copied
verbatim or format-converted (telecom's upstream `db.toml` / `user_db.toml` are
converted to JSON; airline's single large `db.json` is sharded per table).

### Pipecat (BSD 2-Clause, Daily)

The example browser client was adapted from Pipecat's RTVI client sample. The file
`examples/generic_voice_agent/client/src/app.ts` retains its original Daily copyright
line and `SPDX-License-Identifier: BSD 2-Clause License` header.

## Attribution conventions

| Convention | Where |
| --- | --- |
| Full upstream license text | `THIRD_PARTY_NOTICES.md` (repository root) |
| Per-artifact provenance: upstream URL, version, file-by-file mapping, whether verbatim or adapted | `nemo_voice_agent/evaluation/data/README.md` |
| Inline `# Adapted from <upstream-url>` comment near the top of each ported module | Every adapted Python file (46 files today) |
| Original upstream copyright header preserved in place | Adapted files that were copied largely intact, such as the browser client |

For the reader-facing version of the provenance table, see
[Fixture Data & Provenance](../evaluate/data-provenance.md).

## GPL avoidance in the dependency tree

Two dependency overrides in `pyproject.toml` keep the resolved dependency graph free of
copyleft packages that would be incompatible with the project's Apache-2.0 license:

| Override | Reason |
| --- | --- |
| Kokoro's `misaki[en]` extra is dropped; `misaki` is pinned directly | The extra pulls in `phonemizer-fork` (GPL-3.0) and `espeakng_loader`, which bundles a compiled GPL-3.0 `espeak-ng` binary. `spacy` and `num2words` — the permissively licensed parts of that extra needed for dictionary-based G2P — are added as direct dependencies instead. |
| `g2p-en`'s declared `distance` dependency (GPL-3.0) is stripped via `tool.uv.dependency-metadata` | It is a phantom dependency: `g2p-en` declares it in metadata but never imports it. |

Because `misaki.espeak` imports `phonemizer` and `espeakng_loader` unconditionally at
module load time, `nemo_voice_agent/pipecat/services/nemo/_espeak_gpl_shim.py` registers
no-op stand-ins in `sys.modules` so the import succeeds. Constructing the real espeak
backend then raises, which `kokoro.KPipeline` catches internally and falls back to
misaki's dictionary-only G2P. The shim does not shadow a real `phonemizer` if one is
already installed.

## Adding a new upstream source

When importing new evaluation data or porting upstream code, update all three of the
following in the same change:

1. Add a section to `THIRD_PARTY_NOTICES.md` with the project name, upstream URL, pinned
   version or commit, license identifier, the list of files it maps to, and the verbatim
   license text.
2. Add a matching section to `nemo_voice_agent/evaluation/data/README.md` describing each
   imported file and whether it is verbatim, filtered, or format-converted.
3. Add an inline `# Adapted from <upstream-url>` attribution to the top of every ported
   source file, below the SPDX header.

See [Importing evaluation data](../contribute/data-import.md) for the full import
workflow, including the scaffold generators under `scripts/`.

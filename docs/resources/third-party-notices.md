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

This page summarizes the licenses for NeMo Labs Voice Agent and the third-party source code and data
fixtures redistributed in the repository.

**The authoritative document is `THIRD_PARTY_NOTICES.md` at the repository root.** It reproduces the full
license text of every redistributed upstream project. If this summary and that file disagree, follow the
root file.

## Project License

Use the repository files and package metadata in this table as the authoritative project-license references:

| Item | Value |
| --- | --- |
| License | Apache License, Version 2.0 |
| Full text | `LICENSE` at the repository root |
| Package metadata | `license = "Apache-2.0"`, `license-files = ["LICENSE"]` in `pyproject.toml` |
| Copyright | NVIDIA CORPORATION & AFFILIATES |

Every `*.py` file in the repository, except `__init__.py`, carries an SPDX and Apache header in its first
lines. A copyright check runs on every pull request and fails without it. Refer to
[Contributing](contribute/index.md).

## Redistributed Third-Party Code and Data

`THIRD_PARTY_NOTICES.md` covers only upstream material whose **source or data is copied into this
repository**. Regular runtime dependencies, including Pipecat, NeMo, PyTorch, vLLM, and the rest of the
`pyproject.toml` dependency tree, are installed from PyPI. They retain their own licenses, which are not
reproduced in that file. The repository redistributes the following sources:

| Upstream Project | Owner | Pinned Version | License | Redistributed As |
| --- | --- | --- | --- | --- |
| [eva](https://github.com/ServiceNow/eva) | ServiceNow | `0.1.3` | MIT | Evaluation scenarios, tools, and fixtures for the `eva_airline` domain |
| [tau2-bench](https://github.com/sierra-research/tau2-bench) | Sierra Research | Tag `voice-user-sim-v1.0` (commit `17e07b1`) | MIT | Evaluation scenarios, tools, and fixtures for the `tau2_airline`, `tau2_retail`, and `tau2_telecom` domains |
| [Pipecat](https://github.com/pipecat-ai/pipecat) | Daily | — | BSD 2-Clause | The adapted Real-Time Voice Interface (RTVI) browser client in `examples/generic_voice_agent/client/src/app.ts` |

### eva (MIT, ServiceNow)

Used for the [eva_airline](../evaluate/domain-guides/eva-airline.md) evaluation domain (50 scenarios).
Adapted code covers the database-hash canonicalization module
(`nemo_voice_agent/evaluation/db_hash.py`), the airline tool and parameter definitions
(`eva_airline_tools.py`, `eva_airline_params.py`), and the scenario package under
`nemo_voice_agent/evaluation/scenarios/data/eva_airline/`. The scenario databases and
dataset index under `nemo_voice_agent/evaluation/data/eva_airline/` are copied verbatim. The upstream
`airline_agent.yaml` is also included there with only trailing whitespace normalized. Its `role` and
`instructions` provide the eva agent policy.

### tau2-bench (MIT, Sierra Research)

Used for the [tau2_airline](../evaluate/domain-guides/tau2-airline.md) (50 tasks),
[tau2_retail](../evaluate/domain-guides/tau2-retail.md) (114 tasks), and
[tau2_telecom](../evaluate/domain-guides/tau2-telecom.md) (114 tasks) domains. A parallel
`tau2_telecom_workflow` registration uses the same telecom tasks.

Adapted code covers the per-domain tool, parameter, predicate, and initialization-function modules under
`nemo_voice_agent/evaluation/tools/`. It also covers the shared write-tool base (`_write_tool_base.py`) and
the scenario packages under `nemo_voice_agent/evaluation/scenarios/data/`. Databases, task definitions,
split lists, and agent policy Markdown under `nemo_voice_agent/evaluation/data/tau2_*/` are copied verbatim
or format-converted. The telecom upstream `db.toml` and `user_db.toml` files are converted to JSON, and the
single large airline `db.json` is sharded by table.

### Pipecat (BSD 2-Clause, Daily)

The example browser client was adapted from Pipecat's RTVI client sample. The file
`examples/generic_voice_agent/client/src/app.ts` retains its original Daily copyright
line and `SPDX-License-Identifier: BSD 2-Clause License` header.

## Attribution Conventions

The repository records attribution at the following levels:

| Convention | Location |
| --- | --- |
| Full upstream license text | `THIRD_PARTY_NOTICES.md` (repository root) |
| Per-artifact provenance: upstream URL, version, file-by-file mapping, whether verbatim or adapted | `nemo_voice_agent/evaluation/data/README.md` |
| Inline `# Adapted from <upstream-url>` comment near the top of each ported module | Every adapted Python file (47 files today) |
| Original upstream copyright header preserved in place | Adapted files that were copied largely intact, such as the browser client |

For the reader-facing version of the provenance table, refer to
[Fixture Data & Provenance](../evaluate/domain-guides/data-provenance.md).

## GPL Avoidance in the Dependency Tree

Two dependency overrides in `pyproject.toml` keep the resolved dependency graph free of copyleft packages
that would be incompatible with the project's Apache-2.0 license:

| Override | Reason |
| --- | --- |
| Kokoro's `misaki[en]` extra is dropped, and `misaki` is pinned directly | The extra pulls in `phonemizer-fork` (GPL-3.0) and `espeakng_loader`, which bundles a compiled GPL-3.0 `espeak-ng` binary. The permissively licensed `spacy` and `num2words` packages provide the required dictionary-based grapheme-to-phoneme (G2P) support. They are added as direct dependencies instead. |
| `g2p-en`'s declared `distance` dependency (GPL-3.0) is stripped using `tool.uv.dependency-metadata` | This phantom dependency is declared in `g2p-en` metadata but is never imported. |

Because `misaki.espeak` imports `phonemizer` and `espeakng_loader` at module load time,
`nemo_voice_agent/pipecat/services/nemo/_espeak_gpl_shim.py` registers no-op substitutes in `sys.modules`.
This allows the import to succeed. Constructing the real espeak backend then raises an error, which
`kokoro.KPipeline` catches before falling back to misaki's dictionary-only G2P. The shim does not shadow a
real `phonemizer` installation.

## Adding a New Upstream Source

When importing new evaluation data or porting upstream code, update all three of the
following in the same change:

1. Add a section to `THIRD_PARTY_NOTICES.md` with the project name, upstream URL, pinned
   version or commit, license identifier, the list of files it maps to, and the verbatim
   license text.
2. Add a matching section to `nemo_voice_agent/evaluation/data/README.md` describing each
   imported file and whether it is verbatim, filtered, or format-converted.
3. Add an inline `# Adapted from <upstream-url>` attribution to the top of every ported
   source file, below the SPDX header.

Refer to [Importing Evaluation Data](contribute/data-import.md) for the full import workflow, including the
scaffold generators under `scripts/`.

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

# Adding a Domain

A *domain* in the NeMo Labs Voice Agent evaluation harness is a namespace binding a fixture set, a tool
registry bucket, a base scenario class, and a scoring configuration. See
[Authoring Scenarios](authoring-scenarios.md) for individual scenario classes and
[Authoring Tools](authoring-tools.md) for tool internals.

## Pieces of a domain

A complete domain can include the following fixture, runtime, and scoring components.

| Piece | Location | Required |
| --- | --- | --- |
| Fixtures (DB, policy, task index) | `nemo_voice_agent/evaluation/data/<domain>/` | Only if the domain has state |
| Tool module | `nemo_voice_agent/evaluation/tools/<domain>_tools.py` | Yes, if the agent calls tools |
| Base scenario + scaffolds | `nemo_voice_agent/evaluation/scenarios/data/<domain>/` | Yes |
| DB-state predicates | `nemo_voice_agent/evaluation/db_state_predicates.py` registry | Optional |
| Initialization functions | `nemo_voice_agent/evaluation/initialization_functions.py` registry | Optional |
| Sync applier | `nemo_voice_agent/evaluation/sync_appliers.py` registry | Dual-side domains only |

Two different "domain" strings exist and they are not the same thing:

- `Scenario.domain` — the **registry namespace**. Tools, predicates, init functions, and sync appliers
  are all keyed by it. The bridge ships it to both bots as the `tool_domain` argument of
  `update_system_prompt`.
- The `--domain` CLI filter — a **scenario-name prefix** match on `<domain>__`. `run_evaluation.py`
  computes the available list from scenario names, not from `Scenario.domain`. Keep them identical to
  avoid confusion (`tau2_telecom` is the one deliberate exception: `tau2_telecom_workflow__*`
  scenarios carry `domain = "tau2_telecom"` so they share one tool bucket).

## 1. Add fixtures

Fixtures live inside the installed package, under `nemo_voice_agent/evaluation/data/<domain>/`.
`get_eval_data_root()` (in `nemo_voice_agent/evaluation/__init__.py`) resolves that directory, with the
`EVAL_DATA_ROOT` environment variable as an override. Because the bridge and the bot servers can be
different processes with different roots, every fixture reference stored in shared state is a
**relative** path.

`load_db_artifact()` accepts either `<name>.json` or a `<name>/` directory of per-table shards and
returns the same in-memory dict, so DB hashes match across layouts. Shard when a single file would blow
a hosting size cap — `tau2_airline` ships its DB as one file per table for exactly that reason.

Record upstream source, version, and license in `nemo_voice_agent/evaluation/data/README.md` under
`## Sources & Licenses`; see [Data Provenance](../domain-guides/data-provenance.md).

## 2. Register tools in the domain bucket

The registry is `domain → tool name → class`. Same short class name in two domains is fine; a duplicate
inside one domain raises at import time.

```python
from nemo_voice_agent.evaluation.tools import register_schema_tool_for_eval
from nemo_voice_agent.evaluation.tools._write_tool_base import WriteScenarioTool

MYDOMAIN_ACTION_TYPES = ["booking", "refund"]


@register_schema_tool_for_eval(domain="mydomain")
class BookThingTool(WriteScenarioTool):
    ACTION_TYPES = MYDOMAIN_ACTION_TYPES

    async def _execute(self, **kwargs):
        ...  # mutate self.state["db"]
        self._record_action({"action_type": "booking", "name": "book_thing", "arguments": kwargs})
        return {"status": "ok"}
```

Read-only tools subclass `StandardSchemaTool` directly and record nothing. `_record_action` appends to
`shared_state["actions"]` (the bridge pulls that list at end of scenario) and pushes an `action-applied`
RTVI server message that drives cross-side sync. Set a class-level `name` attribute if you want
snake_case LLM-visible function names instead of the class name. Lookup falls back to the `"default"`
bucket for shared harness tools such as `EndConversationTool`, so your scenarios get those for free —
list them under their registered PascalCase key.

## 3. Write the base scenario

Define a base scenario that binds the registry namespace and derives shared behavior for the domain.

```python
from functools import cached_property
from nemo_voice_agent.evaluation.scenarios.classes import Scenario, SuccessSignal


class MyDomainBaseScenario(Scenario):
    domain = "mydomain"
    success_signals = (SuccessSignal.DB_STATE_MATCH, SuccessSignal.CLEAN_EXIT)

    def setup_shared_state(self, state: dict, side: str) -> None:
        if side == "agent":
            state["db_path"] = f"{self.domain}/db.json"

    @cached_property
    def expected_scenario_db(self) -> dict:
        ...  # gold end state, hashed by the runner
```

`success_signals` is validated at class-definition time: any concrete scenario (one that declares
`name`) must resolve a non-empty tuple of `SuccessSignal` members from itself or an ancestor. Declare it
on the base class, or as a `cached_property` when it depends on per-task opt-ins — `tau2_retail` and
`tau2_telecom` both do this to add `NL_ASSERTION` only when the task carries assertions. See
[Scoring](../understand-scoring/scoring.md) for what each signal means.

`setup_shared_state(state, side)` is called once per side. Two seeding styles:

| Style | Use when | Example |
| --- | --- | --- |
| `state["db"] = <dict>` (inline) | Per-scenario fixture, small | `eva_airline` |
| `state["db_path"] = "<domain>/db.json"` | Shared fixture, large | `tau2_airline`, `tau2_telecom` |

Path seeding is mandatory above roughly 1 MB — pipecat's WebSocket frame cap closes the connection with
code 1009 when a bigger payload is inlined.

## 4. Wire the imports

Decorators only fire if the modules are imported. Add your tool module to the import block at the bottom
of `nemo_voice_agent/evaluation/tools/__init__.py`, and your scenario package to
`nemo_voice_agent/evaluation/scenarios/data/__init__.py`. A domain package's own `__init__.py` should
side-import each `group_Nx.py` shard so every scaffolded scenario registers.

## 5. Understand the runtime state flow

The runtime initializes domain state and tools in the following order.

| Step | RTVI client message | Carries |
| --- | --- | --- |
| Prompt + tool surface | `update_system_prompt` | `prompt`, `tools`, `add_suffix`, `tool_domain` |
| Scenario state seeding | `apply_initialization` | `domain`, `shared_state_init` (JSON string), `actions` |
| Cross-side propagation | `apply_sync_delta` | `domain`, `delta` |
| End-of-scenario pull | `get_scenario_summary` | request `include_db`; response `actions`, `db_hash`, optional `db` |

`apply_initialization` is *the* scenario-state initializer. Its handler merges `shared_state_init` into
the bot's `shared_state` (preserving runtime sentinels), resolves `db_path` into `db` if present, then
dispatches initialization functions. The bridge calls it on both bots for every scenario, even when
there are no init actions, because the merge and DB load must always run. Any failure aborts the
scenario rather than running it half-seeded. `update_system_prompt` does no DB loading.

Each bot owns exactly one DB at `shared_state["db"]` and is side-agnostic. The bridge labels the pulls:
the agent bot's response becomes `db_hash` / `db`, the user bot's becomes `user_db_hash` / `user_db`.
`db_hash` is computed with `nemo_voice_agent/evaluation/db_hash.py`, which both the bot and the runner
import, so canonicalization is identical on both ends. See
[RTVI Messages](../../reference/runtime/rtvi-messages.md) for full payload shapes.

## 6. Optional: DB-state assertion predicates

Use these when the domain has an open solution space and several valid action sequences produce
different whole-DB states that all satisfy the same outcome. Predicates are pure, side-agnostic, and
dispatched **runner-side** via `evaluate_db_state_assertion(domain, assertion, db, user_db)`.

```python
from nemo_voice_agent.evaluation.db_state_predicates import register_db_state_predicate


@register_db_state_predicate(domain="mydomain")
def assert_line_active(db: dict, line_id: str) -> bool:
    return db["lines"][line_id]["status"] == "Active"
```

Each entry in `Scenario.db_state_assertions` has the shape
`side`, `func_name`, `arguments`, `assert_value`, and optional `message`. The bridge sets
`include_db=True` on the summary pull whenever `db_state_assertions` is truthy, so the actual dicts
come back for the predicates to read.

## 7. Optional: initialization functions

These mutate the live DB before the conversation starts. Signature is `(db: dict, **arguments) -> None`,
mutating in place, dispatched **bot-side** (the opposite of predicates, because the mutation has to land
in the same dict instance the live tools will see).

```python
from nemo_voice_agent.evaluation.initialization_functions import register_initialization_function


@register_initialization_function(domain="mydomain")
def turn_roaming_off(db: dict, line_id: str) -> None:
    db["lines"][line_id]["roaming_enabled"] = False
```

Populate `Scenario.initialization_actions` with `side`, `func_name`, `arguments` records. The bridge
filters by `side` before sending, and rejects any record whose `side` is neither `"user"` nor
`"agent"`.

## 8. Optional: dual-side domains

A dual-side domain gives the user simulator its own tools and its own DB. `tau2_telecom` is the only
one today — see [tau2-telecom](../domain-guides/tau2-telecom.md). Requirements:

1. Set `has_user_state = True` on the base scenario so gold replay seeds the user side and the bridge
   dual-pulls at end of scenario.
2. Seed both sides from `setup_shared_state` (`side="agent"` and `side="user"` branches).
3. Override `sync_state(agent_db, user_db)` to return per-side deltas, shaped
   `agent` and `user` keys mapping to delta dicts. Returning empty dicts is the base-class no-op that
   keeps single-side domains out of the pipeline entirely.
4. Provide `_build_tool_map(state)` returning name-to-instance pairs where each tool has a **sync**
   `invoke(**kwargs)` method. The bridge replays fired actions onto in-process shadow DBs through this
   map before calling `sync_state`. Tau2 tools get this from `_Tau2InvokeMixin`; EVA tools have only
   the async `_execute` and would need a sync wrapper first.
5. Register a sync applier if your deltas are not plain dotted paths:

```python
from nemo_voice_agent.evaluation.sync_appliers import register_sync_applier


@register_sync_applier(domain="mydomain")
def apply_mydomain_sync_delta(db: dict, delta: dict) -> None:
    ...  # mutate db in place
```

The default applier handles dotted-path assignment such as `surroundings.roaming_allowed`. Anything
richer — list-by-id lookups, post-apply re-derivation — needs your own applier.

Sync runs at two points: once after `apply_initialization` (so the conversation starts from coherent
cross-side state), and again after every `action-applied` event from either bot. Both bots need
`llm.enable_tool_calling: true` in their server config for a dual-side domain.

## 9. Verify

Run the focused registry, scoring, and documentation checks after wiring the new domain.

```bash
cd /path/to/Voice-Agent/evaluation
python run_evaluation.py --list-domains       # your domain, with its scenario count
python run_evaluation.py --scenarios mydomain__1   # smoke-run one scenario
```

Both bot servers must be running first, and `SERVER_CONFIG_PATH` resolves against the current working
directory — see the [Evaluation Quickstart](../run-evaluations/quickstart.md) and the
[Evaluation CLI reference](../../reference/evaluation/eval-cli.md).

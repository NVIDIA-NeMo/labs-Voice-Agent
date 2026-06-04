# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A standalone, pip-installable real-time voice agent built on **Pipecat** (`pipecat-ai==0.0.98`) wired together with NeMo speech models and either a HuggingFace or vLLM LLM backend.

- Python **3.12–3.13** (the parent `NeMo-main` toolkit is 3.10+; this repo is its own venv).
- Default install pulls **CUDA 13.0** PyTorch/vLLM wheels (`torch-backend = "cu130"` in `pyproject.toml`). Override via `pyproject.toml` if you need cu128/cu124/cpu.
- The library lives under `nemo_voice_agent/` at the repo root and is imported as `nemo_voice_agent.*`. `examples/generic_voice_agent/` holds a runnable server + browser client; `evaluation/` holds the two-bot eval harness; `tests/` holds pytest suites.

Top-level layout:

```
nemo_voice_agent/                    # the installable package (import as nemo_voice_agent.*)
├── pipecat/                         # Pipecat services, processors, transports, frames
├── evaluation/                      # bridge, runner, scenarios, tools, db_hash
├── utils/                           # config_manager, audio, tool_calling, logging
└── vllm/v1/                         # vLLM logits-processor + tool/reasoning parser plugins
    ├── sample/                      # reasoning_budget logits processor
    └── parsers/                     # tool-parser + reasoning-parser plugins (loaded by vLLM)
examples/generic_voice_agent/
├── server/                          # server.py + server_configs/ + example_prompts/
└── client/                          # React + Vite browser frontend
evaluation/                          # run_evaluation.py, bot_server.py, server_configs/, data/
tests/                               # pytest suites
```

## Install & run

```bash
# One-shot install (apt deps + uv + venv): bash install.sh
uv sync
source .venv/bin/activate

# Server (terminal 1)
# export SERVER_CONFIG_PATH=examples/generic_voice_agent/server/server_configs/default.yaml   # override default
# export HF_TOKEN=hf_...                          # for gated HF models
python examples/generic_voice_agent/server/server.py

# Client (terminal 2)
cd examples/generic_voice_agent/client && npm install && npm run dev   # vite on http://localhost:5173
```

`pip install nemo_voice_agent` is enough to use the library from elsewhere — no `PYTHONPATH` gymnastics needed. The server/client/eval scripts still expect a repo checkout because they read YAML configs and fixtures from repo-relative paths.

The server binds two ports: a **WebSocket** for the audio pipeline (`WEBSOCKET_PORT`, default `8765`) and a **FastAPI** control plane (`FASTAPI_PORT`, default `7860`). Bind addresses come from `SERVER_HOST` (default `0.0.0.0`).

Browsers block mic access on plain HTTP — Chrome users must allow-list `http://<host>:5173/` via `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.

## Tests

```bash
pytest tests/ -v                                          # all
pytest tests/test_config_manager.py -v                    # config loader
pytest tests/test_reasoning_budget_logits_processor.py    # vLLM reasoning-budget processor (needs CUDA + tokenizer download)
```

Tests import the installed `nemo_voice_agent` package, so a fresh `uv sync` is enough to run them.

## Server architecture

`examples/generic_voice_agent/server/server.py:run_bot_websocket()` is the whole show — it loads a YAML config and assembles a Pipecat pipeline:

```
ws.input → RTVI → STT → [Diar?] → TurnTaking → UserAggregator → LLM → TTS → ws.output → AssistantAggregator
```

Components are constructed via the **builder pattern** in `nemo_voice_agent/pipecat/services/nemo/builders.py` (`build_stt`, `build_diar`, `build_llm`, `build_tts`, `build_turn_taking`, `build_vad_analyzer`, `build_ws_transport`, `build_audio_logger`, `build_context_and_aggregators`). The example file rarely needs editing — most behavioral changes happen in YAML or in the builders/services under `nemo_voice_agent/`.

Key cross-cutting concepts:

- **`ConfigManager`** (`nemo_voice_agent/utils/config_manager.py`) loads `examples/generic_voice_agent/server/server_configs/default.yaml`, then merges in the model-specific YAML referenced by each component's `model_config:` field (or auto-resolves via `server/model_registry.yaml` when `server.use_model_registry: true`). Configs use OmegaConf interpolation (e.g. `${llm.temperature}`) — be aware when adding new keys.
- **LLM backend selection.** `llm.type` is `auto | hf | vllm`. `auto` tries vLLM first and falls back to HF. vLLM is required for tool calling. When `start_vllm_on_init: true` the server spawns vLLM via `vllm serve` with the flags in `vllm_server_params`; otherwise you must start vLLM in another terminal (see README for the Nemotron-Nano-3 30B example).
- **Reasoning / thinking mode.** Off by default. Enable via `llm.enable_reasoning: true` (which switches to the sibling `*_think.yaml` config). `tts.think_tokens=["<think>","</think>"]` causes TTS to skip the reasoning span, so the user only hears the final answer. For vLLM, `--reasoning-parser` filters reasoning out of the OpenAI response entirely (see `nemo_voice_agent/vllm/v1/parsers/nano_v3_reasoning_parser.py`).
- **Backchannels.** `turn_taking.backchannel_phrases_path` (or an inline list) prevents short utterances like "uh-huh" from interrupting the bot. Set to `null` to make any speech interrupt.
- **Single-connection server.** A new WebSocket connection disconnects the previous one (LLM context is preserved). Don't add multi-tenant logic here; this example is single-user by design.

## Config layout (`examples/generic_voice_agent/server/server_configs/`)

```
default.yaml                          # top-level: server/transport/vad/stt/diar/turn_taking/llm/tts
llm_configs/<model>.yaml              # per-model llm sub-config (HF + vLLM params)
llm_configs/<model>_think.yaml        # reasoning-mode variant of the same model
tts_configs/<model>.yaml              # kokoro / fastpitch-hifigan / magpie
stt_configs/nemo_cache_aware_streaming.yaml
NVIDIA_NeMo_models.yaml               # extra NeMo-hosted model defs
```

`examples/generic_voice_agent/server/example_prompts/*.txt` holds reusable system prompts referenceable from `llm.system_prompt` (path-or-literal).

## Tool calling

Two extension points (only works with `llm.type: vllm` + a model whose vLLM tool parser is configured):

1. **Direct functions** — write an async function and pass it to `register_direct_tools_to_llm(...)` in `server.py`. Example: `tool_get_city_weather` from `nemo_voice_agent/utils/tool_calling/basic_tools.py`.
2. **Component-owned tools** — mix `ToolCallingMixin` into a service (STT/TTS/Diar/LLM/TurnTaking) and implement `setup_tool_calling()`. The mixin lives at `nemo_voice_agent/utils/tool_calling/mixins.py`; `KokoroTTSService` in `nemo_voice_agent/pipecat/services/nemo/tts.py` is the canonical example (e.g. "speak faster", "switch to British accent").

## vLLM plugins (tool / reasoning parsers, logits processors)

The tool-parser plugins, reasoning-parser plugins, and the reasoning-budget logits processor live inside the installable package at `nemo_voice_agent/vllm/v1/`:

- `nemo_voice_agent/vllm/v1/parsers/` — tool-parser + reasoning-parser plugins (`--tool-parser-plugin`, `--reasoning-parser-plugin`).
- `nemo_voice_agent/vllm/v1/sample/` — reasoning-budget logits processor.

They run **inside the vLLM process**, not the bot server, so logging/imports there have a different runtime than the rest of the codebase. Because they ship with the package, point vLLM at the installed-package path rather than a repo-relative path — resolve at runtime via something like `importlib.util.find_spec("nemo_voice_agent.vllm.v1.parsers").submodule_search_locations[0]`.

## Evaluation harness (`evaluation/`)

A separate two-bot system: a **simulated user bot** talks to the **agent under test** via a bridge that shuttles audio between two WebSocket Pipecat servers, captures `<final_response>` payloads, and scores them. See `evaluation/README.md` for the full architecture, scenario authoring guide, and tool-system reference. Quick run:

```bash
# Three terminals: user bot (8766), agent bot (8765), bridge
# bot_server.py is the same script for both roles — SERVER_CONFIG_PATH picks the role.
WEBSOCKET_PORT=8766 SERVER_CONFIG_PATH=server_configs/user.yaml  python evaluation/bot_server.py
WEBSOCKET_PORT=8765 SERVER_CONFIG_PATH=server_configs/agent.yaml python evaluation/bot_server.py
python evaluation/run_evaluation.py --domain restaurant
```

Scenario classes live under `nemo_voice_agent/evaluation/scenarios/data/` (one file per domain: `restaurant.py`, `customer_service.py`, `qa.py`, …) and tools under `nemo_voice_agent/evaluation/tools/`. Adding a scenario: subclass the domain's `*BaseScenario`, decorate with `@register_eval_scenario`, override only what differs.

### Eval framework key concepts (read before editing)

The eval framework has evolved beyond a simple `<final_response>` capture. The pieces below are easy to miss if you only read `run_evaluation.py`.

**Shared state via RTVI.** Each side (user / agent) holds a per-scenario `shared_state` dict that the bridge seeds at scenario start. The handle is a `SharedStateRef` dataclass (mirrors `TaskRef`) published by `create_update_system_prompt_action(...)` in `nemo_voice_agent/pipecat/processors/frameworks/rtvi_actions.py` — it gives later RTVI actions a mutable view of the same dict. The bridge passes initial state via the `shared_state_init` JSON-string argument of `update_system_prompt`. Scenarios populate state by overriding `Scenario.setup_shared_state(self, state, side)` (in `scenarios/classes.py`) — same method called twice with `side="user"` / `side="agent"`.

**Bridge-pull summary (not LLM-callable).** End-of-scenario state is **pulled** by the bridge after `<exit>`, not pushed by an LLM tool call. The bridge calls `_retrieve_scenario_summary(ws)` in `nemo_voice_agent/evaluation/bridge.py`, which sends an RTVI `get_scenario_summary` action; the handler (`create_get_scenario_summary_action`) returns `{"actions": [...], "db": {...}}` read straight from `shared_state`. This eliminates the previous double-emit / forgot-to-call / mid-conversation-call class of bugs. **Don't reintroduce a `SubmitTransactionSummaryTool`-style LLM-callable summary.**

**DB-state hash matching (primary signal).** When a scenario sets `expected_scenario_db` (a `cached_property` on the class), the runner compares **hashes** — the actual DB never crosses the WebSocket. The bot computes `get_dict_hash(shared_state["db"])` inside the `get_scenario_summary` RTVI handler and returns the SHA-256 string; the runner computes `get_dict_hash(scenario.expected_scenario_db)` from its in-process gold replay and compares strings. Same module on both sides (`nemo_voice_agent/evaluation/db_hash.py`), so the canonicalization (float `1.0 → 1`, `"none" → None`, `ORDER_INDEPENDENT_LIST_FIELDS`, `HASH_EXCLUDED_KEYS = {"session"}`) is identical. `compute_db_diff` is **no longer invoked on mismatch** — the runner has no actual DB to diff. The action-list (`reference_answer`) remains as a secondary signal. Aggregate: `db_state_success_rate` printed by the runner.

**Auto-aggregated action records.** Each write tool extends `WriteScenarioTool` (in `nemo_voice_agent/evaluation/tools/_write_tool_base.py`) and calls `self._record_action({...})` on success — the record is appended to `shared_state["actions"]` so the bridge picks it up via the pull. Each domain ships its own action-type vocabulary (`AIRLINE_ACTION_TYPES` for eva, `TAU2_AIRLINE_ACTION_TYPES` for tau2) and binds it via the subclass's `ACTION_TYPES` ClassVar. Read tools don't record. The bridge stamps `side="agent"` on each pulled record at write time (telecom M5 will add the symmetric `side="user"` stamp once user-side pull lands).

**DB transfer — path-in, hash-out.** Outbound to the bot: the bridge sends either inline DB content (small per-scenario fixtures, eva_airline) or a path string in `shared_state_init` (`state["db_path"]`, tau2 — its 7 MB shared DB exceeds pipecat's 1 MB WebSocket frame cap and triggers `ConnectionClosedError 1009` when inlined). The bot's `update_system_prompt` handler resolves `state["db_path"]` against `EVAL_DATA_ROOT` and replaces it with `state["db"]` before tools instantiate. Inbound from the bot: `get_scenario_summary` returns only `{actions, db_hash, user_db_hash}` — never the inline DB. Same `db_hash` module imported on both sides ensures byte-identical hashing.

### `eva_airline` domain layout

```
nemo_voice_agent/evaluation/
├── scenarios/data/eva_airline/      # package, not a single file
│   ├── __init__.py                  # re-exports EvaAirlineBaseScenario; imports group_Nx
│   ├── base.py                      # EvaAirlineBaseScenario + 5 hand-authored seeds
│   │                                # (1.1.2, 2.1.1, 3.1.3, 5.1.1, 7.2.1)
│   └── group_{1..7}x.py             # auto-scaffolded scenarios per eva sub-flow
├── tools/eva_airline_tools.py       # 15 ported tools + WriteAirlineTool base
├── tools/eva_airline_params.py      # Pydantic schemas for tool args
└── db_hash.py                       # eva-compatible normalize + hash
```

`EvaAirlineBaseScenario` derives everything from a single class attribute `eva_id` (e.g. `"1.1.2"`) via `cached_property`: `current_date`, `_scenario_db`, `expected_scenario_db`. The dataset metadata is read once per process via `_load_eva_airline_dataset_index()` (cached at module level). Subclasses only declare `name`, `eva_id`, `description`, `user_persona`, `user_task`, `user_actions`.

Voice-readability rule on `EvaAirlineBaseScenario.VOICE_ALPHANUMERIC_RULE`: alphanumerics are spelled **canonical-first**: `EPXYEK (spelled out as E, P, X, Y, E, K)`. Use this constant in both agent and user guidelines.

Fixtures live in `evaluation/data/` at the repo root (resolved by `get_eval_data_root()`, override via `EVAL_DATA_ROOT`). The directory has a `README.md` recording upstream source + license for each domain — append a section when adding a new source. The `get_eval_data_root()` helper is at `nemo_voice_agent/evaluation/__init__.py` and uses `parents[2]` to walk from `nemo_voice_agent/evaluation/__init__.py` to the repo root.

### Scaffolding more eva scenarios

The 5 seed scenarios in `base.py` are hand-authored; the rest are scaffolded from `eva_airline_dataset.jsonl` via a personal-scratch generator (kept outside this repo, in a `nemo_experiments/` dir you maintain yourself):

```bash
python <your-scratch-dir>/generate_eva_airline_scaffolds.py --major 4 \
    >> nemo_voice_agent/evaluation/scenarios/data/eva_airline/group_4x.py
```

The generator emits one `@register_eval_scenario` class per dataset entry, applies the alphanumeric voice rule, and reads `must_have_criteria` / `negotiation_behavior` / `edge_cases` into guidelines. **The output is a starting point, not final** — hand-review prose and prune negotiation/edge-case bullets before committing.

### Tool registry — per-domain namespaces

The tool registry is **`Dict[domain → Dict[class_name → class]]`** (`ALL_SCHEMA_TOOLS_FOR_EVAL` in `tools/__init__.py`). Tool classes keep their natural short names (`CancelReservationTool`, `GetUserDetailsTool`, …); the same name in different domains coexists as distinct entries. Within a single domain, duplicate names raise `ValueError` at decoration time.

Decorator (factory-style, takes a `domain` arg):
```python
@register_schema_tool_for_eval(domain="tau2_airline")
class CancelReservationTool(_Tau2WriteTool): ...
```
Also accepts `@register_schema_tool_for_eval("tau2_airline")` (positional shortcut) and bare `@register_schema_tool_for_eval` (back-compat, registers into `"default"`).

Lookup (`get_schema_tool_for_eval(name, domain="default", ...)`): exact match in the specified domain first; falls back to `"default"` with a warning when a shared harness tool (`EndConversationTool`, `SendScenarioSummaryTool`, …) is invoked from a non-default scenario domain; raises `KeyError` if absent from both.

**Domain assignments:**
- `eva_airline_tools.py` → `"eva_airline"`
- `tau2_airline_tools.py` → `"tau2_airline"`
- (future) `tau2_retail_tools.py` → `"tau2_retail"`, `tau2_telecom_tools.py` → `"tau2_telecom"`
- `basic_tools.py`, `customer_service_tools.py`, `restaurant_tools.py`, `rtvi_control.py`, `waitlist_tools.py` → `"default"` (no collisions today; split into per-domain namespaces if any emerge)

**`Scenario.domain` is the namespace key** the bridge passes to the bots via `update_system_prompt`'s `tool_domain` argument. Set as a class attribute on each domain base: `EvaAirlineBaseScenario.domain = "eva_airline"`, `Tau2AirlineBaseScenario.domain = "tau2_airline"`, default `Scenario.domain = "default"`. For tau2, `domain` does dual duty — it's also the data subdir name (`evaluation/data/tau2_airline/`), so internal calls like `f"{self.domain}/db.json"` and `_load_tau2_voice_task_index(self.domain)` use it directly.

**Action record's `name` field** (driven by `_record_action` inside each tool) stays as the upstream method name (`get_user_details`, `book_reservation`, …) — independent of class names. Class names exist for the registry; action names exist for paper-comparable action-list scoring. `TAU2_AIRLINE_TOOL_NAME_TO_CLASS` in `tau2_airline_tools.py` maps between them when needed.

### `tau2_airline` domain layout

```
nemo_voice_agent/evaluation/
├── scenarios/data/
│   ├── tau2_common.py                  # Tau2BaseScenario + _load_tau2_voice_task_index
│   └── tau2_airline/                   # package
│       ├── __init__.py
│       ├── base.py                     # Tau2AirlineBaseScenario
│       └── group_{0..4}x.py            # 50 auto-scaffolded scenarios (10 each)
├── tools/tau2_airline_tools.py         # 14 short-named tool ports under _Tau2InvokeMixin
└── tools/tau2_airline_params.py        # Pydantic mirror of data_model.py + 14 tool-arg schemas
```

`Tau2AirlineBaseScenario` derives everything from a single `tau2_id` class attribute. Subclasses only declare `name` and `tau2_id`; `current_date` / `db` / `policy` / `tool_map` / `expected_scenario_db` / `reference_answer` / `user_persona` are all cached_property views on the upstream data.

**get_agent_prompt is policy.md verbatim + a small "## Additional Notes to Follow" appendage** (`GENERAL_PROMPT` + `VOICE_ALPHANUMERIC_RULE` from `nemo_voice_agent.utils.voice_prompts`, plus `END_CONVERSATION_GUIDELINE` from `nemo_voice_agent.evaluation.scenarios`). The Persona/Task/Actions stubs exist for Scenario-contract introspection only — they do NOT participate in agent-prompt assembly (would silently edit Sierra's authored prompt and break paper-comparable scores).

**user_persona.name is None on purpose.** Identity is owned by `known_info` from `tasks.json["user_scenario"]["instructions"]`. The tau2 `persona_name` (e.g., `"lisa_brenner"`) lives on `scenario.persona_name` for metric slicing only; it's an acoustic-slicing label, not a narrative name.

**reference_answer wraps in `{"actions": [...]}`** to match eva's existing reference shape, so a single `check_if_task_success` Situation 2 path handles both domains.

`_Tau2InvokeMixin` provides both `invoke(**kwargs)` (sync, used by `Tau2BaseScenario._gold_replay`) and `_execute(params)` (async, used by pipecat at live LLM call time), both routing through `_do_work(p)`. New tau2 tools should subclass `_Tau2ReadTool` or `_Tau2WriteTool` (which already wire the mixin) and only implement `_do_work` + `properties` + `required_properties` + `DESCRIPTION`.

DB-key casing in `tau2_airline_tools.py`: reservation IDs and flight numbers are uppercase in `db.json`; user IDs are lowercase. The helper functions `_get_user_dict` / `_get_reservation_dict` / `_get_flight_dict` apply `.lower()` / `.upper()` normalization on the lookup key — voice ASR after letter-by-letter spelling tends to emit case inconsistently.

### `tau2_retail` domain layout

```
nemo_voice_agent/evaluation/
├── scenarios/data/tau2_retail/         # package
│   ├── __init__.py
│   ├── base.py                         # Tau2RetailBaseScenario
│   └── group_{0..11}x.py               # 114 auto-scaffolded scenarios (10 each, last group has 4)
├── tools/tau2_retail_tools.py          # 16 short-named tool ports under _Tau2InvokeMixin
└── tools/tau2_retail_params.py         # Pydantic mirror of data_model.py + 16 tool-arg schemas
```

`Tau2RetailBaseScenario` mirrors the airline base — derives `db` / `policy` / `tool_map` / `expected_scenario_db` / `reference_answer` from `tau2_id` via `Tau2BaseScenario` machinery. Adds **one new property: `nl_assertions`** (cached, read from `evaluation_criteria.nl_assertions`). 40 of 114 retail tasks carry these natural-language claims (e.g. *"Agent should tell the user that there are 10 t-shirt options available."*); 73 are action-only; 1 is nl-only; 1 (task 57) is chitchat with neither signal. Empty/null upstream lists are normalized to `None` so the runner's truthy check correctly skips verdict aggregation.

**nl_assertions LLM-judge integration.** When a scenario has `nl_assertions`, `LLMJudge.judge_scenario` extends its response to include `nl_assertion_verdicts` (one entry per assertion, `{index, passed, reason}`) and `nl_assertion_pass_rate`. The runner aggregates these into a new run-level `nl_assertion_success_rate` and per-domain breakdown alongside the existing `db_state_success_rate`. Malformed verdicts (missing entries, out-of-range indices, non-bool `passed`) get filled as `passed=False` with explanatory reason text — see `tests/test_llm_judge_nl_assertions.py` for the normalization edge cases.

DB-key casing in `tau2_retail_tools.py`: order IDs are uppercase with a leading `#` (`#W0000000`), user IDs are lowercase, product/item IDs are case-sensitive integers (no normalization needed). `_get_order_dict` normalizes to uppercase + prepends `#` if dropped — speakers often omit the `#` when reading an order id aloud.

### Scaffolding more tau2 scenarios

Same pattern as eva: a one-shot generator in `nemo_experiments/` (gitignored) that reads upstream `tasks.json` + `split_tasks.json[base]` + `tasks_voice.json` and emits one `@register_eval_scenario class Tau2<Domain><Id>` per task. Existing generators: `generate_tau2_airline_scaffolds.py`, `generate_tau2_retail_scaffolds.py`. Re-run only when upstream schema changes.

### Running a single eva_airline scenario

```bash
# After both bots are running (see Quick Start above):
python evaluation/run_evaluation.py \
    --scenarios eva_airline__1_1_2 \
    --duration 900                    # bump from default 600s — voice round-trips are ~10× slower than text
```

Scenario names map from eva ids: `"1.1.2" → "eva_airline__1_1_2"`, class names `"1.1.2" → "EvaAirline112"`.

### Known limitations

- **Parakeet STT misrecognizes spelled alphanumerics.** Letter sequences and digit-words (`"for"` vs `"four"`, `"B Z I W"`) frequently get mangled. Diagnose by checking `bot_logs_user/llm_context.json` to confirm the user simulator emitted the correct text before blaming the user-side LLM. Per the path/hash transport design, the user-simulator-side log is the source of truth for what was actually said.
- **VAD can fragment one logical utterance into multiple ASR segments** when the user TTS introduces natural inter-sentence pauses ≥ `VAD_STOP_SECS` (default 1.0 s). The pipeline aggregator can then drop the early segments if the user resumes speaking within ~150 ms (treated as an interruption). Symptom: spelled-out long IDs land in the agent's context truncated or with parts missing. Mitigations: bump `VAD_STOP_SECS` to 2.0 s, or shorten the user simulator's turns so each utterance is a single self-contained statement.
- **DB diff isn't shown on mismatch.** The runner only sees the SHA-256 hash, not the actual DB (path/hash transport). For diagnostics, inspect `bot_logs_agent/llm_context.json` (agent's tool-call trace) and `bot_logs_user/llm_context.json` (user simulator's reasoning) side-by-side — the discrepancy usually pinpoints which reference action the agent skipped or mangled.
- **Action-list lookups are case-sensitive on the action `type`.** Tool action `type` strings must match the domain's `ACTION_TYPES` list (`AIRLINE_ACTION_TYPES`, `TAU2_AIRLINE_ACTION_TYPES`) exactly. The `WriteScenarioTool._record_action` validation logs a warning on mismatch but still appends (recoverable, but the recorded action won't match any reference).

## Code style

Line length 119, black with `skip_string_normalization`, isort `profile=black` (matches the parent NeMo repo's conventions). Run via the dev extras installed by `uv sync`:

```bash
uv run black --line-length 119 --skip-string-normalization <path>
uv run isort --profile black --line-length 119 <path>
```

## Gotchas

- **Don't run `uv sync` inside an active conda env** — `install.sh` exits early in that case because conda's gcc + system Python headers break C extensions like `cdifflib`. Run `conda deactivate` first.
- The egg-info dir (`nemo_voice_agent.egg-info/`), `.venv/`, `nemo_experiments/` (personal scratch + `.env`), `eval_results/`, and `*.log` files are local artifacts — all are gitignored. Don't commit changes to them, and don't copy them around.
- `nemo_voice_agent/vllm/v1/parsers/*.py` and `nemo_voice_agent/vllm/v1/sample/*.py` are vLLM **plugins** — they run inside the vLLM process. Logging/imports there have a different runtime than the rest of the codebase.
- `bot_server.log` saves the logs from the pipecat pipeline, by default it's rotated every day. Recent failures: check the newest `bot_server.<timestamp>.log`, not just `bot_server.log` (which may be from an in-flight run).

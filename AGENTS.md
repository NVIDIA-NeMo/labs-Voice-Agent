# AGENTS.md

This file provides guidance to AI coding agents working in this repository. It follows the
[AGENTS.md](https://agents.md/) convention, so any agent that reads `AGENTS.md` picks it up.

`CLAUDE.md` is a symlink to this file, so Claude Code reads the same content. Skills are shared the
same way: `.claude/skills/` holds them, with `.cursor/skills` and `.codex/skills` symlinked to it.
Edit this file — not the symlinks.

## What this repo is

A standalone, pip-installable real-time voice agent built on **Pipecat** (`pipecat-ai==1.6.0`) wired together with NeMo speech models and either a HuggingFace or vLLM LLM backend.

- Python **3.12–3.13** (`requires-python = ">=3.12,<3.14"`). This repo is standalone — it has its own venv and no parent NeMo checkout.
- Default install pulls **CUDA 13.0** PyTorch/vLLM wheels (`torch-backend = "cu130"` in `pyproject.toml`). Override via `pyproject.toml` if you need cu128/cu124/cpu.
- The library lives under `nemo_voice_agent/` at the repo root and is imported as `nemo_voice_agent.*`. `examples/generic_voice_agent/` holds a runnable server + browser client; `evaluation/` holds the two-bot eval harness; `tests/` holds pytest suites.

Top-level layout:

```
nemo_voice_agent/                    # the installable package (import as nemo_voice_agent.*)
├── pipecat/                         # bot_server.py + services, processors, frames
│                                    #   (no transports/ — pipecat 1.6 supplies the WS transport)
├── evaluation/                      # bridge, runner, scenarios, tools, db_hash
│   └── data/                        # packaged eval fixtures (eva_airline, tau2_*)
├── utils/                           # config_manager, audio, misc (logging), voice_prompts,
│                                    #   websocket_url, tool_calling/
└── vllm/v1/sample/logits_processor/ # reasoning_budget logits processor (vLLM plugin)
examples/generic_voice_agent/
├── server/                          # server.py + server_configs/ + example_prompts/
│   └── parsers/                     # vLLM tool-call + reasoning parser plugins
└── client/                          # vanilla-TS + Vite browser frontend (@pipecat-ai/client-js)
evaluation/                          # run_evaluation.py, bot_server.py, check_resume.py, server_configs/
tests/                               # pytest suites — tests/unit/ and tests/functional/ only
docs/                                # Fern documentation site (see "Documentation site" below)
scripts/                             # prepare_eva_data/, prepare_tau2_data/ scaffold generators
docker/                              # Dockerfile.ci (CI image only; no runtime image)
```

## Server architecture

`examples/generic_voice_agent/server/server.py:run_bot_websocket()` is the whole show — it loads a YAML config and assembles a Pipecat pipeline:

```
ws.input → VAD → RTVI → STT → [Diar?] → [TurnTaking?] → [UserAudioBuffer?] → UserAggregator
        → LLM → [LLMTextProcessor?] → TTS → ws.output → AssistantAggregator
```

`[?]` stages are omitted when their builder returns `None`. VAD is *not* optional — `build_vad_analyzer`
always returns an analyzer, so `build_vad_processor` never returns `None` on this path.
`UserAudioBuffer` is only inserted for omni/multimodal models (`llm.is_omni_model`).

Components are constructed via the **builder pattern** in `nemo_voice_agent/pipecat/services/nemo/builders.py` (`build_stt`, `build_diar`, `build_llm`, `build_tts`, `build_turn_taking`, `build_vad_analyzer`, `build_vad_processor`, `build_llm_text_processor`, `build_ws_transport`, `build_audio_logger`, `build_context_and_aggregators`). The example file rarely needs editing — most behavioral changes happen in YAML or in the builders/services under `nemo_voice_agent/`.

Key cross-cutting concepts:

- **`ConfigManager`** (`nemo_voice_agent/utils/config_manager.py`) loads `examples/generic_voice_agent/server/server_configs/default.yaml`, then merges in the model-specific YAML referenced by each component's `model_config:` field (or auto-resolves via `server/model_registry.yaml` when `server.use_model_registry: true`). Configs use OmegaConf interpolation (e.g. `${llm.temperature}`) — be aware when adding new keys.
  **Precedence gotcha:** the model sub-YAML *overrides* the top-level config, not the other way round (`config_manager.py:282-293` assigns `server_config.llm[key] = llm_config[key]`). So `default.yaml`'s `llm.type: auto` is overwritten to `vllm` by `llm_configs/nemotron_nano_v3.yaml`'s `type: vllm`. Editing a key in `default.yaml` that the sub-YAML also sets has no effect.
- **LLM backend selection.** `llm.type` is `auto | hf | vllm | nvidia`. `auto` tries vLLM first and falls back to HF; `nvidia` targets hosted NIM/Riva endpoints (`server_configs/default_nvidia.yaml`). When `start_vllm_on_init: true` the server spawns vLLM via `vllm serve` with the flags in `vllm_server_params`; otherwise you must start vLLM in another terminal. **The shipped default (`nemotron_nano_v3.yaml`) sets `start_vllm_on_init: false`, so `python server.py` alone will not work — see the README's "Start the vLLM server" step.**
- **Reasoning / thinking mode.** Off by default. `llm.enable_reasoning: true` switches to the sibling `*_think.yaml` config **only when the model was resolved through `model_registry.yaml` and its entry sets `reasoning_supported: true`** (`config_manager.py:273-278`); an explicit `llm.model_config:` short-circuits registry lookup at `:255`, so the swap does not fire for the shipped default — point `model_config:` at the `*_think.yaml` by hand instead. `tts.think_tokens=["<think>","</think>"]` causes TTS to skip the reasoning span, so the user only hears the final answer. For vLLM, `--reasoning-parser` filters reasoning out of the OpenAI response entirely; the `nemotron_nano_v3*.yaml` configs use vLLM's built-in `nemotron_v3`, the eval configs under `evaluation/server_configs/` use `deepseek_r1`, and the rest set none.
- **Backchannels.** `turn_taking.backchannel_phrases_path` (or an inline list) prevents short utterances like "uh-huh" from interrupting the bot. Set to `null` to make any speech interrupt.
- **Single-connection server.** While a client is connected, new connections are rejected with close code 1013 and the existing client is kept; the server accepts a new client once the current one disconnects. LLM context is preserved across reconnects. (Pipecat 1.0 inverted this — before, a new connection kicked the incumbent.) Don't add multi-tenant logic here; this example is single-user by design.

`examples/generic_voice_agent/server/server_configs/` has `default.yaml` and `default_nvidia.yaml` (top-level) plus `llm_configs/`, `tts_configs/`, `stt_configs/` sub-configs. `example_prompts/*.txt` holds reusable system prompts referenceable from `llm.system_prompt` (path-or-literal).

## Tool calling

Two extension points. Requires `llm.type: vllm` (with a model whose vLLM tool parser is configured) or `llm.type: nvidia`; `server.py` gates registration only on `llm.enable_tool_calling`, with no backend check:

1. **Direct functions** — write an async function and pass it to `register_direct_tools_to_llm(...)` in `server.py`. Example: `tool_get_city_weather` from `nemo_voice_agent/utils/tool_calling/basic_tools.py`.
2. **Component-owned tools** — mix `ToolCallingMixin` into a service (STT/TTS/Diar/LLM/TurnTaking) and implement `setup_tool_calling()`. The mixin lives at `nemo_voice_agent/utils/tool_calling/mixins.py`; `KokoroTTSService` in `nemo_voice_agent/pipecat/services/nemo/tts.py` is the canonical example (e.g. "speak faster", "switch to British accent").


## Evaluation harness (`evaluation/`)

A separate two-bot system: a **simulated user bot** talks to the **agent under test** via a bridge that shuttles audio between two WebSocket Pipecat servers, captures `<final_response>` payloads, and scores them. See `evaluation/README.md` for the full architecture, scenario authoring guide, and tool-system reference. Quick run:

```bash
# Three terminals: user bot (8766), agent bot (8765), bridge.
# bot_server.py is the same script for both roles — SERVER_CONFIG_PATH picks the role.
# SERVER_CONFIG_PATH is resolved against the CWD, NOT against the script dir — you must cd first.
cd evaluation && WEBSOCKET_PORT=8766 SERVER_CONFIG_PATH=server_configs/user.yaml  python bot_server.py
cd evaluation && WEBSOCKET_PORT=8765 SERVER_CONFIG_PATH=server_configs/agent.yaml python bot_server.py
cd evaluation && python run_evaluation.py --domain restaurant
```

Scenario classes live under `nemo_voice_agent/evaluation/scenarios/data/`. Small domains are one file (`restaurant.py`, `customer_service.py`, `qa.py`, `fastbite.py`, `simple_qa.py`); the four large ones are **packages** (`eva_airline/`, `tau2_airline/`, `tau2_retail/`, `tau2_telecom/`), each a `base.py` plus `group_Nx.py` shards. Tools live under `nemo_voice_agent/evaluation/tools/`. Adding a scenario: subclass the domain's `*BaseScenario`, decorate with `@register_eval_scenario`, override only what differs.

### Eval framework key concepts (read before editing)

The eval framework has evolved beyond a simple `<final_response>` capture. The pieces below are easy to miss if you only read `run_evaluation.py`.

**Shared state via RTVI.** Each side (user / agent) holds a per-scenario `shared_state` dict that the bridge seeds at scenario start. The handle is a `SharedStateRef` dataclass (mirrors `TaskRef`) published by `create_update_system_prompt_action(...)` in `nemo_voice_agent/pipecat/processors/frameworks/rtvi_actions.py` — it gives later RTVI handlers a mutable view of the same dict. The bridge passes initial state via the `shared_state_init` JSON-string argument of **`apply_initialization`** (not `update_system_prompt`, whose payload is only `{prompt, tools, add_suffix, tool_domain}`). Scenarios populate state by overriding `Scenario.setup_shared_state(self, state, side)` (in `scenarios/classes.py`) — same method called twice with `side="user"` / `side="agent"`.

**Bridge-pull summary (not LLM-callable).** End-of-scenario state is **pulled** by the bridge after `<exit>`, not pushed by an LLM tool call. The bridge calls `_retrieve_scenario_summary(ws)` in `nemo_voice_agent/evaluation/bridge.py`, which sends an RTVI `get_scenario_summary` action; the handler (`create_get_scenario_summary_action`) returns `{"actions": [...], "db_hash": "<sha256>"}` read straight from `shared_state`; the inline `db` dict is added only when the caller passes `include_db: true`. This eliminates the previous double-emit / forgot-to-call / mid-conversation-call class of bugs. **Don't reintroduce a `SubmitTransactionSummaryTool`-style LLM-callable summary.**

**DB-state hash matching (primary signal).** When a scenario sets `expected_scenario_db` (a `cached_property` on the class), the runner compares **hashes** — the actual DB never crosses the WebSocket. The bot computes `get_dict_hash(shared_state["db"])` inside the `get_scenario_summary` RTVI handler and returns the SHA-256 string; the runner computes `get_dict_hash(scenario.expected_scenario_db)` from its in-process gold replay and compares strings. Same module on both sides (`nemo_voice_agent/evaluation/db_hash.py`), so the canonicalization (float `1.0 → 1`, `"none" → None`, `ORDER_INDEPENDENT_LIST_FIELDS`, `HASH_EXCLUDED_KEYS = {"session"}`) is identical. `compute_db_diff` is **no longer invoked on mismatch** — the runner has no actual DB to diff. The action-list (`reference_answer`) remains as a secondary signal. Aggregate: `db_state_success_rate` printed by the runner.

**Auto-aggregated action records.** Each write tool extends `WriteScenarioTool` (in `nemo_voice_agent/evaluation/tools/_write_tool_base.py`) and calls `self._record_action({...})` on success — the record is appended to `shared_state["actions"]` so the bridge picks it up via the pull. Each domain ships its own action-type vocabulary (`AIRLINE_ACTION_TYPES` for eva, `TAU2_AIRLINE_ACTION_TYPES` for tau2) and binds it via the subclass's `ACTION_TYPES` ClassVar. Read tools don't record. The bridge stamps `side="agent"` on each pulled record from the agent ws and `side="user"` on each pulled record from the user ws (telecom dual-pull). `_save_final_response` preserves any pre-stamped `side` field rather than overwriting it, so the user-side records keep their `"user"` tag through the merge.

Per-action emit (telecom cross-side sync). `_record_action` additionally pushes an `action-applied` `RTVI.ServerMessage` via `push_transport_message` when `shared_state["__rtvi__"]` is set (stashed by the `update_system_prompt` handler). The bridge listens for it to drive the cross-side sync pipeline — see the "Cross-side state sync" section below. Single-side domains receive the message harmlessly; their `Scenario.sync_state` is the inherited no-op so the pipeline early-exits.

**DB transfer — path-in, hash-out (with opt-in inline DB for db_state_assertions).** Outbound to the bot: the bridge sends either inline DB content (small per-scenario fixtures, eva_airline) or a path string in `shared_state_init` (`state["db_path"]`, tau2 — its 7 MB shared DB exceeds pipecat's 1 MB WebSocket frame cap and triggers `ConnectionClosedError 1009` when inlined). The bot's **`apply_initialization`** handler (`create_apply_initialization_action`) resolves `state["db_path"]` against `EVAL_DATA_ROOT` via `load_db_artifact` (probes `<path>.json` then `<path>/` directory; tau2_airline ships sharded because its 6.8 MB single file exceeds the GitLab mirror's 5 MB cap) and replaces it with `state["db"]` before tools instantiate. `update_system_prompt` does **no** DB loading — see its docstring at `rtvi_actions.py:277-281`. The seeding happens regardless of `enable_tool_calling`, so user-sim bots with tool calling off — single-side domains — still get DBs loaded for `get_scenario_summary`. Inbound from the bot: `get_scenario_summary` returns `{actions, db_hash}` by default — never the inline DB. An opt-in `include_db: bool` arg makes the response also carry this bot's inline `db` dict so the runner can invoke `db_state_assertions` predicates on it; the bridge sets it when `scenario.db_state_assertions` is truthy (telecom-only today). Each bot returns only ITS OWN DB; the bridge dual-pulls (`get_scenario_summary` from BOTH `agent_ws` and `user_ws`, each pull happening inside that thread's `async with` block to avoid the cross-thread WS-close race) and labels the responses into `scenario_summary["db_hash"]` / `["db"]` (from agent) vs `["user_db_hash"]` / `["user_db"]` (from user). The merge runs in `run_scenario` after both threads join. Same `db_hash` module imported on both sides ensures byte-identical hashing.

**`db_state_assertions` (per-predicate scoring).** Third scoring signal alongside `db_state_match` (binary whole-DB hash equality) and `nl_assertions` (LLM-judged transcript predicates). Each entry is a deterministic predicate function `(db: dict, **arguments) -> bool` evaluated against the pulled DB and compared to an expected `assert_value`. Predicates are pure (no I/O, no randomness) and **side-agnostic** — they don't know which side's DB they're checking. They register flat under `(domain, func_name)` via `@register_db_state_predicate(domain="tau2_telecom")` in `nemo_voice_agent/evaluation/db_state_predicates.py`. Dispatch is **runner-side** (`evaluate_db_state_assertion(domain, assertion, db, user_db)`) — the dispatcher picks `db` vs `user_db` based on the assertion record's `side` field, then invokes the predicate on that single dict. Aggregation mirrors `nl_assertions`: per-scenario `db_state_assertion_pass_rate` in `metrics.json`, run-level `db_state_assertion_success_rate` in `all_summary.txt`. Used by telecom, where open solution spaces mean multiple valid action sequences land in different DBs but satisfy the same outcome predicates. **Upstream tau2-bench calls this `env_assertions`; we rename at the scenario translation boundary** for parallelism with `db_state_match` and `nl_assertions` (renamed `env_type` → `side`, `"assistant"` → `"agent"` for vocab consistency with the bridge's existing side-tagging).

**`initialization_actions` + scenario state init (bot-side).** Bot-side counterpart to `db_state_assertions`. The single `apply_initialization` RTVI client message (handler in `nemo_voice_agent/pipecat/processors/frameworks/rtvi_actions.py:create_apply_initialization_action`) does three things in one call:

1. **Merge `shared_state_init` JSON** (per-side payload from `Scenario.setup_shared_state`) into the bot's `shared_state` dict, preserving bot-side runtime sentinels (`__rtvi__`, `__tool_domain__`) that `update_system_prompt` stashed.
2. **Resolve `db_path` → `db`**: if a `db_path` key is present in the merged state, load the JSON file from `EVAL_DATA_ROOT` and replace it under the `db` key. Idempotent — skipped when `db` is already present.
3. **Apply init-function mutations**: each entry `{func_name, arguments}` in `actions` is dispatched against the now-loaded `db`. Init functions register flat under `(domain, func_name)` via `@register_initialization_function` in `nemo_voice_agent/evaluation/initialization_functions.py` with signature `(db: dict, **arguments) -> None` (mutates in place; side-agnostic — each bot owns one DB at `state["db"]`).

The bridge's `_apply_initialization(scenario, ...)` always calls this on both bots — even when `scenario.initialization_actions` is empty — because steps (1) and (2) (state merge + DB load) must run for every tau2 scenario regardless. Per-side filtering by `side` happens bridge-side before dispatch; the bot dispatcher operates on its own `state["db"]` without side awareness. Mirrors upstream tau2-bench's `Environment.run_env_function_call` semantics. The handler aborts on any per-bot `success: false` so `prepare_for_scenario` fails loud rather than producing partial seeding.

This action is **the** scenario-state initializer — `update_system_prompt` only handles prompt + tool registration + runtime sentinels; ALL scenario fixture data (`db_path`, custom shared-state keys, init mutations) flows through `apply_initialization`.

### `eva_airline` domain

`EvaAirlineBaseScenario` derives everything from a single class attribute `eva_id` (e.g. `"1.1.2"`) via `cached_property`: `current_date`, `_scenario_db`, `expected_scenario_db`. The dataset metadata is read once per process via `_load_eva_airline_dataset_index()` (cached at module level). Subclasses only declare `name`, `eva_id`, `description`, `user_persona`, `user_task`, `user_actions`.

Voice-readability rule: `VOICE_ALPHANUMERIC_RULE` is a **module-level constant in `nemo_voice_agent/utils/voice_prompts.py`** (the eva base merely imports it — it is not a class attribute). The rule says to speak **ONLY the spelled-out form** — `E, P, X, Y, E, K`, never the canonical `EPXYEK` alongside it. Use this constant in both agent and user guidelines.

Fixtures are **packaged inside the library** at `nemo_voice_agent/evaluation/data/`. `get_eval_data_root()` (in `nemo_voice_agent/evaluation/__init__.py`) resolves in order: `$EVAL_DATA_ROOT` → the packaged `<pkg>/evaluation/data/` dir → a `parents[2]`-based `<repo>/evaluation/data/` legacy fallback that no longer exists and is never reached. The data dir has a `README.md` recording upstream source + license for each domain — append a section when adding a new source.

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
- `tau2_retail_tools.py` → `"tau2_retail"`
- `tau2_telecom_user_tools.py` + `tau2_telecom_tools.py` → `"tau2_telecom"` (split because telecom is the first dual-side domain; user-side tools live in their own file)
- `basic_tools.py`, `customer_service_tools.py`, `restaurant_tools.py`, `rtvi_control.py`, `waitlist_tools.py` → `"default"` (no collisions today; split into per-domain namespaces if any emerge)

**`Scenario.domain` is the namespace key** the bridge passes to the bots via `update_system_prompt`'s `tool_domain` argument. Set as a class attribute on each domain base: `EvaAirlineBaseScenario.domain = "eva_airline"`, `Tau2AirlineBaseScenario.domain = "tau2_airline"`, default `Scenario.domain = "default"`. For tau2, `domain` does dual duty — it's also the data subdir name (`nemo_voice_agent/evaluation/data/tau2_airline/`, resolved relative to `get_eval_data_root()`), so internal calls like `f"{self.domain}/db.json"` and `_load_tau2_voice_task_index(self.domain)` use it directly.

**Action record's `name` field** (driven by `_record_action` inside each tool) stays as the upstream method name (`get_user_details`, `book_reservation`, …) — independent of class names. Class names exist for the registry; action names exist for paper-comparable action-list scoring. `TAU2_AIRLINE_TOOL_NAME_TO_CLASS` in `tau2_airline_tools.py` maps between them when needed.

### `tau2_airline` domain

`Tau2AirlineBaseScenario` derives everything from a single `tau2_id` class attribute. Subclasses only declare `name` and `tau2_id`; `db` / `policy` / `expected_scenario_db` / `reference_answer` / `user_persona` are cached_property views on the upstream data. (There is no `current_date` or `tool_map` attribute — the tool map is built on demand by the abstract `_build_tool_map(state)`.)

**get_agent_prompt is policy.md verbatim + a small "## Additional Notes to Follow" appendage** (`GENERAL_PROMPT` + `VOICE_ALPHANUMERIC_RULE` from `nemo_voice_agent.utils.voice_prompts`, plus `END_CONVERSATION_GUIDELINE` **and `EXECUTION_HONESTY_GUIDELINE`** from `nemo_voice_agent.evaluation.scenarios`). The Persona/Task/Actions stubs exist for Scenario-contract introspection only — they do NOT participate in agent-prompt assembly (would silently edit Sierra's authored prompt and break paper-comparable scores).

**user_persona.name is None on purpose.** Identity is owned by `known_info` from `tasks.json["user_scenario"]["instructions"]`. The tau2 `persona_name` (e.g., `"lisa_brenner"`) lives on `scenario.persona_name` for metric slicing only; it's an acoustic-slicing label, not a narrative name.

**reference_answer wraps in `{"actions": [...]}`** to match eva's existing reference shape, so a single `check_if_task_success` Situation 2 path handles both domains.

`_Tau2InvokeMixin` provides both `invoke(**kwargs)` (sync, used by `Tau2BaseScenario._gold_replay`) and `_execute(**kwargs)` (async, used by pipecat at live LLM call time), both routing through `_do_work(p)`. Since commit f5823d8 the base contract is `async def _execute(self, **kwargs) -> Any` — it no longer receives a `params` object and no longer delivers the result itself; `StandardSchemaTool.__call__` owns delivery. Only `_do_work(p)` takes the validated params model. New tau2 tools should subclass `_Tau2ReadTool` or `_Tau2WriteTool` (which already wire the mixin) and only implement `_do_work` + `properties` + `required_properties` + `DESCRIPTION`.

DB-key casing in `tau2_airline_tools.py`: reservation IDs and flight numbers are uppercase in `db.json`; user IDs are lowercase. The helper functions `_get_user_dict` / `_get_reservation_dict` / `_get_flight_dict` apply `.lower()` / `.upper()` normalization on the lookup key — voice ASR after letter-by-letter spelling tends to emit case inconsistently.

### `tau2_retail` domain

`Tau2RetailBaseScenario` mirrors the airline base — derives `db` / `policy` / `expected_scenario_db` / `reference_answer` from `tau2_id` via `Tau2BaseScenario` machinery. Adds **one new property: `nl_assertions`** (cached, read from `evaluation_criteria.nl_assertions`). 40 of 114 retail tasks carry these natural-language claims (e.g. *"Agent should tell the user that there are 10 t-shirt options available."*); 73 are action-only; 1 is nl-only; 1 (task 57) is chitchat with neither signal. Empty/null upstream lists are normalized to `None` so the runner's truthy check correctly skips verdict aggregation.

**nl_assertions LLM-judge integration.** When a scenario has `nl_assertions`, `LLMJudge.judge_scenario` extends its response to include `nl_assertion_verdicts` (one entry per assertion, `{index, passed, reason}`) and `nl_assertion_pass_rate`. The runner aggregates these into a new run-level `nl_assertion_success_rate` and per-domain breakdown alongside the existing `db_state_success_rate`. Malformed verdicts (missing entries, out-of-range indices, non-bool `passed`) get filled as `passed=False` with explanatory reason text — see `tests/unit/test_llm_judge_nl_assertions.py` for the normalization edge cases.

DB-key casing in `tau2_retail_tools.py`: order IDs are uppercase with a leading `#` (`#W0000000`), user IDs are lowercase, product/item IDs are case-sensitive integers (no normalization needed). `_get_order_dict` normalizes to uppercase + prepends `#` if dropped — speakers often omit the `#` when reading an order id aloud.

### `tau2_telecom` domain

Telecom is the **first dual-side domain**: a separate user-side `TelecomUserDB` (mock phone attributes + user surroundings) lives alongside the agent-side `TelecomDB`. `Tau2TelecomBaseScenario.has_user_state = True` triggers user-side seeding in `_gold_replay` and bridge dual-pull at end of scenario. The 30 user-side tools are exposed to the user-sim's LLM (`enable_tool_calling: true` on the user bot); the 13 agent-side tools are exposed to the agent.

**Policy variants — two parallel registrations.** Mirrors upstream tau2's `--domain telecom` vs `--domain telecom-workflow` registration split. Each base-split task is emitted as **two** scenario classes by the scaffold generator:

- `tau2_telecom__X` (inherits `Tau2TelecomBaseScenario`, `policy_variant="manual"`) — uses `tech_support_manual.md` (long-form prose).
- `tau2_telecom_workflow__X` (inherits `Tau2TelecomWorkflowBaseScenario`, `policy_variant="workflow"`) — uses `tech_support_workflow.md` (procedural step-by-step).

Both share the same upstream task → identical `tau2_id`, `db`, `user_db`, `reference_answer`, `db_state_assertions`, `initialization_actions`, `nl_assertions`, agent/user tool surface, and sync_state logic. **Only the rendered policy file differs.**

**Three telecom-specific agent-prompt addenda** appended after the parent's voice-realization notes (in `get_agent_prompt`), compensating for structural gaps vs upstream tau2's text-mode evaluation:
1. `TELECOM_AGENT_TOOL_AVAILABILITY_NOTE` — explicit enumeration of the 13 agent-callable tools vs the 30 user-controlled phone tools, by snake_case name matching `policy.md` references. Prevents the LLM from hallucinating user-side tool calls (which would return `unknown_tool` errors).
2. `TELECOM_AGENT_STAY_ON_TASK_GUIDELINE` — instructs the agent to address the user's primary stated problem and not pivot to incidentally-discovered issues (overdue bills, sibling line suspensions) unless they're directly causing the symptom.
3. `TELECOM_AGENT_HOME_NETWORK_NOTE` — states the home country (US, derived from data: US-format phone numbers, US-state addresses, USD billing) and frontloads a location-probe rule: ASK the user where they are physically located as the FIRST diagnostic question for any connectivity complaint, then check `line.roaming_enabled` if abroad. Addresses the failure mode where the user-sim doesn't volunteer "I'm in France" until prompted.

**User-sim addenda** (in `user_actions`):
- `TELECOM_USER_INSTRUCTIONS` — 4-step procedural script: greet + describe → follow agent's diagnostic steps → verbally report tool results → confirm + close.
- `TELECOM_PASSIVE_TOOL_USE_GUIDELINE` — passive-only tool use; never call a phone tool unless the agent has just instructed a specific action. Prevents the user-sim from free-firing all phone tools on the agent's greeting turn.

**Snake_case tool names.** Telecom tools declare a class-level `name = "snake_case"` attribute (`StandardSchemaTool` honors it). Single source of truth: registry key, LLM-visible function name, `_record_action["name"]`, and `_build_tool_map` gold-replay dispatch key all use the same snake_case identifier matching upstream method names + `policy.md` references. Other domains' tools (eva / airline / retail) don't set `name` and fall back to the class name as before.

**DB-key casing in telecom**: phone numbers `555-NNN-NNNN`, customer IDs uppercase `C1001`, line IDs uppercase `L1002`, plan IDs uppercase `P1001`, bill IDs uppercase `B1001`, device IDs uppercase `D1001`. No normalization helpers needed (all upstream IDs are case-stable).

### Cross-side state sync (telecom; `Environment.sync_tools` equivalent)

Telecom is the first domain to opt into `Scenario.sync_state`, the voice-mode equivalent of upstream's `Environment.sync_tools()`. The framework is **domain-agnostic** at the bridge level — only the propagation logic is telecom-specific.

**Why it's needed.** In voice mode each bot owns one DB (`shared_state["db"]`); the bridge has no shared `Environment` object. So when the agent's `refuel_data` mutates `line.data_refueling_gb`, the user-sim's `surroundings.mobile_data_usage_exceeded` doesn't automatically update — and the user-sim sees stale state. Without sync, ~62% of telecom base scenarios (those involving `refuel_data` / `enable_roaming` / `send_payment_request` / `resume_line`) would deadlock because the user-sim never observes the agent's fix.

**Pipeline (3 invocation points mirror upstream's `sync_tools` call sites):**
1. **Post-init (mirrors upstream `Environment.__init__` sync).** After `_apply_initialization` completes, bridge's `_setup_cross_side_sync` loads shadow DBs, replays init actions onto them, runs `scenario.sync_state` once, and dispatches the resulting per-side deltas to the bots so they start the conversation with coherent cross-side state (e.g. agent's `set_data_usage(15.1)` → user's `surroundings.mobile_data_usage_exceeded=True`).
2. **Per-action (mirrors upstream `run_env_function_call` sync).** `WriteScenarioTool._record_action` emits an `action-applied` `RTVI.ServerMessage` via `push_transport_message`. Bridge listens for it in `_monitor_*_message`, replays the action onto the shadow DBs via the scenario's `_build_tool_map`'s sync `invoke()`, runs `scenario.sync_state`, and dispatches deltas via the `apply_sync_delta` RTVI client message.
3. **(Not yet exercised — would be per-LLM-turn).** Upstream also syncs in `get_response`; we don't need this for voice mode because all bot-side mutations land in `_record_action` and trigger #2.

**Components:**
- `Scenario.sync_state(agent_db, user_db) → {"agent": {...}, "user": {...}}` — default no-op on the base; telecom overrides to delegate to `sync_telecom_state` (pure function in `tau2_telecom_sync.py`).
- `nemo_voice_agent/evaluation/sync_appliers.py` — per-domain applier registry (`@register_sync_applier(domain="...")`) + generic dotted-path default. Telecom registers `apply_telecom_sync_delta` which handles `surroundings.field` paths AND list-by-id paths (`bills[B1002].status`) AND triggers `_simulate_network_search` when surroundings change.
- `apply_sync_delta` RTVI client-message handler (in `rtvi_actions.py`) — parallel to `apply_initialization`; receives `{domain, delta}` and dispatches via the applier registry.
- Bridge state: `self.shadow_state`, `self.shadow_tool_map`, `self.sync_enabled`, `self.sync_lock` — only populated when `type(scenario).sync_state is not Scenario.sync_state` (i.e., the scenario opted in).

**Five propagation paths** in `sync_telecom_state` (telecom-specific):
1. `line.status` → `surroundings.line_active` (agent → user)
2. `line.roaming_enabled` → `surroundings.roaming_allowed` (agent → user)
3. `line.data_used_gb` + `data_refueling_gb` + `plan.data_limit_gb` → `surroundings.mobile_data_usage_exceeded` (agent → user)
4. Any bill `AWAITING_PAYMENT` → `surroundings.payment_request = {bill_id, amount, paid: False}` (agent → user)
5. User's `payment_request.paid = True` → `bills[id].status = Paid` (**user → agent**, reverse direction)

**Contract for new dual-side domains.** Any scenario that overrides `sync_state` MUST also provide `_build_tool_map(state) → {name: tool}` where each tool has a sync `invoke(**kwargs)` method (used by the bridge for shadow-DB replay). Tau2 satisfies this via `_Tau2InvokeMixin`; EVA's tools have only `_execute` (async) and would need a sync wrapper if EVA ever became dual-side. Single-side domains keep the default no-op `sync_state` and skip the whole pipeline.

**Validated** end-to-end against three scenarios covering all 5 propagation paths bidirectionally — see `tests/unit/test_tau2_telecom_sync.py` for the unit-level coverage matrix (26 tests).


## Code style

**Ruff is the only formatter and linter** — black and isort are not installed and are not dependencies. Config
lives in `ruff.toml`: line length 119, `target-version = "py312"`, `quote-style = "double"` (note: strings ARE
normalized to double quotes), and `I` (isort) in the lint `select` with `known-first-party = ["nemo_voice_agent"]`
and `lines-after-imports = 2`. `E501` is ignored — the formatter owns line length.

```bash
uv run ruff format <path>          # format
uv run ruff check --fix <path>     # lint + sort imports
```

`.pre-commit-config.yaml` runs `ruff` + `ruff-format`, and lint is an enforced CI gate.

## Testing

Suites live in `tests/unit/` and `tests/functional/` — there are no test modules directly under `tests/`.
`pyproject.toml` defines the markers `unit`, `functional`, `gpu`, `slow`, `skipduringci`, `pleasefixme`.

```bash
uv run pytest tests/unit -m "not gpu"
```

## Does your change need a docs update?

**Ask this before finishing any code change.** `docs/` is user-facing documentation that describes runtime
behaviour, so a behaviour change that skips it turns the site into a source of false statements — and merging
to `main` publishes it live. An audit of this repo once found ~40 stale doc claims accumulated exactly this
way, each one correct when written.

Check the matching page whenever you touch:

| You changed | Check |
|---|---|
| a config key, default, or `server_configs/**` YAML | `docs/build-voice-agents/configure/`, `docs/reference/runtime/config-schema.md` |
| a `build_*` function or the pipeline order | `docs/about/architecture.md`, `docs/build-voice-agents/extend/pipelines/builders.md` |
| an STT / TTS / diarization / turn-taking service or model | the matching `docs/about/core-concepts/speech-pipeline/*.md` page |
| LLM backend selection, reasoning, or omni | `docs/about/core-concepts/language-models/{llm,reasoning,multimodal}.md` |
| vLLM serving, plugins, or hosted NVIDIA endpoints | `docs/build-voice-agents/model-serving/{vllm,vllm-plugins,nvidia-nim}.md` |
| tool calling or `utils/tool_calling/**` | `docs/build-voice-agents/tools/{tool-calling,custom-tools}.md` |
| an RTVI action or the `/connect` handshake | `docs/build-voice-agents/extend/protocols/{rtvi-actions,client-protocol}.md`, `docs/reference/runtime/rtvi-messages.md` |
| `run_evaluation.py` flags, defaults, or scoring | `docs/evaluate/understand-scoring/scoring.md`, `docs/evaluate/run-evaluations/resume.md`, `docs/reference/evaluation/eval-cli.md` |
| eval scenarios, tools, or fixtures for a domain | `docs/evaluate/domain-guides/*.md`, `docs/evaluate/create-evaluations/authoring-*.md` |
| a metric written to `metrics.json` / `all_summary.txt` | `docs/reference/evaluation/metrics.md` |
| an env var | `docs/reference/runtime/environment.md` |
| deps, Python version, test layout, or lint tooling | `docs/resources/contribute/{index,testing}.md` |

Rules of thumb:

- **Documentation-in-code counts.** An argparse `help=` string, a config comment, or a docstring is
  documentation; fix it in the same change. A help string that contradicts its own default is the single most
  common defect found here.
- **Verify, don't copy.** Never restate a claim from `README.md` or this file without checking it against
  source — both have been wrong. Cite the file you actually opened.
- **`uv run pytest tests/unit/test_docs_consistency.py`** catches the mechanical subset (counts vs enums, CLI
  defaults vs argparse, referenced paths existing). It cannot catch prose that is simply wrong about
  behaviour, so it is a backstop, not a substitute for checking.
- If a change makes a doc page wrong and you cannot fix it in scope, say so explicitly in your summary rather
  than leaving it silently stale.

## Documentation site

`docs/` is a **Fern** site published to `docs.nvidia.com/nemo/labs-voice-agent` (see `docs/fern/docs.yml`).
**⚠️ Merging any `docs/**` change to `main` publishes it live within about a minute** — `publish-fern-docs.yml`
triggers on push to `main` (as well as on Release publication and `workflow_dispatch`), gated only on the
`PUBLISH_FERN` repo variable, which is set. There is no staging channel; review on the PR preview.

Three CI workflows carry a `docs/**` path filter: `fern-docs-ci.yml` (the validation gates),
`fern-docs-preview-build.yml`, and `publish-fern-docs.yml`. `fern-docs-preview-comment.yml` chains off the
preview build via `workflow_run`.

Four things to know before touching it:

- **Navigation is GENERATED — never hand-edit it.** `docs/fern/versions/nightly.yml` and `docs/index.yml` are
  both emitted from `docs/fern/nav.json` by `docs/fern/scripts/gen-nav.mjs`. To add, remove, or reorder a page,
  edit `nav.json` and run `npm --prefix docs/fern run nav:gen`. `fern-docs-ci.yml` runs `nav:check`, which
  fails if the generated files drift from the manifest.
  Why it is generated: Fern needs the two files to use *different* path conventions (`nightly.yml` relative to
  itself, so `../../…`; `index.yml` relative to `docs/`, bare), and only `nightly.yml` is validated by
  `fern check` — `index.yml` is read solely by `publish-fern-docs.yml` at release time. Hand-maintaining both
  is how a page silently disappears from the released channel, which had already happened once.
- **Hard CI gates:** no non-self-closing `<img …>` anywhere under `docs/`; `fern check`; and lychee `--offline`
  over `docs/**/*.md`, which requires every relative link target to exist on disk (no `.lycheeignore` exists).
- Author pages as `.md` (`.mdx` is the generated-only format). Fern renders `.md` through MDX, so bare `{`, `}`,
  `<` outside code fences break the build — that is why `docs/index.md` uses a `{/* … */}` comment header.
- `docs/fern/product-docs/**` is the generated Python API reference: gitignored and regenerated per build.
  `fern check` does **not** need it on disk (CI has no generate step and still passes) — `npm run check` alone
  is enough for a prose change. Only a locally rendered `npm run dev` needs it, via
  `npm run generate:library:local`, which additionally requires temporarily uncommenting the
  `nemo-voice-agent-local` block in `docs.yml` and re-commenting it afterwards.

## Gotchas

- **Don't run `uv sync` inside an active non-`base` conda env** — `install.sh` exits early in that case
  (`[ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "base" ]`) because conda's gcc + system Python
  headers break C extensions like `cdifflib`. Run `conda deactivate` first. An active `base` env passes the
  guard silently; `pyproject.toml`'s `python-preference = "only-managed"` is the actual mitigation.
- Prefer `bash install.sh` over a bare `uv sync`: it also installs `build-essential` and `python3-dev`
  (`cdifflib` is sdist-only and compiles from source) and prefetches NLTK data.
- Every `*.py` file except `__init__.py` needs an SPDX/Apache header in its first 10 lines —
  `copyright-check.yml` runs on every PR and hard-fails without it. Markdown and YAML are not checked, but
  repo convention is to include one anyway.
- The egg-info dir (`nemo_voice_agent.egg-info/`), `.venv/`, `nemo_experiments/` (personal scratch + `.env`), `eval_results/`, and `*.log` files are local artifacts — all are gitignored. Don't commit changes to them, and don't copy them around.
- `examples/generic_voice_agent/server/parsers/*.py` and
  `nemo_voice_agent/vllm/v1/sample/logits_processor/*.py` are vLLM **plugins** — they run inside the vLLM
  process, so logging/imports there have a different runtime than the rest of the codebase.
  `nemotron_toolcall_parser_streaming.py` is **current** — Nemotron-Nano-v2 still needs it, loaded by
  `nemotron_nano_v2.yaml` via `--tool-parser-plugin`. The other two are **deprecated**, superseded by vLLM
  built-ins for Nemotron-3 and newer: `nano_v3_reasoning_parser.py` by `--reasoning-parser nemotron_v3`, and
  `ReasoningBudgetLogitsProcessor` by the `thinking_token_budget` request parameter (see
  `nemotron_nano_v3_think.yaml`). Neither is loaded by any shipped config; both are kept only for deployments
  pinned to older vLLM releases. Don't wire them into new configs — see `docs/build-voice-agents/model-serving/vllm-plugins.md`.
- `bot_server.log` saves the logs from the pipecat pipeline, by default it's rotated every day. Recent failures: check the newest `bot_server.<timestamp>.log`, not just `bot_server.log` (which may be from an in-flight run).

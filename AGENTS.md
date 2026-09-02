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

A two-bot system: a **simulated user bot** talks to the **agent under test** through a bridge that shuttles
audio between two WebSocket Pipecat servers, captures the agent's tool calls, and scores each scenario.

```bash
# Three terminals: user bot (8766), agent bot (8765), bridge.
# bot_server.py is the same script for both roles — SERVER_CONFIG_PATH picks the role.
# SERVER_CONFIG_PATH resolves against the CWD, NOT the script dir — you must cd first.
cd evaluation && WEBSOCKET_PORT=8766 SERVER_CONFIG_PATH=server_configs/user.yaml  python bot_server.py
cd evaluation && WEBSOCKET_PORT=8765 SERVER_CONFIG_PATH=server_configs/agent.yaml python bot_server.py
cd evaluation && python run_evaluation.py --domain restaurant
```

**The reference material lives in `docs/` — read the page, don't reconstruct it from source.** These are
in-repo markdown files, so reading them costs one tool call and they are kept current by
`tests/unit/test_docs_consistency.py`.

| Working on | Read |
| --- | --- |
| Architecture, the bridge, what you get out | `docs/evaluate/index.md` |
| Running, resuming, reading results | `docs/evaluate/run-evaluations/` |
| The six scoring signals and `success_signals` | `docs/evaluate/understand-scoring/scoring.md` |
| Adding a scenario, tool, or domain | `docs/evaluate/create-evaluations/` |
| Per-domain quirks, fixtures, provenance | `docs/evaluate/domain-guides/` |
| RTVI actions and the wire protocol | `docs/build-voice-agents/extend/protocols/rtvi-actions.md`, `docs/reference/runtime/rtvi-messages.md` |
| Every CLI flag and its real default | `docs/reference/evaluation/eval-cli.md` |

Scenario classes live under `nemo_voice_agent/evaluation/scenarios/data/`. Small domains are one file
(`restaurant.py`, `customer_service.py`, `qa.py`, `fastbite.py`, `simple_qa.py`); the four large ones are
**packages** (`eva_airline/`, `tau2_airline/`, `tau2_retail/`, `tau2_telecom/`), each a `base.py` plus
`group_Nx.py` shards. Tools live under `nemo_voice_agent/evaluation/tools/`.

### Constraints that are easy to violate

These are the mistakes this codebase has actually absorbed. The docs explain the mechanisms; this list is
what not to undo.

- **Don't reintroduce an LLM-callable summary tool.** End-of-scenario state is *pulled* by the bridge via the
  `get_scenario_summary` RTVI action, never pushed by the model. A `SubmitTransactionSummaryTool`-style tool
  brings back a whole class of forgot-to-call / double-call / mid-conversation-call bugs.
- **`apply_initialization` seeds scenario state, not `update_system_prompt`.** The latter handles only prompt,
  tool registration, and runtime sentinels; it does no DB loading. Putting fixture data back into it is a
  recurring wrong turn.
- **Fixtures are packaged at `nemo_voice_agent/evaluation/data/`**, not `evaluation/data/`. Resolve through
  `get_eval_data_root()`; the repo-root path is a dead legacy fallback.
- **`CLEAN_EXIT` belongs in every domain's `success_signals`.** A regression test enforces it. An agent that
  does the right work but never stops talking is not a successful agent.
- **A domain that overrides `sync_state` must also provide `_build_tool_map(state)`** whose tools expose a
  sync `invoke(**kwargs)` — the bridge needs it for shadow-DB replay.
- **The tool registry is per-domain** (`Dict[domain -> Dict[name -> class]]`). The same class name in two
  domains is fine; a duplicate *within* a domain raises at decoration time.
- **Hashes cross the WebSocket, not databases.** `db_hash.py` is imported by both sides so canonicalization is
  byte-identical. Changing it on one side only silently breaks `db_state_match`.

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

## Documentation

Before completing a code change, determine whether it affects a user-visible surface. This includes public
APIs, command-line interfaces, configuration, the browser client, workflows, defaults, errors, and other
product behavior.

When a change has user-visible impact, start a documentation subagent in parallel. Give it the changed source
files and the identified user impact, and direct it to read `docs/AGENTS.md`. Require the subagent to update the
affected documentation and run the documented validation while the primary agent continues the implementation.
Reconcile the documentation changes and validation evidence before completing the task, and include required
documentation in the same change.

If the current host cannot run subagents, the primary task must read `docs/AGENTS.md`, complete the same
documentation work, and run the same validation. Do not omit documentation because parallel execution is
unavailable. The scoped guide contains the Writing Style Guide, optional DORI routing, the source-to-page impact
map, Fern maintenance rules, and documentation validation commands.

### Documentation Writer Review Receipt

Every pull request that changes code or documentation must include one
`## Documentation Writer Review` section from
`.github/PULL_REQUEST_TEMPLATE.md`. Complete the review after the changes and
applicable validation are finished.

- Check the review-completion box and keep exactly one result:
  `docs-updated`, `no-docs-needed`, or `blocked`.
- Name the changed documentation in **Evidence**, or explain why documentation
  is not needed or why the review is blocked.
- Record the agent product and surface that performed the review.
- After committing the reviewed changes, fill the hidden head and guidance
  fields with `git rev-parse --short HEAD` and
  `git rev-parse --short HEAD:AGENTS.md`.
- Any later commit makes the receipt stale. Rerun the documentation review and
  refresh both hidden fields.

The `CI / Documentation Writer Review` workflow checks the receipt in advisory
mode. Use the following command to measure adoption. The report also supports
`json` and `csv` formats.

```bash
python scripts/docs-review-receipt.py report --since <YYYY-MM-DD> --format summary
```

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

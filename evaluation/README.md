# NeMo Voice Agent Evaluator

Evaluate a voice agent by having a simulated user (another voice agent) talk to it through a live audio connection. The bridge routes audio, measures latency, captures the agent's actions, propagates cross-side DB state when needed, and scores success across up to six orthogonal signals.

## Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
  - [0. Install dependencies](#0-install-dependencies)
  - [1. Start the two bot servers](#1-start-the-two-bot-servers)
  - [2. Run an evaluation](#2-run-an-evaluation)
  - [3. Scoring](#3-scoring)
  - [4. Resuming a partial run](#4-resuming-a-partial-run)
  - [Helper launch scripts](#helper-launch-scripts)
- [CLI Reference](#cli-reference)
  - [`run_evaluation.py` flags](#run_evaluationpy-flags)
  - [Bot server environment variables](#bot-server-environment-variables)
- [Available Domains and Scenarios](#available-domains-and-scenarios)
  - [eva_airline domain notes](#eva_airline-domain-notes)
  - [tau2 domain notes](#tau2-domain-notes)
- [Evaluation Metrics](#evaluation-metrics)
  - [Signal matrix](#signal-matrix)
  - [1. Action-list match (deterministic)](#1-action-list-match-deterministic)
  - [2. DB-state hash match (deterministic, path-independent)](#2-db-state-hash-match-deterministic-path-independent)
  - [3. DB-state assertions (deterministic, per-predicate)](#3-db-state-assertions-deterministic-per-predicate)
  - [4. NL assertions (LLM-judged, per-assertion)](#4-nl-assertions-llm-judged-per-assertion)
  - [5. LLM judge (soft overall signal)](#5-llm-judge-soft-overall-signal)
  - [External agent compatibility](#external-agent-compatibility)
  - [Composite `is_successful` (per-scenario whitelist)](#composite-is_successful-per-scenario-whitelist)
  - [Run-level aggregation](#run-level-aggregation)
- [Output Structure](#output-structure)
- [Extending the System](#extending-the-system) — pointer to [`EXTENDING_DATA.md`](EXTENDING_DATA.md) + [`EXTENDING_PIPELINE.md`](EXTENDING_PIPELINE.md)
- [Notes](#notes)

## Architecture

```
┌──────────────────────┐        Audio +        ┌──────────────────────┐       Audio +       ┌──────────────────────┐
│   User Bot Server    │        RTVI           │                      │       RTVI          │   Agent Bot Server   │
│  (Simulated User)    │◄─────────────────────►│        Bridge        │◄───────────────────►│  (Agent Under Test)  │
│                      │                       │                      │                     │                      │
│  ASR → LLM → TTS     │                       │  Audio routing       │                     │  ASR → LLM → TTS     │
│  WebSocket on 8766   │                       │  Latency metrics     │                     │  WebSocket on 8765   │
│                      │                       │  Transcript capture  │                     │                      │
│  shared_state["db"]  │                       │  <exit> detection    │                     │  shared_state["db"]  │
│  (user-side DB:      │                       │  RTVI prompt updates │                     │  (agent-side DB:     │
│   phone state for    │                       │  Init action replay  │                     │   reservations,      │
│   telecom user-sim)  │                       │  Cross-side sync     │                     │   bills, lines, ...) │
└──────────────────────┘                       │  (telecom only)      │                     └──────────────────────┘
                                               │  Dual-pull at end    │
                                               └──────────────────────┘
```

- **Two independent WebSocket bot servers.** Each runs its own Pipecat pipeline (NeMo ASR → LLM → TTS), speaks RTVI, and holds a per-scenario `shared_state` dict on its side. For dual-side domains (telecom), the user bot's `shared_state["db"]` is a separate mock-phone DB (`TelecomUserDB`) from the agent bot's customer/billing DB (`TelecomDB`).
- **Bridge process.** Opens a WebSocket client to each bot, runs two threads (one per bot), and shuttles audio between them via thread-safe queues. Resamples audio at the source to match each bot's sample rate. Monitors RTVI events for transcripts, turn timing, `<final_response>` (legacy structured result), `<exit>` (graceful termination), and `action-applied` (per-write-tool event that drives the cross-side sync pipeline for telecom).
- **Scenario-state init.** Per scenario, the bridge sends `update_system_prompt` (sets prompt + registers tools + clears prior shared_state) followed by `apply_initialization` (merges scenario fixture data into `shared_state`, resolves `db_path` to a loaded DB, dispatches any per-side init function mutations like `set_data_usage` / `turn_airplane_mode_on`). Both bots always receive `apply_initialization`, even when no init mutations are needed — the DB-load step runs regardless.
- **Cross-side state sync.** When the agent's write tool fires (e.g., `enable_roaming`, `refuel_data`, `send_payment_request` in `tau2_telecom`), the bot emits an `action-applied` RTVI event. The bridge replays the action onto in-process shadow DBs, runs `scenario.sync_state(agent_db, user_db)`, and pushes the resulting per-side delta to the *other* bot via the `apply_sync_delta` RTVI client message. Mirrors upstream tau2's `Environment.sync_tools()` across three invocation points (post-init, per-action, plus reverse direction when the user-sim makes a payment). Single-side domains (`eva_airline`, `tau2_airline`, `tau2_retail`) skip this pipeline entirely — `Scenario.sync_state` is the inherited no-op default.
- **End-of-scenario pull.** The bridge dual-pulls scenario summaries from BOTH bots via `get_scenario_summary` — each pull happens *inside* its thread's `async with` WebSocket scope to avoid the cross-thread close race. The merged result carries `db_hash` + per-side action records (`side="agent"` / `side="user"` tagged at merge time) for the runner's scoring pass. For telecom scenarios with `db_state_assertions`, the inline `user_db` dict also comes back so the runner can evaluate predicates on it.

## Quick Start

### 0. Install dependencies

From the repo root (`NeMo-Voice-Agent/`):

```bash
uv sync
source .venv/bin/activate
```

`nemo_voice_agent` is pip-installable — no `PYTHONPATH` gymnastics are required. The eval scripts below still expect a repo checkout because they read YAML configs and fixtures from repo-relative paths.

### 1. Start the two bot servers

**Terminal 1 — Simulated User**

```bash
cd evaluation
export SERVER_CONFIG_PATH=server_configs/user.yaml
export WEBSOCKET_PORT=8766
export CUDA_VISIBLE_DEVICES=0
python bot_server.py
```

**Terminal 2 — Agent Under Test**

```bash
cd evaluation
export SERVER_CONFIG_PATH=server_configs/agent.yaml
export WEBSOCKET_PORT=8765
export CUDA_VISIBLE_DEVICES=1
python bot_server.py
```

### 2. Run an evaluation

**Terminal 3 — Evaluation Bridge**

```bash
cd evaluation
python run_evaluation.py \
    --user-url ws://localhost:8766 \
    --agent-url ws://localhost:8765 \
    --domain restaurant
```

### 3. Scoring

Each scenario can be scored by up to **six orthogonal signals** — action-list match, DB-state hash match, DB-state assertions, NL assertions, LLM judge, and clean exit — combined into a single `is_successful` composite via strict conjunction (every applicable signal must pass). Pass `--judge-url`, `--judge-model`, and `--judge-api-key` to enable the LLM judge alongside the deterministic signals; the judge runs independently and contributes its own verdict to the conjunction (when `--judge-threshold` is set). All signals + the composite + a per-scenario `success_breakdown` land in `metrics.json`; the LLM judge's full output (score, reason, per-assertion verdicts, verbatim prompt) lives in `judge_result.json`. See [Evaluation Metrics](#evaluation-metrics) for the full signal model.

### 4. Resuming a partial run

If a long run gets killed (or you killed it intentionally), re-invoke `run_evaluation.py` with `--resume <timestamp>` to pick up where it left off. The runner reuses the existing `eval_<timestamp>/` directory:

```bash
python run_evaluation.py \
    --user-url ws://localhost:8766 --agent-url ws://localhost:8765 \
    --domain tau2_retail \
    --resume 20260609_181545
```

Per-scenario filesystem state determines what happens to each scenario:

| State | What's on disk | Behavior |
|---|---|---|
| **Completed** | `metrics.json` exists and parses cleanly | Skipped. Metrics loaded from disk and folded into the run-level aggregate so `all_summary.txt` covers BOTH this and the previous session. |
| **In-flight** | Subdir exists but no (or malformed) `metrics.json`; or `metrics.json` exists but `agent_turns < --min-agent-turns` | Moved to `<scenario>.killed.<resume_ts>/` and re-run fresh. |
| **Fresh** | No subdir | Runs normally. |

**Invocation history.** Every run writes `<session_dir>/run_args.json` recording the CLI invocation (parsed args, `argv`, resolved scenario list). Resume invocations append a new entry rather than overwrite, so the file shows the full history of the run dir. The LLM judge API key is redacted before writing.

**Consistency soft-check.** On `--resume`, scoring-relevant flags (`--domain` / `--scenarios` / `--duration` / `--judge-*` / `--strict-match`) are compared against the previous invocation; mismatches log a warning but never block — the operator decides whether to proceed. Mixing flags across resumed sessions produces incoherent aggregates (some scenarios scored one way, some another), so for a clean comparison rerun from scratch.

**Final aggregates always regenerate.** `all_metrics.json`, `all_summary.txt`, and `all_latencies.csv` are written fresh at the end of the resume session, covering every scenario in `all_results` (both freshly-run and loaded-from-disk).

**Preview a resume before running it.** `check_resume.py` reports which scenarios a `--resume` would re-run, without touching anything on disk. Note that its `--min-agent-turns` default is `0` (stall detection off), unlike `run_evaluation.py`'s `3` — pass the value you intend to resume with, or the preview will under-report:

```bash
python check_resume.py eval_results/eval_20260618_072325 --min-agent-turns 3
```

### Helper launch scripts

`run_agent.sh` and `run_user.sh` wrap the two `bot_server.py` invocations from [Step 1](#1-start-the-two-bot-servers) with `SERVER_CONFIG_PATH`, `WEBSOCKET_PORT` and `FASTAPI_PORT` already exported. **Run them from `evaluation/`** — they resolve their own directory only to locate `bot_server.py`; the exported `SERVER_CONFIG_PATH` stays relative and `ConfigManager` resolves it against the current working directory, so another cwd fails with `FileNotFoundError: Server configuration file not found at server_configs/agent.yaml`.

```bash
cd evaluation
./run_agent.sh   # agent role,    server_configs/agent.yaml, ws 8765
./run_user.sh    # user-sim role, server_configs/user.yaml,  ws 8766
```

## CLI Reference

### `run_evaluation.py` flags

| Flag | Description |
|------|-------------|
| `--user-url` | WebSocket URL of the user bot (default: `ws://localhost:8766`) |
| `--agent-url` | WebSocket URL of the agent bot (default: `ws://localhost:8765`) |
| `--scenarios <name …>` | Run specific scenarios by name |
| `--domain <name>` | Run all scenarios in a domain (matches `{domain}__*` prefix) |
| `--list` | List all registered scenarios and exit |
| `--list-domains` | List available domains and exit |
| `--audio-chunk-in-seconds <seconds>` | Audio chunk in seconds for the audio stream (default: 0.016) |
| `--duration <seconds>` | Max duration per scenario, in seconds. **Overrides** the scenario's own `max_duration`. Unset by default, in which case each scenario's `max_duration` applies. |
| `--pause <seconds>` | Pause between scenarios (default: 0.5) |
| `--output-dir <path>` | Output directory root (default: `./eval_results`) |
| `--output-sample-rate <hz>` | Sample rate for recorded stereo WAV (default: 16000) |
| `--judge-url <url>` | LLM judge endpoint (OpenAI-compatible chat completions) |
| `--judge-model <model>` | Judge model name |
| `--judge-api-key <key>` | Judge API key (defaults to env var if set) |
| `--judge-threshold <threshold>` | Threshold for the LLM judge score if binary result is desired (default: 0.9) |
| `--judge-timeout <seconds>` | LLM judge request timeout (default: 120) |
| `--judge-thinking-token-budget <tokens>` | Optional provider-specific thinking token budget. Omitted by default for OpenAI-compatible endpoint portability. |
| `--judge-include-conversation` | Include bridge transcript turns in the judge input. Disabled by default because interrupted/cross-talk segments can make these turns noisy. |
| `--judge-compact-context` | Compact agent/user context histories before sending them to the judge. Disabled by default. |
| `--judge-context-message-limit <count>` | Maximum context messages to send when `--judge-compact-context` is enabled. |
| `--judge-context-system-string-limit <chars>` | Maximum system-message content length when `--judge-compact-context` is enabled. |
| `--judge-context-string-limit <chars>` | Maximum non-system string length when `--judge-compact-context` is enabled. |
| `--min-agent-turns <N>` | **Default: 3.** Scenarios with fewer completed agent turns are **counted as failures** in the composite success rate and skipped in the per-signal rates; on `--resume` they are also treated as in-flight. Pass `0` to disable. See [Notes](#notes). |

If neither `--scenarios` nor `--domain` is given, all registered scenarios run.

### Bot server environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `SERVER_CONFIG_PATH` | Path to the YAML server config | `server_configs/agent.yaml` / `server_configs/user.yaml` |
| `SERVER_HOST` | Host to bind | `0.0.0.0` |
| `WEBSOCKET_PORT` | Pipecat WebSocket port | `8765` (agent) / `8766` (user) |

## Available Domains and Scenarios


**Primary benchmarks** 

| Domain | Count | Summary tool | Description |
|--------|-------|--------------|-------------|
| `eva_airline` | 50 | bridge-pulled (no LLM summary tool) | SkyWay Airlines voice agent — flight changes, IRROPS, refunds, vouchers. Full 15-tool eva surface ported from [ServiceNow/eva](https://github.com/ServiceNow/eva/tree/0.1.3) (MIT). See [eva_airline domain notes](#eva_airline-domain-notes) below. |
| `tau2_airline` | 50 | bridge-pulled | Airline reservation flows (cancel / refund / rebook / upgrade) ported from [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench/tree/voice-user-sim-v1.0) (MIT). 13 LLM-callable agent tools + DB-state hash scoring. See [tau2 domain notes](#tau2-domain-notes) below. |
| `tau2_retail` | 114 | bridge-pulled | Online retail customer service — order cancel / exchange / return, address changes, partial-fulfillment edge cases. Same upstream source. 16 agent tools. 40 of 114 tasks also carry `nl_assertions` natural-language claims judged by the LLM. |
| `tau2_telecom` | 114 | bridge-pulled | Telecom tech support — mobile data troubleshooting, MMS issues, line suspension + payment recovery. **First dual-side domain**: the user-sim has its own 30 phone-control tools (toggle airplane mode, run speed test, etc.) alongside the agent's 13 lookup/billing/line-management tools. Cross-side state-propagation pipeline mirrors upstream's `Environment.sync_tools()`. Uses the `manual` policy variant. |
| `tau2_telecom_workflow` | 114 | bridge-pulled | Same 114 telecom tasks but the agent receives `tech_support_workflow.md` (procedural step-by-step) instead of `tech_support_manual.md`. Pure A/B knob over policy prose — every other framework piece is shared with `tau2_telecom`. |
| `restaurant` | 11 | `PlaceOrderTool`, `JoinWaitListTool` / `DropWaitListTool` | Ordering food at pizza, burger, and deli restaurants, plus a waitlist join/drop scenario (demonstrates shared state across tools). Integration example for the in-repo scenario authoring pattern. |
| `customer_service` | 10 | `ResolveTicketTool` | TechCorp customer service — billing disputes, order delays, defective returns, plan upgrades, account access, warranty claims, subscription cancellations, wrong items, and service outages. In-repo smoke set. |
| `qa` | 10 | `SaveQuestionAnswerTool` | Single-turn Q&A — geography, math, science, history, literature, weather (uses `GetCityWeatherTool`), and general knowledge. In-repo smoke set. |

(eva_airline + tau2_*) come first — these are the ported, externally-comparable evaluation sets. The simpler in-repo domains (`restaurant` / `customer_service` / `qa`) below them serve as smoke tests + integration examples, not headline benchmarks.

Run `python run_evaluation.py --list` for the full list of scenario names, or `--list-domains` for just the domain summary.

### eva_airline domain notes

The `eva_airline` domain is the first port from an external scenario library and introduces a few patterns worth knowing:

- **Symmetric inline DB transfer.** Each scenario's `setup_shared_state` writes the full scenario DB content (`nemo_voice_agent/evaluation/data/eva_airline/{eva_id}.json`) into `state["db"]`. The bridge serializes this into the `shared_state_init` payload of the `apply_initialization` RTVI message and the bot server uses it as-is — no filesystem coupling on the server side. At end-of-scenario the bridge pulls the (mutated) DB back via `get_scenario_summary`. Full content travels both ways. (A path-based fallback in the action handler remains for any future domain whose fixture is too large to ship inline — see [`nemo_voice_agent/evaluation/data/README.md`](../nemo_voice_agent/evaluation/data/README.md).)
- **Auto-aggregated action records, bridge-pulled.** Write tools subclass `WriteAirlineTool` and call `self._record_action(...)` on success, populating `shared_state["actions"]`. There is **no LLM-callable summary tool**; the bridge pulls `{"actions": ..., "db": ...}` at end-of-scenario via the `get_scenario_summary` RTVI client message (mirrors `get_context_history`). This eliminates summary-tool failure modes (forget-to-call, double-call, mid-conversation call) and a class of hallucinations (mis-formatted numbers, dropped fields, wrong enum values). Scoring measures what tools actually did.
- **DB-state hash matching (path-independent scoring).** Each scenario binds to an `expected_scenario_db` via a `cached_property` that reads from `eva_airline_dataset.jsonl`'s `ground_truth.expected_scenario_db` field. The bridge pulls the post-run DB; the runner SHA-256-hashes both states and compares. Path-independent: any sequence of agent actions that lands in the right end state passes. Verified empirically on `eva_airline__voluntary_date_change` — a canonical happy path produces a DB whose hash matches eva's expected state exactly. Action-list comparison still runs alongside as a separate signal — useful when you want to specifically score "did the agent perform action X" vs. "did the world end up in state Y". Scenarios that don't mutate state can opt out of DB-state scoring by setting `expected_scenario_db = None`. (Hash utility adapted from eva's `task_completion` metric.)
- **Single-source-of-truth metadata.** Each scenario subclass declares only `eva_id`. `current_date` is a `cached_property` derived from the bound JSON's `_current_date` — no manual mirror, no drift.
- **`disallow_extra_items` opt-in.** Off by default for airline scenarios (lenient — agent extras like reverted-then-redone rebooks pass). Enable per-scenario for clean-path runs where first-attempt correctness matters.
- **ASR speakability.** Confirmation numbers (e.g., `ZK3FFW`) and flight numbers (e.g., `SK703`) round-trip poorly through ASR/TTS. The user persona is instructed to spell every character (e.g., "Z, K, three, F, F, W"), and the agent guideline forbids reading internal journey IDs aloud.
- **Tuning `max_duration`.** `EvaAirlineBaseScenario.max_duration = 900` (15 minutes) is the domain default; the `--duration` CLI flag overrides per run. Voice round-trips are slow (~30–40s/turn on a healthy run), so leave plenty of headroom. The closing protocol alone (confirm + ask anything else + goodbye) costs 3–4 turns after the work is done.
- **Known STT limitations.** Parakeet STT mis-recognizes spelled-out alphanumerics (homophones like "four" / "for", letter sequences sometimes collapsed into a single word). When a scenario fails, compare the user-sim's text in `bot_logs_user/llm_context.json` against the agent-side STT output in `bot_logs_agent/llm_context.json` to distinguish user-simulator prompt-following failures from voice-pipeline accuracy issues.
- **Attribution.** Tool function bodies and Pydantic param models are adapted from [ServiceNow/eva](https://github.com/ServiceNow/eva/tree/0.1.3) (MIT). Inline `# Adapted from ...` comments are present at each ported block; see [`nemo_voice_agent/evaluation/data/README.md`](../nemo_voice_agent/evaluation/data/README.md) for the full source/license inventory.

### tau2 domain notes

The `tau2_airline`, `tau2_retail`, `tau2_telecom`, and `tau2_telecom_workflow` domains are ports of [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench/tree/voice-user-sim-v1.0) (commit `17e07b1`, MIT-licensed). They share patterns introduced by eva_airline but add several new ones; the highlights:

- **Path-based DB transfer (vs. eva's inline).** Tau2 DBs are large (the agent-side `db.json` for telecom is ~7 MB — well over pipecat's 1 MB WebSocket frame cap). `Scenario.setup_shared_state` writes a relative `db_path` into `state`; the bot's `apply_initialization` RTVI handler resolves it against `EVAL_DATA_ROOT` and loads the JSON before tools instantiate. Inbound is hash-only (`get_scenario_summary` returns `{actions, db_hash}` by default). Predicate evaluation is the one exception — when `scenario.db_state_assertions` is truthy, the bridge sets `include_db=True` and the bot inlines the (smaller, telecom-only) `user_db` so the runner can call predicate functions on it.
- **Dual-side architecture (telecom only).** `Tau2TelecomBaseScenario.has_user_state=True` activates a parallel user-side `TelecomUserDB` (mock phone state) seeded from `user_db.json`. The user-sim's LLM has 30 phone-control tools (toggle airplane mode, run speed test, check status bar, etc.). The bridge dual-pulls both sides' state at end-of-scenario; the merged action list carries `side="agent"` / `side="user"` tags.
- **Cross-side state-propagation (telecom only).** When the agent's `refuel_data` mutates `line.data_refueling_gb` (agent DB), the user-sim's `surroundings.mobile_data_usage_exceeded` (user DB) needs to flip — but the two bots run in separate processes. Bridge implements an in-process shadow + per-action `action-applied` RTVI event + `apply_sync_delta` push-back mechanism that mirrors upstream's `Environment.sync_tools()` across the three invocation points (post-init, per-write-action). See `evaluation/sync_appliers.py` + `tools/tau2_telecom_sync.py`.
- **Policy variants (telecom only).** Each base-split telecom task is registered TWICE: `tau2_telecom__X` (uses `tech_support_manual.md`) and `tau2_telecom_workflow__X` (uses `tech_support_workflow.md`). Same task, same reference actions, same predicates — only the agent's policy text differs. Pure A/B knob over policy prose.
- **Per-domain agent-prompt addenda (telecom only).** Three blocks appended after the upstream-verbatim policy: a tool-availability disclaimer (so the LLM doesn't hallucinate user-side tool names that the agent's tool surface doesn't include), a stay-on-task guideline, and a home-network/location-probe rule that tells the agent the telecom is US-based and to ask the user's location first when diagnosing connectivity issues. See `nemo_voice_agent/evaluation/scenarios/data/tau2_telecom/base.py` for the constants.
- **Multi-signal scoring.** Tau2 domains opt into more scoring signals than eva_airline alone — see [Evaluation Metrics](#evaluation-metrics) below for the full signal model and how the composite `is_successful` is derived as a strict conjunction across applicable signals.
- **Scaffold generators are committed.** `scripts/prepare_tau2_data/generate_{airline,retail,telecom}_scaffolds.py` regenerate `nemo_voice_agent/evaluation/scenarios/data/tau2_*/group_*x.py` from upstream `tasks.json` + `split_tasks.json[base]`. Run them only when the upstream schema or scaffold template changes. The telecom generator emits paired manual+workflow classes per upstream task.
- **Data import script.** `scripts/prepare_tau2_data/prepare_telecom.py` does TOML→JSON conversion for the telecom upstream (airline/retail ship as JSON; only telecom needs conversion). See `scripts/prepare_tau2_data/README.md` for details.

## Evaluation Metrics

The runner emits up to **six orthogonal scoring signals** per scenario, plus a composite `is_successful` derived from them via strict conjunction. The canonical list is the `SuccessSignal` enum in `nemo_voice_agent/evaluation/scenarios/classes.py`. Each signal is independently opt-in at the scenario level — most domains use 2–3, telecom uses 4 plus the judge — except **clean exit, which is in every domain's whitelist** (see [Composite `is_successful`](#composite-is_successful-per-scenario-whitelist)). All six (and the composite) land in `metrics.json` per scenario and roll up in `all_summary.txt` per run.

### Signal matrix

| Signal | Type | Path-dependent? | Opted in by | `metrics.json` field |
|---|---|---|---|---|
| Action-list match | `bool \| "N/A"` | **Yes** (action sequence) | Any scenario with a `reference_answer` | `is_action_match` |
| DB-state hash match | `bool \| "N/A"` | No (end state only) | eva_airline + all tau2 domains | `db_state_match` |
| DB-state assertions | float ∈ [0, 1] | No | tau2_telecom (6 predicates) | `db_state_assertion_pass_rate` |
| NL assertions | float ∈ [0, 1] | No (judged on transcript) | tau2_retail (40/114 tasks), tau2_telecom (a subset) | `nl_assertion_pass_rate` |
| LLM judge score | float ∈ [0, 1] | — | Any run with `--judge-url` | `judge_score` (+ `judge_passed` when `--judge-threshold` set) |
| **Clean exit** | `bool` | No | **Every domain** — closure discipline is universal | `clean_exit` (+ `stop_reason`) |
| **Composite (is_successful)** | `bool \| "N/A"` | — | Always computed | **`is_successful`** + `success_breakdown` |

### 1. Action-list match (deterministic)

`check_if_task_success` performs a recursive comparison between each scenario's `reference_answer` and the agent's bridge-pulled action list (or its `<final_response>` for legacy scenarios that still use an LLM-callable summary tool):

- **Dict vs. Dict** — every key/value in the reference must be present and match in the prediction. Extra keys in the prediction are allowed.
- **Dict vs. List-of-Dicts** — the reference dict must match the **last** dict in the prediction list.
- **List-of-Dicts vs. List-of-Dicts** — every dict in the reference must find a matching dict in the prediction (order-independent, each prediction can match at most one reference).

String matching respects the scenario's `ignore_capitalization`, `ignore_punctuation`, and `clean_text` flags. Numeric values are compared with `np.isclose`. Set `disallow_extra_items=True` on the scenario (or pass `--strict-match` globally) to require length-match between lists.

**Path-DEPENDENT.** Two scenarios that land on the same final DB state but via different tool sequences will score differently — useful when you specifically want to verify the agent took the "right" steps.

Saved as `metrics.json["is_action_match"]` (bool, or `"N/A"` when no reference exists).

### 2. DB-state hash match (deterministic, path-independent)

Each scenario can declare an `expected_scenario_db` `cached_property` (a dict). The bot computes `get_dict_hash(state["db"])` inside the `get_scenario_summary` RTVI handler and returns the SHA-256 string only; the runner computes `get_dict_hash(scenario.expected_scenario_db)` and compares.

**Path-INDEPENDENT.** Any sequence of agent actions that lands in the right end state passes. Useful for open-ended scenarios where multiple action sequences are all valid.

The hash module is canonical (lives at `evaluation/db_hash.py`) and imported by both bridge and bot side, so the canonicalization (float `1.0 → 1`, `"none" → None`, `ORDER_INDEPENDENT_LIST_FIELDS`, `HASH_EXCLUDED_KEYS = {"session"}`) is identical.

For telecom (dual-side), the bridge dual-pulls both `db_hash` (agent) and `user_db_hash` (user-sim); a scenario passes both hash checks for `db_state_match=True`.

Saved as `metrics.json["db_state_match"]` (bool, or `"N/A"` when scenario doesn't define `expected_scenario_db`).

### 3. DB-state assertions (deterministic, per-predicate)

Each scenario can declare a `db_state_assertions: List[dict]` `cached_property` — a list of `{func_name, arguments, assert_value, side}` records evaluated against the pulled post-run DB. Each predicate is a pure `(db: dict, **arguments) -> bool` function registered flat under `(domain, func_name)` in `evaluation/db_state_predicates.py`. Side-aware dispatch: `side="user"` runs on `user_db`, `side="agent"` on `db`.

Useful when **multiple end states** all satisfy the same outcome predicates — e.g., "data refueling amount ≥ 2 GB on line L1002" doesn't care which specific bill ID got the corresponding charge, so the whole-DB hash match can fail (legitimate variation) while the predicate still passes.

Saved as `metrics.json["db_state_assertion_pass_rate"]` (float in [0, 1]) plus `metrics.json["db_state_assertion_verdicts"]` (one entry per predicate, `{func_name, side, passed, actual, error}`). Currently used by `tau2_telecom`.

### 4. NL assertions (LLM-judged, per-assertion)

Each scenario can declare a `nl_assertions: List[str]` `cached_property` — natural-language claims about what the conversation must demonstrate (e.g., *"Agent should tell the user that there are 10 t-shirt options available."*). When the LLM judge runs, it extends its response with `nl_assertion_verdicts` (one entry per assertion, `{index, passed, reason}`).

Used by `tau2_retail` (40 of 114 tasks) and `tau2_telecom` (a subset of tasks ported from upstream's voice-user-sim-v1.0).

Saved as `metrics.json["nl_assertion_pass_rate"]` (float in [0, 1]) plus the per-assertion verdicts in `judge_result.json["nl_assertion_verdicts"]`. Only populated when both the scenario has `nl_assertions` AND the judge ran.

### 5. LLM judge (soft overall signal)

When `--judge-url` and `--judge-model` are provided, `LLMJudge.judge_scenario` scores each scenario on a 0–1 scale. The judge receives:

- `<reference>` — the gold action list / DB outcome, when available.
- `<prediction>` — the agent's recorded actions, when available.
- `<conversation>` — the transcribed user/agent turns when `--judge-include-conversation` is set.
- `<agent_context_history>` — the agent bot's LLM context (system prompt + every turn + every tool call).
- `<user_context_history>` — the user-sim bot's LLM context (essential for dual-side domains where reference actions are performed by the user-sim, not the agent).
- `<nl_assertions>` — numbered list when the scenario declares them.

The judge prompt is **side-aware**: it knows about `side="user"` vs `side="agent"` reference actions and scores accordingly (user-side actions are scored on the agent's guidance quality, not on whether the agent invoked the tool itself — since user-side tools aren't in the agent's surface). The runner no longer requires both `reference_answer.json` and `final_agent_response.json` before invoking the judge; if context history, natural-language assertions, or opted-in transcript turns are available, the judge scores that evidence. The full prompt + user content is preserved in `judge_result.json["judge_input"]` for triage.

Provider-specific judge parameters are opt-in. In particular, `thinking_token_budget` is only sent when `--judge-thinking-token-budget` is provided, because some hosted OpenAI-compatible endpoints reject unknown fields.

Context compaction is also opt-in. By default the judge receives full `llm_context.json` histories. Pass `--judge-compact-context` to enable truncation, and override the fallback limits with the `--judge-context-*` flags when needed.

Saved per scenario as:
- `metrics.json["judge_score"]` — raw 0–1 float (always set when judge ran).
- `metrics.json["judge_passed"]` — bool (`score >= --judge-threshold`). Only set when both `--judge-url` and `--judge-threshold` are provided.

The judge catches presentation issues and fabrications that deterministic comparators miss — but it's noisy enough that running it alongside deterministic signals (rather than instead of) gives the most reliable verdict.

### External agent compatibility

External agents can be evaluated as long as they speak the pipecat WebSocket transport protocol and register the evaluator-compatible RTVI actions used by the bridge. See [`EXTENDING_PIPELINE.md`](EXTENDING_PIPELINE.md) for the canonical pipeline contract, including scenario initialization, context retrieval, scenario summary, sync, and reset actions.

Agents may also write optional architecture-specific diagnostics to either `trace_metrics.json` or `bot_logs_agent/trace_metrics.json` inside the scenario directory. The runner copies this into `metrics.json["trace_metrics"]` without interpreting it. This keeps cascaded-agent diagnostics, such as internal handoff quality, available without hard-coding architecture-specific metrics into the evaluator.

### Composite `is_successful` (per-scenario whitelist)

`metrics.json["is_successful"]` is the strict-AND of every **applicable** signal in the scenario's `success_signals` whitelist. Each domain declares which signals gate its verdict — see the `SuccessSignal` enum in `nemo_voice_agent/evaluation/scenarios/classes.py` for the canonical list (`ACTION_MATCH`, `DB_STATE_MATCH`, `DB_STATE_ASSERTION`, `NL_ASSERTION`, `JUDGE_PASSED`, `CLEAN_EXIT`).

**Per-domain `success_signals` matrix** (the authoritative wiring lives on each domain's base scenario):

| Domain | Gating signals | Rationale |
|---|---|---|
| `eva_airline` | `DB_STATE_MATCH` + `CLEAN_EXIT` | Gold expected DB shipped per scenario; path-independent. |
| `tau2_airline` | `DB_STATE_MATCH` + `CLEAN_EXIT` | Same — derived expected DB; no judge dependency. |
| `tau2_retail` | `DB_STATE_MATCH` + `NL_ASSERTION` (when set) + `CLEAN_EXIT` | 40/114 tasks opt into NL claims; derived per-scenario. |
| `tau2_telecom` / `_workflow` | `DB_STATE_ASSERTION` + `NL_ASSERTION` (when set) + `CLEAN_EXIT` | Open solution space — whole-DB hash and action match are informational. |
| `restaurant`, `customer_service`, legacy `fastbite` | `ACTION_MATCH` + `CLEAN_EXIT` | Single canonical `reference_answer` (structured order / ticket / receipt) checked recursively. |
| `qa`, legacy `simple_qa_*` | `JUDGE_PASSED` + `CLEAN_EXIT` | Free-form text answers — LLM judge is the only principled signal. |

**`CLEAN_EXIT` is in every domain's whitelist.** A scenario passes only when the agent called `EndConversationTool` voluntarily (stop reason `[EXIT]`); `[TIMEOUT]` always fails. Closure discipline is universal — an agent that does the right work but then never stops talking is not a successful agent (real callers would have hung up, and a 1800 s "successful" TIMEOUT consumes 5-10× the compute of a clean exit). For the cases where another gating signal already failed, `CLEAN_EXIT` is redundant; it only changes the verdict when the outcome was right but closure was wrong. Most notably this catches **policy-refusal scenarios** where `expected_scenario_db == initial_scenario_db` and an agent that crashed at greeting would otherwise pass `DB_STATE_MATCH` "by inaction." A regression test (`test_every_concrete_scenario_includes_clean_exit`) prevents future domains from silently omitting it.

**Principle:** `JUDGE_PASSED` gates only when no deterministic alternative exists. `NL_ASSERTION` still runs through the judge but contributes per-assertion verdicts, so it can gate when opted in. `ACTION_MATCH` is appropriate when the domain has a single canonical correct `reference_answer` (either a structured outcome summary OR a tool-call trajectory where only one sequence is considered successful); it's not appropriate when the solution space is open (multiple valid trajectories satisfying the same outcome — telecom-style). For domains that ship an `expected_scenario_db` (eva, tau2 airline, tau2 retail), we prefer `DB_STATE_MATCH` over `ACTION_MATCH` because it's path-independent — any sequence landing on the correct end state passes.

If none of the whitelisted signals are applicable for a given scenario (e.g. a QA run without `--judge-url`), `is_successful = "N/A"`. The runner emits a warning at run start when any queued scenario references `JUDGE_PASSED` / `NL_ASSERTION` but the judge isn't configured.

**`success_breakdown`** in `metrics.json` carries four buckets:

- `passed` — whitelisted signals that returned `True`.
- `failed` — whitelisted signals that returned `False` (these dragged the verdict down).
- `not_applicable` — whitelisted signals that couldn't be evaluated for this scenario.
- `excluded` — signals computed and saved but NOT in this scenario's whitelist (informational; useful for spotting "all gating signals passed but the agent took an unusual path").

**Strict thresholds for float signals.** The float-valued pass rates (`db_state_assertion_pass_rate`, `nl_assertion_pass_rate`) must equal exactly `1.0` for the composite to pass. Rationale: every assertion is supposed to be true. A 95% pass rate means one assertion failed — that's a real defect to investigate, not noise to round away.

**Authoring a new domain.** Concrete scenarios MUST declare a non-empty `success_signals` on themselves or an ancestor base — `Scenario.__init_subclass__` raises `TypeError` at class-definition time if a scenario with `name` set has nothing resolvable. For mixed-composition domains (some tasks have NL assertions, some don't), use a `cached_property` that derives from `self.nl_assertions` rather than per-scenario declarations. See [`EXTENDING_DATA.md`](EXTENDING_DATA.md).

The per-scenario whitelist plus the scenario's `expected_db_hash` / `expected_user_db_hash` / `db_state_assertions` / `nl_assertions` / `initialization_actions` are also written to `scenario_config/metadata.json` per scenario, so an old eval run remains fully interpretable without re-loading the scenario class.

### Run-level aggregation

`all_summary.txt` shows the composite headline plus the per-signal breakdown so a 60% `Overall Success Rate` doesn't hide that 4/5 deterministic signals are at 100%:

```
Overall Success Rate: 60.00% (3/5 scenarios — strict conjunction across all applicable signals)

Per-signal pass rates:
  Action-list match:           100.00% (5/5 scenarios)
  DB-State match:               80.00% (4/5 scenarios)
  DB-State-Assertion pass:     100.00% (3/3 predicates)
  NL-Assertion pass:            75.00% (3/4 assertions)
  Judge score mean:              0.880 (across 5 scenarios)
  Judge passed (>= threshold):  80.00% (4/5 scenarios)
```

Per-domain breakdowns (`Per-Domain Success Rate`, etc.) print when a run spans more than one domain.

## Output Structure

Each run creates a timestamped session directory. Within it, each scenario has its own subdirectory.

```
eval_results/eval_YYYYMMDD_HHMMSS/
├── evaluation_log.txt              # Top-level runner log
├── all_metrics.json                # Aggregated metrics across all scenarios
├── all_latencies.csv               # Every latency measurement as CSV rows
├── all_summary.txt                 # Human-readable summary (per-scenario + overall stats)
└── <scenario_name>/                # One directory per scenario
    ├── conversation_log.txt        # Timestamped transcript with latency annotations
    ├── conversation_log.seglst.json  # segLST-format speaker segments
    ├── conversation_log.wav        # Stereo audio: L=user→agent, R=agent→user
    ├── bridge_log.txt              # Bridge debug/info log
    ├── final_agent_response.json   # Bridge-pulled action list (or legacy <final_response> payloads)
    ├── final_scenario_db_hash.txt   # Post-run DB hash(es): db_hash + user_db_hash (if applicable)
    ├── metrics.json                # Per-scenario metrics — see Evaluation Metrics above for the full field list
    ├── judge_result.json           # LLM judge output (present only when --judge-url is set)
    ├── scenario_config/            # Snapshot of the scenario definition used for this run
    │   ├── metadata.json           # name, description, max_duration, matching flags, noise config
    │   ├── reference_answer.json   # The expected answer that was compared against
    │   ├── user_prompt.txt         # Rendered system prompt for the user bot
    │   ├── user_tools.json         # Tool config sent to the user bot
    │   ├── agent_prompt.txt        # Rendered system prompt for the agent bot
    │   └── agent_tools.json        # Tool config sent to the agent bot
    ├── bot_logs_user/
    │   └── llm_context.json        # Full LLM context history retrieved from the user bot
    └── bot_logs_agent/
        └── llm_context.json        # Full LLM context history retrieved from the agent bot
```

Key files to inspect:
- **`metrics.json`** — turn count, duration, latency stats (mean/P50/P95/min/max), per-turn latencies, every signal under [Evaluation Metrics](#evaluation-metrics) (`is_action_match`, `db_state_match`, `db_state_assertion_pass_rate`, `nl_assertion_pass_rate`, `judge_score`, `judge_passed`), the composite `is_successful`, and the `success_breakdown` of which signals passed / failed / weren't applicable.
- **`final_agent_response.json`** — what the agent (and, for telecom, the user-sim) actually did. Bridge-pulled action records with `side` tags for dual-side domains.
- **`judge_result.json`** — when judge ran: `score`, `reason`, per-assertion `nl_assertion_verdicts`, and `judge_input` (the verbatim prompt + user content) for triage.
- **`conversation_log.wav`** — listen to the actual conversation.
- **`bot_logs_{user,agent}/llm_context.json`** — full LLM conversation including tool calls and results, useful for debugging agent behavior. For dual-side telecom scenarios, both files are essential — user-sim's tool calls (`toggle_data`, etc.) live on the user side, not the agent side.

## Extending the System

Two dedicated guides cover the two axes of extension. Pick the one that matches what you're changing:

**[`EXTENDING_DATA.md`](EXTENDING_DATA.md) — the data layer.** How to add scenarios, tools, and whole domains. Covers:

- **Extension points** — at-a-glance index of new-scenario / new-tool / new-domain workflows.
- **Scenario Structure** — the 8 per-side properties, scenario-level fields (`domain`, `reference_answer`, `expected_scenario_db`, `db_state_assertions`, `nl_assertions`, `initialization_actions`, ...), and the domain base-class pattern.
- **Creating a New Scenario** — pick or create a domain, subclass the base, verify with `run_evaluation.py --list`.
- **Tool System** — tool configuration, shared state, the two termination contract patterns (bridge-pull vs legacy LLM-summary), and the `@register_schema_tool_for_eval(domain="...")` registration flow.

**[`EXTENDING_PIPELINE.md`](EXTENDING_PIPELINE.md) — the bot pipeline layer.** How to swap models, add a custom processor, or build a whole new pipecat pipeline. Covers:

- **Tier 1 — YAML swap** — different LLM / TTS / STT model via `server_configs/*.yaml`, no Python.
- **Tier 2 — Custom processor** — insert a `FrameProcessor` (e.g., Markdown sanitizer between LLM and TTS).
- **Tier 3 — Whole new pipecat pipeline** — replace `run_bot_websocket()` entirely. The eval-compatibility contract is narrow: a pipecat WebSocket transport plus an `RTVIProcessor` registering six required actions. Everything else (services, processors, pipeline shape) is your choice.


## Notes

- The pipecat version used in this evaluation system is 1.6.0. Custom bot control uses the RTVI
  client-message / server-response pattern (`on_client_message`); the pre-1.0 `RTVIAction` /
  `register_action` API no longer exists.

- **Resuming after server hangs (`--min-agent-turns`).** When the agent LLM server hangs mid-scenario (e.g. LLM API server stops responding after the first agent turn), scenarios time out after only one agent turn (the greeting). These look like failures in the summary but the root cause is infrastructure, not actual agent quality. `--min-agent-turns` scores them as failures in the composite rate, skips them in the per-signal rates, and automatically re-runs them on `--resume`.

  **This is ON by default, at `--min-agent-turns 3`.** Scenarios in which the agent completed fewer than 3 turns are already being recorded as hard failures. Pass `--min-agent-turns 0` to disable the filter and score every scenario on its own merits:

  ```bash
  # Explicitly lower the threshold, or pass 0 to disable
  python run_evaluation.py \
      --user-url ws://localhost:8766 --agent-url ws://localhost:8765 \
      --domain tau2_retail \
      --resume 20260618_072325 \
      --min-agent-turns 2
  ```

  Scenarios below the threshold are recorded as `False` in the composite `is_successful` and task-success rates (`runner.py` appends `False` to the success buckets and returns early), and are skipped in the per-signal action-match / DB-state / NL-assertion / judge rates. Token usage is still rolled up so cost accounting stays accurate. On `--resume`, they are additionally treated as in-flight (moved aside and re-run), so a single resume command both cleans the aggregate and retries the stalled scenarios.

  Because the filter is on by default, **a stalled scenario silently depresses the headline success rate while shrinking the denominator of every per-signal rate.** Before comparing runs, check the `WARNING: N scenario(s) had fewer than 3 agent turn(s) ... and were counted as failures.` line in `all_summary.txt`.

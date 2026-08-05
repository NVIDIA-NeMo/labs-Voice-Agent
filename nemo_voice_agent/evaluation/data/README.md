# Evaluation Fixture Data

This directory holds scenario fixtures (databases, ground-truth metadata) consumed by
the voice-agent evaluation system. Files here are loaded into `shared_state` on the
bot server via `Scenario.setup_shared_state` (see
[`evaluation/README.md`](../../../evaluation/README.md)). This directory is the
default fixture root — `get_eval_data_root()` returns it unless `EVAL_DATA_ROOT`
overrides it.

Per-domain data is grouped under a domain-prefixed subdirectory or filename so
that scenarios from different upstream libraries don't collide.

## Sources & Licenses

### eva (`eva_airline_*`)

- **Upstream**: [github.com/ServiceNow/eva](https://github.com/ServiceNow/eva)
- **Version**: `0.1.3`
- **License**: MIT
- **Contents** (verbatim copy, no local modifications):
  - `eva_airline/{eva_id}.json` (50 files) — scenario databases (`reservations`,
    `journeys`, `disruptions`, etc.). Each file is a self-contained world state
    keyed by an eva scenario ID like `1.1.2.json`. Source: `data/airline_scenarios/`.
  - `eva_airline/eva_airline_dataset.jsonl` (50 lines) — scenario metadata (`user_goal`,
    `decision_tree`, `information_required`, `ground_truth.expected_scenario_db`).
    Used to translate eva scenarios into NeMo `Scenario` subclasses. Source:
    `data/airline_dataset.jsonl`.
- **Bound code**: `nemo_voice_agent/evaluation/scenarios/data/eva_airline/` (package: `base.py` holds the `EvaAirlineBaseScenario` + 5 hand-authored seed scenarios; `group_Nx.py` modules carry the auto-scaffolded scenarios for each eva sub-flow)
  + `nemo_voice_agent/evaluation/tools/eva_airline_tools.py` +
  `eva_airline_params.py`. Each ported code file carries an inline
  `# Adapted from https://github.com/ServiceNow/eva/tree/0.1.3` attribution.

### tau2-bench (`tau2_airline/`)

- **Upstream**: [github.com/sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
- **Version**: tag `voice-user-sim-v1.0` → commit `17e07b1` (the annotated tag's
  metadata anchors it as "the voice user simulator as used in the τ³-bench
  1.0.0 release"; the tag-object SHA `d1eff9e6` is the annotated-tag wrapper,
  not a commit SHA — dereference it with `git rev-parse voice-user-sim-v1.0^{commit}`).
- **License**: MIT (Sierra Research, 2025)
- **Contents** (verbatim copy, no local modifications):
  - `tau2_airline/db/` (sharded, ~4.7 MB total) — the full shared airline DB
    split into one file per top-level table (`flights.json`, `users.json`,
    `reservations.json`). The single upstream `db.json` is ~6.8 MB which
    exceeds the GitLab mirror's 5 MB per-file cap; the sharded layout
    reassembles into a byte-identical in-memory dict via
    `nemo_voice_agent.evaluation.load_db_artifact`, so DB hashes and gold
    replays are unchanged. Re-shard with
    `python scripts/prepare_tau2_data/shard_db.py <path/to/db.json>` after
    a fresh upstream pull. Used as the seeded state for every airline
    scenario; mutations land in a per-scenario deep copy. Source:
    `data/tau2/domains/airline/db.json`.
  - `tau2_airline/tasks.json` (50 tasks) — task definitions
    (`description`, `user_scenario`, `initial_state`, `evaluation_criteria`,
    `annotations`). Used to derive expected_scenario_db + reference_answer
    via the runtime gold-env replay on `Tau2BaseScenario._gold_replay`.
    Source: `data/tau2/domains/airline/tasks.json`.
  - `tau2_airline/tasks_voice.json` — voice-eligible id list + per-task
    `persona_name` (control preset). Filtered + joined with `tasks.json`
    via `_load_tau2_voice_task_index("airline")`. Source:
    `data/tau2/domains/airline/tasks_voice.json`.
  - `tau2_airline/split_tasks.json` — split membership; `base` (50 ids =
    train ∪ test) is the default eval surface for airline.
    Source: `data/tau2/domains/airline/split_tasks.json`.
  - `tau2_airline/policy.md` — agent system prompt (full markdown). Loaded
    once per process by `Tau2AirlineBaseScenario.policy` and returned
    verbatim from `get_agent_prompt()` so our scores stay comparable to
    tau2's published voice-leaderboard numbers. Source:
    `data/tau2/domains/airline/policy.md`.
  - `tau2_airline/audio_difficulty.json` — kept for traceability;
    **not consumed** by the eval pipeline (tau2's persona acoustic stack
    is out of scope — see plan §1 non-goal).
- **Bound code**: `nemo_voice_agent/evaluation/scenarios/data/tau2_airline/`
  (package: `base.py` holds `Tau2AirlineBaseScenario` + hand-authored seeds;
  `group_Nx.py` modules carry auto-scaffolded scenarios) +
  `nemo_voice_agent/evaluation/tools/tau2_airline_tools.py` +
  `tau2_airline_params.py`. Each ported code file carries an inline
  `# Adapted from https://github.com/sierra-research/tau2-bench/tree/voice-user-sim-v1.0`
  attribution.

### tau2-bench (`tau2_retail/`)

- **Upstream**: [github.com/sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
- **Version**: tag `voice-user-sim-v1.0` → commit `17e07b1` (same pin as
  `tau2_airline/` — see that section for the tag-object/commit distinction).
- **License**: MIT (Sierra Research, 2025)
- **Contents** (verbatim copy, no local modifications):
  - `tau2_retail/db.json` (~2.8 MB) — the full shared retail DB
    (`products`, `users`, `orders`). Used as the seeded state for every
    retail scenario; mutations land in a per-scenario deep copy. Source:
    `data/tau2/domains/retail/db.json`.
  - `tau2_retail/tasks.json` (114 tasks) — task definitions
    (`description`, `user_scenario`, `initial_state`, `evaluation_criteria`
    including `nl_assertions`, `annotations`). 73 tasks are actions-only,
    39 carry both `actions` and `nl_assertions`, 1 is nl-only, 1 is a
    chitchat scenario with neither (task 57). Source:
    `data/tau2/domains/retail/tasks.json`.
  - `tau2_retail/tasks_voice.json` — voice-eligible id list + per-task
    `persona_name` (control preset). Source:
    `data/tau2/domains/retail/tasks_voice.json`.
  - `tau2_retail/split_tasks.json` — split membership; `base` (114 ids =
    train 74 ∪ test 40) is the default eval surface for retail. Source:
    `data/tau2/domains/retail/split_tasks.json`.
  - `tau2_retail/policy.md` — agent system prompt (full markdown). Loaded
    once per process by `Tau2RetailBaseScenario.policy`. Source:
    `data/tau2/domains/retail/policy.md`.
  - `tau2_retail/audio_difficulty.json` — kept for traceability;
    **not consumed** by the eval pipeline.
  - `tau2_retail/task_issues/` — upstream task-level annotations
    (3 files). Kept for traceability; not consumed by the pipeline today.
- **Bound code**: `nemo_voice_agent/evaluation/scenarios/data/tau2_retail/`
  (`base.py` + `group_Nx.py` modules) +
  `nemo_voice_agent/evaluation/tools/tau2_retail_tools.py` +
  `tau2_retail_params.py`. Each ported code file carries an inline
  `# Adapted from https://github.com/sierra-research/tau2-bench/tree/voice-user-sim-v1.0`
  attribution.

### tau2_telecom (`tau2_telecom__*`)

- **Upstream**: [github.com/sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
- **Tag**: `voice-user-sim-v1.0` (commit
  [`17e07b1`](https://github.com/sierra-research/tau2-bench/commit/17e07b1da2bbc0cadfddeea36412686e0604127b))
- **License**: MIT (Sierra Research, 2025)
- **Imported via**: `scripts/prepare_tau2_data/prepare_telecom.py` — see that
  directory's README for re-import instructions. This script handles the
  TOML → JSON conversion for `db.toml` / `user_db.toml` (upstream telecom
  diverges from airline / retail's `.json`-only convention). Other files
  are copied verbatim.
- **Contents** (under `tau2_telecom/`):
  - `db.json` — agent-facing telecom DB (plans, devices, lines, customers,
    bills). Loaded once per process by `Tau2TelecomBaseScenario.db`. Source:
    `data/tau2/domains/telecom/db.toml` (converted via `tomllib` →
    re-serialized as indent=2 JSON).
  - `user_db.json` — user-facing telecom DB (mock phone device state +
    user surroundings). Loaded once per process by
    `Tau2TelecomBaseScenario.user_db`. Source:
    `data/tau2/domains/telecom/user_db.toml` (converted as above).
    The raw TOML does not carry every Pydantic-default field (e.g.
    `surroundings.signal_strength` per-network technology defaults), so
    `prepare_telecom.py` round-trips the converted JSON through the ported
    `TelecomUserDB` model to materialize them before downstream code reads it.
  - `tasks.json` — task definitions. **Filtered to the 114 base-split ids
    at import time** by `prepare_telecom.py` (~660 KB after filtering vs.
    ~14 MB upstream verbatim — 4.7% kept). The 2171 non-base task
    definitions are dead weight because our eval surface never references
    them. Hard-fails on import if any base id has no upstream definition.
  - `tasks_voice.json` — voice-eligibility entries (acoustic config +
    persona name per task). **Filtered to the 114 base-split ids at
    import time** by `prepare_telecom.py` (~3 MB after filtering, vs.
    ~62 MB upstream verbatim). Other splits (`small`, `train`, `test`,
    `full`) are stripped — the eval surface ships only `base`, so the
    other ~2171 entries are dead weight. Re-imports preserve this
    filter; the script hard-fails if any base id is missing from
    upstream.
  - `split_tasks.json` — base/small/train/test/full split-id lists
    (verbatim — small file, useful for debugging which scenario IDs
    upstream considered "curated"). We ship the `base` (114 ids) eval
    surface; `small` (20 ids, disjoint from base) is upstream-debug-only.
  - `audio_difficulty.json` — upstream-emitted audio difficulty annotations
    (verbatim). Kept for traceability; **not consumed** by the eval pipeline.
  - `main_policy.md`, `tech_support_workflow.md`, `tech_support_manual.md`
    — agent prompt sources (verbatim). `Tau2TelecomBaseScenario.policy`
    concatenates `main_policy.md` with the variant's
    `tech_support_{manual,workflow}.md`, separated by a `---` rule.
  - `workflows/` — 10 per-issue workflow markdown files referenced by
    `tech_support_workflow.md`.
- **Bound code**: `nemo_voice_agent/evaluation/scenarios/data/tau2_telecom/`
  + `nemo_voice_agent/evaluation/tools/tau2_telecom_tools.py`,
  `tau2_telecom_params.py`, `tau2_telecom_predicates.py`,
  `tau2_telecom_init_functions.py`. Each ported code file
  carries an inline
  `# Adapted from https://github.com/sierra-research/tau2-bench/tree/voice-user-sim-v1.0`
  attribution.

## Adding a new source

Add a new section above using the same fields. If the new source overlaps a
domain already present (e.g., a second airline-data library), namespace the
files with a distinct prefix (`tau_airline_*`, `cs_airline_*`, etc.) to avoid
collisions, and register a corresponding `Scenario` subclass that resolves
fixtures via `get_eval_data_root() / "<prefix>_scenarios"`.

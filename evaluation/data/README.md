# Evaluation Fixture Data

This directory holds scenario fixtures (databases, ground-truth metadata) consumed by
the voice-agent evaluation system. Files here are loaded into `shared_state` on the
bot server via `Scenario.setup_shared_state` (see
[`evaluation/README.md`](../README.md)). The directory is resolvable via the
`EVAL_DATA_ROOT` env var (defaults to this path).

Per-domain data is grouped under a domain-prefixed subdirectory or filename so
that scenarios from different upstream libraries don't collide.

## Sources & Licenses

### eva (`eva_airline_*`)

- **Upstream**: [github.com/ServiceNow/eva](https://github.com/ServiceNow/eva)
- **Version**: `0.1.3`
- **License**: MIT
- **Contents** (verbatim copy, no local modifications):
  - `eva_airline_scenarios/` (50 files) — scenario databases (`reservations`,
    `journeys`, `disruptions`, etc.). Each file is a self-contained world state
    keyed by an eva scenario ID like `1.1.2.json`. Source: `data/airline_scenarios/`.
  - `eva_airline_dataset.jsonl` (50 lines) — scenario metadata (`user_goal`,
    `decision_tree`, `information_required`, `ground_truth.expected_scenario_db`).
    Used to translate eva scenarios into NeMo `Scenario` subclasses. Source:
    `data/airline_dataset.jsonl`.
- **Bound code**: `nemo/agents/voice_agent/evaluation/scenarios/data/eva_airline/` (package: `base.py` holds the `EvaAirlineBaseScenario` + 5 hand-authored seed scenarios; `group_Nx.py` modules carry the auto-scaffolded scenarios for each eva sub-flow)
  + `nemo/agents/voice_agent/evaluation/tools/eva_airline_tools.py` +
  `eva_airline_params.py`. Each ported code file carries an inline
  `# Adapted from https://github.com/ServiceNow/eva/tree/0.1.3` attribution.

### tau2-bench (`tau2_airline/`)

- **Upstream**: [github.com/sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
- **Version**: tag `voice-user-sim-v1.0` → commit `17e07b1` (the annotated tag's
  metadata anchors it as "the voice user simulator as used in the τ³-bench
  1.0.0 release"; the tag-object SHA `d1eff9e6` is the annotated-tag wrapper,
  not a commit SHA — see `nemo_experiments/add_tau2_domains_plan.md` §1 + §7 item 8).
- **License**: MIT (Sierra Research, 2025)
- **Contents** (verbatim copy, no local modifications):
  - `tau2_airline/db.json` (~7 MB) — the full shared airline DB
    (`flights`, `users`, `reservations`, etc.). Used as the seeded state for
    every airline scenario; mutations land in a per-scenario deep copy.
    Source: `data/tau2/domains/airline/db.json`.
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

## Adding a new source

Add a new section above using the same fields. If the new source overlaps a
domain already present (e.g., a second airline-data library), namespace the
files with a distinct prefix (`tau_airline_*`, `cs_airline_*`, etc.) to avoid
collisions, and register a corresponding `Scenario` subclass that resolves
fixtures via `get_eval_data_root() / "<prefix>_scenarios"`.

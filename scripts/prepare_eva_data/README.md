# `prepare_eva_data/`

One-shot developer-only scripts for the eva_airline domain — companion to
`prepare_tau2_data/` (which handles the tau2 domains).

## Scripts

| Script | What it does |
|---|---|
| `generate_airline_scaffolds.py` | Reads `evaluation/data/eva_airline_dataset.jsonl` and emits `@register_eval_scenario class Eva...` definitions for the major group requested via `--major N`. **Streams to stdout** (unlike the tau2 scaffolders which fully overwrite group files) because eva scenarios carry curated prose that benefits from hand-review before committing. |

## When to re-run

- After updating `eva_airline_dataset.jsonl` (e.g. adding a new edge-case entry).
- After tweaking the scaffold template / prose conventions.
- Never in CI — output is a starting point, not final. The 5 hand-authored seeds (1.1.2, 2.1.1, 3.1.3, 5.1.1, 7.2.1) remain the canonical reference for prose style and are skipped by the generator.

## Usage

```bash
# Single major group → append to its group file. Hand-review prose +
# prune negotiation / edge-case bullets before committing.
python scripts/prepare_eva_data/generate_airline_scaffolds.py --major 1 \
    >> nemo_voice_agent/evaluation/scenarios/data/eva_airline/group_1x.py

# All major groups in one pass — useful for previewing total scaffold
# size or piping into a per-group split script. Stdout carries each
# group prefixed by ``# === eva_airline major group N.x (K scenarios) ===``
# section markers so the output is splittable mechanically. Stderr
# carries the per-group manifest.
python scripts/prepare_eva_data/generate_airline_scaffolds.py > /tmp/all_scaffolds.py

# Override dataset location (rare; useful for testing alternates)
python scripts/prepare_eva_data/generate_airline_scaffolds.py --major 1 \
    --dataset /path/to/alternate_dataset.jsonl > /tmp/preview.py
```

The generator applies the alphanumeric voice-readability rule (spelling out confirmation numbers, flight numbers, airport codes character-by-character) and reads `must_have_criteria` / `negotiation_behavior` / `edge_cases` from each dataset entry into the scenario's guidelines.

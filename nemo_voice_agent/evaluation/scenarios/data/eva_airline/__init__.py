# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""eva_airline scenario package.

Layout:
    base.py        — ``EvaAirlineBaseScenario`` + 5 hand-authored seed scenarios.
    group_Nx.py    — auto-scaffolded scenarios for eva sub-flow ``N.*``
                     (generated from ``eva_airline_dataset.jsonl`` via
                     ``nemo_experiments/generate_eva_airline_scaffolds.py``).

This ``__init__`` re-exports ``EvaAirlineBaseScenario`` so the group submodules
can keep their canonical import path (``from ...eva_airline import EvaAirlineBaseScenario``)
and triggers the group submodule imports so their ``@register_eval_scenario``
decorators fire.
"""

# Trigger @register_eval_scenario decorators in each group module. Order matches
# the eva taxonomy (1.x = voluntary, 2.x = IRROPS, etc.).
from nemo_voice_agent.evaluation.scenarios.data.eva_airline import (  # noqa: F401, E402
    group_1x,
    group_2x,
    group_3x,
    group_4x,
    group_5x,
    group_6x,
    group_7x,
)
from nemo_voice_agent.evaluation.scenarios.data.eva_airline.base import (
    EvaAirlineBaseScenario,
)


__all__ = ["EvaAirlineBaseScenario"]

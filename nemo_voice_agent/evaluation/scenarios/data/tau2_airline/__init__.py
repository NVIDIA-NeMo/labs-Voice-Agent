# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""tau2_airline scenarios package.

``base.py`` holds ``Tau2AirlineBaseScenario``. ``group_Nx.py`` modules hold
auto-scaffolded scenarios (see nemo_experiments/generate_tau2_airline_scaffolds.py).
Importing this package side-imports every group so registry decoration fires
for all 50 base-split tau2_airline scenarios.
"""

from nemo_voice_agent.evaluation.scenarios.data.tau2_airline.base import (  # noqa: F401
    Tau2AirlineBaseScenario,
)

# Side-import groups so @register_eval_scenario fires.
from nemo_voice_agent.evaluation.scenarios.data.tau2_airline import group_0x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_airline import group_1x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_airline import group_2x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_airline import group_3x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_airline import group_4x  # noqa: F401

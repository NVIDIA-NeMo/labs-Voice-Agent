# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""tau2_retail scenarios package.

``base.py`` holds ``Tau2RetailBaseScenario``. ``group_Nx.py`` modules hold
auto-scaffolded scenarios (see nemo_experiments/generate_tau2_retail_scaffolds.py).
Importing this package side-imports every group so registry decoration fires
for all 114 base-split tau2_retail scenarios.
"""

from nemo_voice_agent.evaluation.scenarios.data.tau2_retail.base import (  # noqa: F401
    Tau2RetailBaseScenario,
)

# Side-import groups so @register_eval_scenario fires.
from nemo_voice_agent.evaluation.scenarios.data.tau2_retail import group_0x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_retail import group_10x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_retail import group_11x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_retail import group_1x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_retail import group_2x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_retail import group_3x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_retail import group_4x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_retail import group_5x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_retail import group_6x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_retail import group_7x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_retail import group_8x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_retail import group_9x  # noqa: F401

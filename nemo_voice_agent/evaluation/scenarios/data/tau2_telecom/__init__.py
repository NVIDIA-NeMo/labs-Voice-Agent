# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""tau2_telecom scenarios package.

``base.py`` holds ``Tau2TelecomBaseScenario``. ``group_Nx.py`` modules hold
auto-scaffolded scenarios (see scripts/prepare_tau2_data/generate_telecom_scaffolds.py).
Importing this package side-imports every group so registry decoration fires
for all 114 base-split tau2_telecom scenarios.
"""

# Side-import groups so @register_eval_scenario fires.
from nemo_voice_agent.evaluation.scenarios.data.tau2_telecom import (
    group_0x,  # noqa: F401
    group_1x,  # noqa: F401
    group_2x,  # noqa: F401
    group_3x,  # noqa: F401
    group_4x,  # noqa: F401
    group_5x,  # noqa: F401
    group_6x,  # noqa: F401
    group_7x,  # noqa: F401
    group_8x,  # noqa: F401
    group_9x,  # noqa: F401
    group_10x,  # noqa: F401
    group_11x,  # noqa: F401
)
from nemo_voice_agent.evaluation.scenarios.data.tau2_telecom.base import (  # noqa: F401
    Tau2TelecomBaseScenario,
)

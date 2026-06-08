# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""tau2_telecom scenarios package.

``base.py`` holds ``Tau2TelecomBaseScenario``. ``group_Nx.py`` modules
hold the auto-scaffolded scenarios when bulk-shipped. Importing this
package side-imports every group so registry decoration fires for all
in-tree telecom scenarios.
"""

from nemo_voice_agent.evaluation.scenarios.data.tau2_telecom.base import (  # noqa: F401
    Tau2TelecomBaseScenario,
)

# Side-import groups so @register_eval_scenario fires.
from nemo_voice_agent.evaluation.scenarios.data.tau2_telecom import group_0x  # noqa: F401

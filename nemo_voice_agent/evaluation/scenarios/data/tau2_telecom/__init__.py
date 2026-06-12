# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""tau2_telecom scenarios package.

``base.py`` holds ``Tau2TelecomBaseScenario``. ``group_Nx.py`` modules hold
auto-scaffolded scenarios (see scripts/prepare_tau2_data/generate_telecom_scaffolds.py).
Importing this package side-imports every group so registry decoration fires
for all 114 base-split tau2_telecom scenarios.
"""

from nemo_voice_agent.evaluation.scenarios.data.tau2_telecom.base import (  # noqa: F401
    Tau2TelecomBaseScenario,
)

# Side-import groups so @register_eval_scenario fires.
from nemo_voice_agent.evaluation.scenarios.data.tau2_telecom import group_0x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_telecom import group_1x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_telecom import group_2x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_telecom import group_3x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_telecom import group_4x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_telecom import group_5x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_telecom import group_6x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_telecom import group_7x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_telecom import group_8x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_telecom import group_9x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_telecom import group_10x  # noqa: F401
from nemo_voice_agent.evaluation.scenarios.data.tau2_telecom import group_11x  # noqa: F401

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# Hand-authored seed scenario(s). Auto-scaffolded group_Nx.py modules
# (generated from tau2's tasks.json) will be added incrementally as
# additional scenarios are validated end-to-end.

"""Hand-authored tau2_telecom scenarios.

The first scenario is the simplest pure-user-side base-split candidate
from upstream — both reference actions are user-side toggles, both
``env_assertions`` are user-side, and ``PERSONA:None`` avoids
persona-styling complications. Use as the bring-up smoke target;
auto-scaffolded scenarios reuse the same ``Tau2TelecomBaseScenario``
machinery.
"""

# pylint: disable=line-too-long
# flake8: noqa: E501

from nemo_voice_agent.evaluation.scenarios import register_eval_scenario
from nemo_voice_agent.evaluation.scenarios.data.tau2_telecom.base import (
    Tau2TelecomBaseScenario,
)


@register_eval_scenario
class Tau2TelecomMobileDataIssueAirplaneOnDataOff(Tau2TelecomBaseScenario):
    """Mobile-data-not-working scenario, dual root cause: airplane mode is ON
    AND mobile data is OFF.

    The agent must diagnose both faults via ``check_status_bar`` /
    ``run_speed_test`` reads, then instruct the user to toggle airplane
    mode off and toggle mobile data on. Resolution gate is
    ``assert_mobile_data_status(True) && assert_internet_speed(200, "excellent")``.

    Init actions (all user-side): ``set_user_info`` (set name + phone),
    ``turn_airplane_mode_on`` (the bad state #1), ``turn_data_off`` (the
    bad state #2).
    """

    name = "tau2_telecom__mobile_data_issue__airplane_mode_on__data_mode_off"
    tau2_id = "[mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:None]"

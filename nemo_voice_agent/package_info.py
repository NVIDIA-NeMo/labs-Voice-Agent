# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import os as _os
import subprocess as _subprocess


MAJOR = 0
MINOR = 1
PATCH = 0
PRE_RELEASE = ""

VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = ".".join(map(str, VERSION[:3]))
__version__ = ".".join(map(str, VERSION[:3])) + "".join(VERSION[3:])

if not int(_os.getenv("NO_VCS_VERSION", "1")):
    try:
        _git = _subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            cwd=_os.path.dirname(_os.path.abspath(__file__)),
            check=True,
            text=True,
        )
    except (_subprocess.CalledProcessError, OSError):
        pass
    else:
        __version__ += f"+{_git.stdout.strip()}"

__package_name__ = "nemo_voice_agent"
__contact_names__ = "NVIDIA"
__contact_emails__ = "nemo-toolkit@nvidia.com"
__homepage__ = "https://github.com/NVIDIA-NeMo/Voice-Agent"
__repository_url__ = "https://github.com/NVIDIA-NeMo/Voice-Agent"
__download_url__ = "https://github.com/NVIDIA-NeMo/Voice-Agent/releases"
__description__ = "NeMo Voice Agent"
__license__ = "Apache-2.0"
__keywords__ = "voice agent, speech, LLM, NeMo, Pipecat, NVIDIA"

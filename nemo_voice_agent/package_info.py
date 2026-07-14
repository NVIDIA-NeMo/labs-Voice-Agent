# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

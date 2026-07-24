#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
# Launch bot_server_nemotron.py in USER role (simulated user bot for eval).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WEBSOCKET_PORT=8766
export FASTAPI_PORT=7861
export LOG_FILE="bot_user_nemotron.log"
export VAD_PROFILE="Silero"
export VAD_STOP_SECS=1.2
export TTS_VOICE_ID="Magpie-Multilingual.EN-US.Leo"
export TALK_FIRST="false"

export NVIDIA_LLM_MODEL="nvidia/nemotron-3-super-120b-a12b"
export ENABLE_TOOL_CALLING="true"
export ENABLE_THINKING="true"
export THINKING_BUDGET=1500
export TEMPERATURE=1.0
export TOP_P=0.95
export MAX_TOKENS=2048

python "$SCRIPT_DIR/bot_server_nemotron.py"


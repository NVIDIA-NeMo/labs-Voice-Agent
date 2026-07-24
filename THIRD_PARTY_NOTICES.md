<!--
Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Third-Party Notices

The NeMo Voice Agent project is licensed under the Apache License, Version 2.0
(see [`LICENSE`](./LICENSE)).

This file reproduces the copyright notices and license texts of third-party
open-source software whose source code and/or data fixtures are copied into
and redistributed as part of this repository, as required by their licenses.

For a per-file description of which upstream artifacts were imported, their
exact versions/commits, and which files in this repository they map to, see
[`nemo_voice_agent/evaluation/data/README.md`](./nemo_voice_agent/evaluation/data/README.md).
Adapted source files additionally carry an inline `# Adapted from <url>`
attribution at the top of the file.

---

## 1. eva (ServiceNow)

- **Project**: eva
- **Upstream**: https://github.com/ServiceNow/eva
- **Version**: `0.1.3`
- **License**: MIT
- **Used in this repository**:
  - `nemo_voice_agent/evaluation/db_hash.py` (adapted)
  - `nemo_voice_agent/evaluation/tools/eva_airline_tools.py`,
    `eva_airline_params.py` (adapted)
  - `nemo_voice_agent/evaluation/scenarios/data/eva_airline/` (adapted)
  - `nemo_voice_agent/evaluation/data/eva_airline/` (data fixtures, verbatim)

```
MIT License

Copyright (c) 2026 ServiceNow

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. tau2-bench (Sierra Research)

- **Project**: tau2-bench
- **Upstream**: https://github.com/sierra-research/tau2-bench
- **Version**: tag `voice-user-sim-v1.0` (commit `17e07b1`)
- **License**: MIT
- **Used in this repository**:
  - `nemo_voice_agent/evaluation/tools/tau2_airline_tools.py`,
    `tau2_airline_params.py`, `tau2_retail_tools.py`, `tau2_retail_params.py`,
    `tau2_telecom_tools.py`, `tau2_telecom_params.py`,
    `tau2_telecom_init_functions.py`, `tau2_telecom_user_tools.py` (adapted)
  - `nemo_voice_agent/evaluation/tools/_write_tool_base.py` and other
    evaluation modules carrying an inline `# Adapted from` attribution
  - `nemo_voice_agent/evaluation/scenarios/data/tau2_airline/`,
    `tau2_retail/`, `tau2_telecom/` (adapted)
  - `nemo_voice_agent/evaluation/data/tau2_airline/`, `tau2_retail/`,
    `tau2_telecom/` (data fixtures, verbatim or format-converted)

```
MIT License

Copyright (c) 2025 Sierra Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 3. Pipecat (Daily)

- **Project**: Pipecat
- **Upstream**: https://github.com/pipecat-ai/pipecat
- **License**: BSD 2-Clause
- **Used in this repository**:
  - `examples/generic_voice_agent/client/src/app.ts` (adapted RTVI client;
    retains its original Daily copyright and BSD 2-Clause header)

```
BSD 2-Clause License

Copyright (c) 2024–2025, Daily

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

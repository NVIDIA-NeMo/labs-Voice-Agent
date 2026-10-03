#!/usr/bin/env bash
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
#
# Prepare a pip-installed NVIDIA CUDA toolkit for native/JIT compilation, for the case where
# vLLM builds a kernel at runtime and cannot find nvcc or the CUDA runtime. The prebuilt vLLM
# wheels need none of this; run it only when a build actually fails.
#
# Source this file so the exported paths stay active in the shell that launches vLLM:
#   source scripts/setup_cuda_toolchain.sh
#
# Deliberately no `set -euo pipefail`: sourcing applies those options to the calling shell and
# they persist afterwards, so a failure here would kill an interactive shell and a later unset
# variable would kill it too. Strictness lives inside the function instead.

setup_cuda_toolchain() {
  local setup_script_dir project_root venv_dir python uv_bin
  local runtime_version runtime_series runtime_major runtime_minor runtime_rest
  local nvcc_version nvcc_major nvcc_minor nvcc_rest nvcc_series cuda_upper_bound requirement nvcc_path
  local cuda_lib_dir cuda_link_dir library filename linker_name
  local -a cuda_versioned_libraries

  setup_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  project_root="${PROJECT_ROOT:-$(cd -- "${setup_script_dir}/.." && pwd)}"
  venv_dir="${VENV_DIR:-${VIRTUAL_ENV:-${project_root}/.venv}}"
  python="${PYTHON:-${venv_dir}/bin/python}"

  if [[ ! -x "${python}" ]]; then
    echo "Python virtual environment not found at ${venv_dir}." >&2
    echo "Run 'bash install.sh' from ${project_root} first." >&2
    return 1
  fi

  runtime_version="$("${python}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    print(version("nvidia-cuda-runtime"))
except PackageNotFoundError:
    raise SystemExit(1)
PY
  )" || {
    echo "The NVIDIA CUDA runtime wheel is not installed in ${venv_dir}." >&2
    return 1
  }

  IFS="." read -r runtime_major runtime_minor runtime_rest <<<"${runtime_version}"
  if [[ ! "${runtime_major}" =~ ^[0-9]+$ || ! "${runtime_minor}" =~ ^[0-9]+$ ]]; then
    echo "Cannot determine the CUDA series from runtime version '${runtime_version}'." >&2
    return 1
  fi
  runtime_series="${runtime_major}.${runtime_minor}"
  cuda_upper_bound="${runtime_major}.$((10#${runtime_minor} + 1))"

  nvcc_version="$("${python}" - <<'PY' 2>/dev/null || true
from importlib.metadata import version

print(version("nvidia-cuda-nvcc"))
PY
  )"
  nvcc_series=""
  if [[ -n "${nvcc_version}" ]]; then
    IFS="." read -r nvcc_major nvcc_minor nvcc_rest <<<"${nvcc_version}"
    nvcc_series="${nvcc_major}.${nvcc_minor}"
  fi

  uv_bin="$(command -v uv || true)"
  # nvcc must match the runtime series, so an existing mismatched nvcc is replaced rather than kept.
  if [[ "${nvcc_series}" != "${runtime_series}" ]]; then
    if [[ -z "${uv_bin}" ]]; then
      echo "uv is required to install the matching CUDA compiler components." >&2
      return 1
    fi
    requirement="cuda-toolkit[nvcc,crt,cccl,nvvm]>=${runtime_series},<${cuda_upper_bound}"
    echo "Installing compiler components matching CUDA runtime ${runtime_series}..."
    "${uv_bin}" pip install --python "${python}" "${requirement}" || return 1
  fi

  nvcc_path="$("${python}" - <<'PY'
from importlib.metadata import PackageNotFoundError, distribution

try:
    package = distribution("nvidia-cuda-nvcc")
    nvcc = next(package.locate_file(path) for path in package.files or [] if str(path).endswith("/nvcc"))
except (PackageNotFoundError, StopIteration):
    raise SystemExit(1)
print(nvcc.resolve())
PY
  )" || {
    echo "nvcc is still unavailable after CUDA compiler setup." >&2
    return 1
  }

  CUDA_HOME="$(dirname -- "$(dirname -- "${nvcc_path}")")"
  cuda_lib_dir="${CUDA_HOME}/lib"
  cuda_link_dir="${XDG_CACHE_HOME:-${HOME}/.cache}/nemo-voice-agent/cuda-link-${runtime_series}"

  shopt -s nullglob
  cuda_versioned_libraries=("${cuda_lib_dir}"/lib*.so.*)
  shopt -u nullglob
  if (( ${#cuda_versioned_libraries[@]} == 0 )); then
    echo "CUDA shared libraries were not found under ${cuda_lib_dir}." >&2
    return 1
  fi

  # The wheels ship only versioned SONAMEs (libcudart.so.13), but `-lcudart` resolves the
  # unversioned linker name at link time. Build the missing symlinks in a cache directory.
  mkdir -p "${cuda_link_dir}" || return 1
  for library in "${cuda_versioned_libraries[@]}"; do
    filename="${library##*/}"
    linker_name="${filename%%.so.*}.so"
    ln -sfn "${library}" "${cuda_link_dir}/${linker_name}" || return 1
  done

  if [[ ! -x "${venv_dir}/bin/ninja" ]]; then
    if [[ -z "${uv_bin}" ]]; then
      echo "ninja is missing and uv is unavailable to install it." >&2
      return 1
    fi
    echo "Installing the Ninja build tool..."
    "${uv_bin}" pip install --python "${python}" ninja || return 1
  fi

  export CUDA_HOME
  export PATH="${venv_dir}/bin:${CUDA_HOME}/bin:${PATH}"
  # These precede any system CUDA so the toolchain matches the wheels torch was built against.
  export LIBRARY_PATH="${cuda_link_dir}:${cuda_lib_dir}:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="${cuda_link_dir}:${cuda_lib_dir}:${LD_LIBRARY_PATH:-}"

  # Catch compiler/header and missing-linker-name problems before loading a model.
  "${CUDA_HOME}/bin/nvcc" -x cu - -o /dev/null -lcudart -lcublas -lcublasLt <<'CUDA' || return 1
#include <cuda_runtime.h>

int main() {
  return 0;
}
CUDA

  nvcc_version="$("${python}" -c 'from importlib.metadata import version; print(version("nvidia-cuda-nvcc"))')"
  echo "CUDA toolchain ready: runtime ${runtime_version}, nvcc ${nvcc_version}, home ${CUDA_HOME}"
}

if setup_cuda_toolchain; then
  if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Setup is complete. Source this script from launchers to inherit its environment."
  fi
else
  echo "CUDA toolchain setup failed." >&2
  # Return when sourced, exit when executed: never kill the caller's shell.
  return 1 2>/dev/null || exit 1
fi

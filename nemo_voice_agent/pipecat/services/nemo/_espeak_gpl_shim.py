# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stand-ins for ``phonemizer``/``espeakng_loader`` so importing ``kokoro``/``misaki``
does not require installing GPL-3.0-licensed packages.

``misaki.espeak`` unconditionally imports ``phonemizer`` and ``espeakng_loader`` at
module load time to provide an out-of-vocabulary word fallback backed by espeak-ng.
Both packages are GPL-3.0 (``phonemizer``/``phonemizer-fork``) or bundle a compiled
GPL-3.0 binary (``espeakng_loader`` ships ``libespeak-ng.so``), which is incompatible
with this project's Apache-2.0 license. This project does not depend on them.

These stubs satisfy misaki's unconditional imports; constructing the real espeak
backend then raises, which ``kokoro.KPipeline`` already catches internally and falls
back to ``fallback=None`` (misaki's dictionary-only G2P, with no espeak-ng-backed
fallback for out-of-vocabulary words).
"""

import sys
import types


def install() -> None:
    """Register no-op stand-ins for ``phonemizer``/``espeakng_loader`` in ``sys.modules``."""
    if "phonemizer" in sys.modules or "espeakng_loader" in sys.modules:
        return
    try:
        import phonemizer  # noqa: F401

        return  # Real package is present (e.g. installed manually); don't shadow it.
    except ImportError:
        pass

    espeakng_loader_stub = types.ModuleType("espeakng_loader")
    espeakng_loader_stub.get_library_path = lambda: None
    espeakng_loader_stub.get_data_path = lambda: None
    sys.modules["espeakng_loader"] = espeakng_loader_stub

    class _EspeakWrapper:
        @staticmethod
        def set_library(path):
            pass

        @staticmethod
        def set_data_path(path):
            pass

    class _EspeakBackend:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("espeak backend unavailable: phonemizer/espeak-ng (GPL-3.0) are intentionally excluded")

    phonemizer_stub = types.ModuleType("phonemizer")
    backend_stub = types.ModuleType("phonemizer.backend")
    espeak_stub = types.ModuleType("phonemizer.backend.espeak")
    wrapper_stub = types.ModuleType("phonemizer.backend.espeak.wrapper")
    wrapper_stub.EspeakWrapper = _EspeakWrapper
    espeak_stub.wrapper = wrapper_stub
    backend_stub.espeak = espeak_stub
    backend_stub.EspeakBackend = _EspeakBackend
    phonemizer_stub.backend = backend_stub

    sys.modules["phonemizer"] = phonemizer_stub
    sys.modules["phonemizer.backend"] = backend_stub
    sys.modules["phonemizer.backend.espeak"] = espeak_stub
    sys.modules["phonemizer.backend.espeak.wrapper"] = wrapper_stub

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline unit coverage for the Sortformer streaming overrides.

These run on CPU against a bare ``SortformerModules`` -- no checkpoint, no GPU. They exist because
``nn.Module.__setattr__`` accepts any attribute name, so an override naming a parameter that NeMo has
renamed is a silent no-op that leaves the checkpoint value in force.
"""

from types import SimpleNamespace

import pytest
from nemo.collections.asr.modules.sortformer_modules import SortformerModules

from nemo_voice_agent.pipecat.services.nemo.streaming_diar import DiarizationConfig, NeMoStreamingDiarService


def _service(**overrides):
    service = NeMoStreamingDiarService.__new__(NeMoStreamingDiarService)
    service.cfg = DiarizationConfig(**overrides)
    return service


def _model():
    """A stand-in for a loaded checkpoint, carrying a real (default-valued) sortformer module."""
    return SimpleNamespace(sortformer_modules=SortformerModules())


def test_every_configured_override_reaches_the_sortformer_module():
    model = _model()
    service = _service(spkcache_update_period=144, fifo_len=188, chunk_len=6, chunk_right_context=7)

    service.apply_streaming_config(model)

    modules = model.sortformer_modules
    assert modules.chunk_len == 6
    assert modules.fifo_len == 188
    assert modules.chunk_right_context == 7
    assert modules.log is False
    # The knob NeMo renamed: writing the stale name left this at the checkpoint's value.
    assert modules.spkcache_update_period == 144
    assert not hasattr(modules, "spkcache_refresh_rate")


def test_none_valued_overrides_leave_the_checkpoint_values_in_place():
    model = _model()
    baseline = SortformerModules()
    service = _service(spkcache_len=None, chunk_left_context=None)

    service.apply_streaming_config(model)

    assert model.sortformer_modules.spkcache_len == baseline.spkcache_len
    assert model.sortformer_modules.chunk_left_context == baseline.chunk_left_context


def test_explicit_overrides_win_over_the_checkpoint_values():
    model = _model()
    service = _service(spkcache_len=200, chunk_left_context=2)

    service.apply_streaming_config(model)

    assert model.sortformer_modules.spkcache_len == 200
    assert model.sortformer_modules.chunk_left_context == 2


def test_legacy_spkcache_attribute_name_is_still_honoured():
    """NeMo releases before the ``spkcache_update_period`` rename expose ``spkcache_refresh_rate``."""
    legacy = SimpleNamespace(chunk_len=0, chunk_right_context=0, fifo_len=0, log=True, spkcache_refresh_rate=0)
    service = _service(spkcache_update_period=144)

    service.apply_streaming_config(SimpleNamespace(sortformer_modules=legacy))

    assert legacy.spkcache_refresh_rate == 144
    assert not hasattr(legacy, "spkcache_update_period")


def test_unknown_streaming_parameter_name_raises_instead_of_writing_silently():
    unknown = SimpleNamespace(chunk_len=0, chunk_right_context=0, fifo_len=0, log=True)
    service = _service()

    with pytest.raises(AttributeError, match="spkcache_update_period"):
        service.apply_streaming_config(SimpleNamespace(sortformer_modules=unknown))

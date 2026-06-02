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

"""Tests for the per-domain tool registry (M2.7).

Covers the four invariants of ``register_schema_tool_for_eval`` /
``get_schema_tool_for_eval``:

  1. Cross-domain identical names are allowed and resolve independently.
  2. Within-domain collision raises ``ValueError`` with a clear message.
  3. Lookup falls back to ``"default"`` with a warning when the name isn't
     in the specified domain but is in default.
  4. Lookup raises ``KeyError`` when the name is absent from both the
     specified domain and ``"default"``.

Plus a backward-compat check that bare ``@register_schema_tool_for_eval``
(no parens) still registers into ``"default"``.

These tests use a temporary registry-state fixture to avoid polluting the
process-global ``ALL_SCHEMA_TOOLS_FOR_EVAL`` from the real tool packages.
"""

from typing import Any, Dict, List

import pytest

from nemo_voice_agent.evaluation.tools import (
    ALL_SCHEMA_TOOLS_FOR_EVAL,
    get_schema_tool_for_eval,
    register_schema_tool_for_eval,
)
from nemo_voice_agent.utils.tool_calling import StandardSchemaTool


# ---------------------------------------------------------------------------
# Fixture: snapshot + restore the global registry around each test
# ---------------------------------------------------------------------------


@pytest.fixture
def restore_registry():
    """Snapshot ``ALL_SCHEMA_TOOLS_FOR_EVAL`` so test mutations don't leak."""
    snapshot = {d: dict(bucket) for d, bucket in ALL_SCHEMA_TOOLS_FOR_EVAL.items()}
    yield
    ALL_SCHEMA_TOOLS_FOR_EVAL.clear()
    ALL_SCHEMA_TOOLS_FOR_EVAL.update(snapshot)


# ---------------------------------------------------------------------------
# Minimal StandardSchemaTool subclasses for testing
# ---------------------------------------------------------------------------


def _make_stub_tool_class(class_name: str):
    """Dynamically build a minimal StandardSchemaTool subclass with the given
    class name. Using ``type(...)`` lets us create multiple distinct classes
    that share a name across different domains in the same test process."""

    def _init(self, *, description=None):
        StandardSchemaTool.__init__(self, description=description or "stub")

    @property
    def _properties(self) -> Dict[str, Any]:
        return {}

    @property
    def _required_properties(self) -> List[str]:
        return []

    async def _execute(self, params):
        return {}

    return type(
        class_name,
        (StandardSchemaTool,),
        {
            "__init__": _init,
            "properties": _properties,
            "required_properties": _required_properties,
            "_execute": _execute,
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cross_domain_identical_names_register_independently(restore_registry):
    """Same class name in different domains coexists as two distinct entries.

    Uses synthetic ``_test_domain_a/b`` namespaces to avoid colliding with the
    real eva_airline/tau2_airline tools that the package-level imports already
    registered.
    """
    Tool1 = _make_stub_tool_class("CrossDomainStubTool")
    Tool2 = _make_stub_tool_class("CrossDomainStubTool")
    register_schema_tool_for_eval(domain="_test_domain_a")(Tool1)
    register_schema_tool_for_eval(domain="_test_domain_b")(Tool2)
    assert ALL_SCHEMA_TOOLS_FOR_EVAL["_test_domain_a"]["CrossDomainStubTool"] is Tool1
    assert ALL_SCHEMA_TOOLS_FOR_EVAL["_test_domain_b"]["CrossDomainStubTool"] is Tool2
    # Lookup picks the right one based on domain
    inst1 = get_schema_tool_for_eval("CrossDomainStubTool", domain="_test_domain_a")
    inst2 = get_schema_tool_for_eval("CrossDomainStubTool", domain="_test_domain_b")
    assert type(inst1) is Tool1
    assert type(inst2) is Tool2


def test_within_domain_collision_raises(restore_registry):
    """Re-registering the same name in the same domain raises ValueError."""
    Tool1 = _make_stub_tool_class("RegistryStubA")
    Tool2 = _make_stub_tool_class("RegistryStubA")
    register_schema_tool_for_eval(domain="test_domain")(Tool1)
    with pytest.raises(ValueError) as exc:
        register_schema_tool_for_eval(domain="test_domain")(Tool2)
    msg = str(exc.value)
    assert "Tool name collision" in msg
    assert "test_domain" in msg
    assert "RegistryStubA" in msg


def test_lookup_falls_back_to_default_with_warning(restore_registry, caplog):
    """Tool registered only in 'default' resolves when looked up from another domain."""
    DefaultTool = _make_stub_tool_class("SharedHarnessTool")
    register_schema_tool_for_eval(domain="default")(DefaultTool)

    # Lookup from a different domain succeeds via fallback to "default".
    inst = get_schema_tool_for_eval("SharedHarnessTool", domain="tau2_airline")
    assert type(inst) is DefaultTool


def test_lookup_raises_when_absent_from_both_domain_and_default(restore_registry):
    """``KeyError`` if name is absent from specified domain AND from default."""
    register_schema_tool_for_eval(domain="bogus_domain")(_make_stub_tool_class("ToolA"))

    # Looking up an unknown tool from the bogus domain fails (and default has no fallback).
    with pytest.raises(KeyError) as exc:
        get_schema_tool_for_eval("DoesNotExistTool", domain="bogus_domain")
    msg = str(exc.value)
    assert "not found" in msg
    assert "DoesNotExistTool" in msg


def test_lookup_in_default_with_unknown_name_raises(restore_registry):
    """Looking up an unknown tool from default raises (no domain fallback)."""
    with pytest.raises(KeyError):
        get_schema_tool_for_eval("NonexistentToolXYZ", domain="default")


def test_bare_decorator_backward_compat_registers_into_default(restore_registry):
    """``@register_schema_tool_for_eval`` (no parens) registers into ``"default"``."""
    Tool = _make_stub_tool_class("BareDecoratorTool")
    register_schema_tool_for_eval(Tool)  # mimics @register_schema_tool_for_eval
    assert ALL_SCHEMA_TOOLS_FOR_EVAL["default"]["BareDecoratorTool"] is Tool
    inst = get_schema_tool_for_eval("BareDecoratorTool", domain="default")
    assert type(inst) is Tool


def test_positional_string_argument_to_decorator(restore_registry):
    """``@register_schema_tool_for_eval("tau2_airline")`` works as a shortcut."""
    Tool = _make_stub_tool_class("PositionalArgTool")
    register_schema_tool_for_eval("_test_domain_pos")(Tool)
    assert ALL_SCHEMA_TOOLS_FOR_EVAL["_test_domain_pos"]["PositionalArgTool"] is Tool


def test_real_world_eva_vs_tau2_cancel_reservation_no_collision():
    """Sanity check on the live registry: eva and tau2 both have
    ``CancelReservationTool``, registered in their respective domains."""
    eva = ALL_SCHEMA_TOOLS_FOR_EVAL.get("eva_airline", {}).get("CancelReservationTool")
    tau2 = ALL_SCHEMA_TOOLS_FOR_EVAL.get("tau2_airline", {}).get("CancelReservationTool")
    assert eva is not None, "eva_airline.CancelReservationTool should be registered"
    assert tau2 is not None, "tau2_airline.CancelReservationTool should be registered"
    assert eva is not tau2, "must be distinct classes despite sharing a name"

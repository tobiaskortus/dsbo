import pytest
import torch
from unittest.mock import MagicMock
from dsbo.workflow import FusedModule
from tests.workflow.conftest import SimpleModule


class TestFusedModule:
    def test_fuses_multiple_modules(self):
        m1 = SimpleModule()
        m2 = SimpleModule()
        fused = FusedModule(m1, m2)
        result = fused(design={"x": torch.tensor(3.0)})
        assert torch.allclose(result["out"], torch.tensor(6.0))

    def test_run_calls_all_modules(self):
        m1 = SimpleModule()
        m2 = SimpleModule()
        fused = FusedModule(m1, m2)
        result = fused.run(design={"x": torch.tensor(3.0)})
        assert torch.allclose(result["out"], torch.tensor(6.0))

    def test_asserts_matching_resources(self):
        m1 = SimpleModule()
        m2 = SimpleModule()
        m1.resources = MagicMock()
        m2.resources = MagicMock()
        m1.resources.__eq__ = lambda self, o: False
        m2.resources.__eq__ = lambda self, o: False
        with pytest.raises(AssertionError, match="same resources"):
            FusedModule(m1, m2)

    def test_resources_from_first_module(self):
        m1 = SimpleModule()
        m2 = SimpleModule()
        fused = FusedModule(m1, m2)
        assert fused.resources is m1.resources

    def test_name_joins_module_names(self):
        m1 = SimpleModule()
        m2 = SimpleModule()
        fused = FusedModule(m1, m2)
        assert fused.name == "[SimpleModule,SimpleModule]"

    def test_resources_setter_is_noop(self):
        m1 = SimpleModule()
        m2 = SimpleModule()
        fused = FusedModule(m1, m2)
        fused.resources = "ignored"
        assert fused.resources is m1.resources
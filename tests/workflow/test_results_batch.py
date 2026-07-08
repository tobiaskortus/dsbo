import pytest
import torch
from dsbo.workflow import ResultsBatch


class TestResultsBatch:
    def test_init_from_list(self):
        rb = ResultsBatch([{"a": 1}])
        assert len(rb) == 1

    def test_init_from_args(self):
        rb = ResultsBatch({"a": 1}, {"a": 2})
        assert len(rb) == 2

    def test_getitem(self):
        rb = ResultsBatch({"a": 1}, {"a": 2})
        assert rb[0] == {"a": 1}

    def test_keys_property(self):
        rb = ResultsBatch({"a": 1, "b": 2}, {"a": 3, "b": 4})
        assert list(rb.keys) == ["a", "b"]

    def test_keys_empty_batch(self):
        rb = ResultsBatch()
        assert rb.keys == []

    def test_tensor_property(self):
        rb = ResultsBatch(
            {"a": torch.tensor(1.0), "b": torch.tensor(2.0)},
            {"a": torch.tensor(3.0), "b": torch.tensor(4.0)},
        )
        t = rb.tensor
        assert t.shape == (2, 2)

    def test_tensor_empty_batch_raises(self):
        rb = ResultsBatch()
        with pytest.raises(RuntimeError, match="stack"):
            _ = rb.tensor

    def test_getitem_key_concat_tensors(self):
        rb = ResultsBatch(
            {"out": torch.tensor([1.0, 2.0])},
            {"out": torch.tensor([3.0, 4.0])},
        )
        values = rb["out"]
        assert isinstance(values, torch.Tensor)
        assert values.shape == (4,)
        torch.testing.assert_close(values, torch.tensor([1.0, 2.0, 3.0, 4.0]))
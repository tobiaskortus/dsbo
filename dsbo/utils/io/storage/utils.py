import numpy as np
from typing import Literal, overload

import torch
from tensordict import TensorDict


def assert_valid_keys(x: TensorDict | dict[str, torch.Tensor | np.ndarray]):
    if len(x.keys()) == 0:
        raise ValueError("TensorDict must contain at least one key.")
    for key in x.keys():
        if not isinstance(key, str) or not key or key == "." or "/" in key or "\x00" in key:
            raise ValueError(f"Invalid TensorDict key: {key!r}")


def assert_same_batch_sizes(x: TensorDict | dict[str, torch.Tensor | np.ndarray]):
    batch_size = tuple(x.batch_size) if isinstance(x, TensorDict) else None
    if batch_size is None:
        first = next((tuple(t.shape) for t in x.values() if t.ndim), ())
        batch_size = first[:1]

    for key, value in x.items():
        if tuple(value.shape[:len(batch_size)]) != batch_size:
            raise AssertionError(
                "All tensors must share the batch size, got: "
                + ", ".join(f"{key!r}={tuple(value.shape)}" for key, value in x.items())
            )

    return batch_size


def assert_flat_dict_of_type_tensor(x: TensorDict | dict[str, torch.Tensor]):
    for t in x.values():
        assert isinstance(t, torch.Tensor)


def to_tensordict(x: TensorDict | dict[str, torch.Tensor | np.ndarray]) -> TensorDict:
    assert_valid_keys(x)
    batch_size = assert_same_batch_sizes(x)
    values = {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value for key, value in x.items()}
    assert_flat_dict_of_type_tensor(values)
    return TensorDict(values, batch_size=batch_size)


@overload
def to_dict(x: TensorDict, *, numpy: Literal[False] = False) -> dict[str, torch.Tensor]: 
    ...


@overload
def to_dict(x: TensorDict, *, numpy: Literal[True]) -> dict[str, np.ndarray]: 
    ...


def to_dict(x: TensorDict, *, numpy: bool = False) -> dict[str, torch.Tensor] | dict[str, np.ndarray]:
    assert_valid_keys(x)
    assert_same_batch_sizes(x)
    assert_flat_dict_of_type_tensor(x)
    if numpy:
        return {key: value.detach().cpu().numpy() for key, value in x.items()}
    return x.to_dict()

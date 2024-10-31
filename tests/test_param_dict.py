import numpy as np
import pytest
from snippets import param_dict
import torch as th
from typing import Tuple


@pytest.fixture
def shapes() -> param_dict.ShapeDict:
    return {
        "a": (),
        "b": (4,),
        "c": (7, 8),
    }


@pytest.fixture(params=[(), (3,), (9, 5)])
def batch_shape(request: pytest.FixtureRequest) -> Tuple[int]:
    return request.param


@pytest.fixture(params=[False, True], ids=["numpy", "torch"])
def use_torch(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture
def params(
    batch_shape: Tuple[int], shapes: param_dict.ShapeDict, use_torch: bool
) -> "param_dict.ParamDict":
    return {
        param: (
            th.randn(batch_shape + shape)
            if use_torch
            else np.random.normal(0, 1, batch_shape + shape)
        )
        for param, shape in shapes.items()
    }


def test_param_dict_roundtrip(
    params: "param_dict.ParamDict",
    shapes: param_dict.ShapeDict,
    batch_shape: param_dict.ShapeDict,
) -> None:
    collapsed = param_dict.from_param_dict(params, shapes)
    assert collapsed.shape[:-1] == batch_shape
    reconstructed = param_dict.to_param_dict(collapsed, shapes)

    assert set(params) == set(reconstructed)
    for param, value in params.items():
        np.testing.assert_array_equal(value, reconstructed[param])


def test_from_param_dict_mismatched_batch_shape(
    params: "param_dict.ParamDict", shapes: param_dict.ShapeDict
) -> None:
    params["b"] = params["b"].reshape((1, *params["b"].shape))
    with pytest.raises(ValueError, match="Expected batch shape"):
        param_dict.from_param_dict(params, shapes)


def test_from_param_dict_empty() -> None:
    with pytest.raises(ValueError, match="The parameter dictionary is empty."):
        param_dict.from_param_dict(None, None)


def test_from_param_dict_mismatch() -> None:
    with pytest.raises(ValueError, match="do not match."):
        param_dict.from_param_dict({"a": 1}, {"b": 2})


def test_to_param_dict_empty() -> None:
    with pytest.raises(ValueError, match="The shape dictionary is empty."):
        param_dict.to_param_dict(None, None)


def test_to_param_dict_size_mismatch(
    params: "param_dict.ParamDict", shapes: param_dict.ShapeDict
) -> None:
    collapsed = param_dict.from_param_dict(params, shapes)
    with pytest.raises(ValueError, match=r"Expected \d+ elements"):
        param_dict.to_param_dict(collapsed[..., :3], shapes)

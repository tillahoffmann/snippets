import numpy as np
import pytest
from snippets.nn import Affine, StopOnPlateau
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from unittest import mock


@pytest.mark.parametrize("mode", ["min", "max"])
@pytest.mark.parametrize("patience", [3, 7])
@pytest.mark.parametrize("threshold", [0.1, 0.2])
@pytest.mark.parametrize("threshold_mode", ["abs", "rel"])
def test_stop_on_plateau(mode: str, patience: int, threshold: float, threshold_mode: str) -> None:
    # Construct a dummy learning rate scheduler which should behave the same as our `StopOnPlateau`.
    module = torch.nn.Linear(2, 3)
    optim = torch.optim.Adam(module.parameters())
    scheduler = ReduceLROnPlateau(optim, mode=mode, patience=patience, threshold=threshold,
                                  threshold_mode=threshold_mode)
    stop = StopOnPlateau(mode=mode, patience=patience, threshold=threshold,
                         threshold_mode=threshold_mode)

    with mock.patch("torch.optim.lr_scheduler.ReduceLROnPlateau._reduce_lr") as _reduce_lr:
        while True:
            value = np.exp(np.random.normal(0, 1))
            scheduler.step(value)
            if stop.step(value):
                break
            _reduce_lr.assert_not_called()
        _reduce_lr.assert_called_once()


def test_stop_on_plateau_invalid() -> None:
    with pytest.raises(ValueError, match="foobar"):
        StopOnPlateau(mode="foobar")

    stop = StopOnPlateau()
    stop.mode = "foobar"
    with pytest.raises(ValueError, match="foobar"):
        stop.is_better(3, 4)


@pytest.mark.parametrize("loc_shape, scale_shape, value_shape, expected_shape", [
    ((3,), (3, 4), (17, 4), (17, 3)),
    ((), (1, 2,), (7, 2), (7, 1)),
    ((3,), (), (13, 3), (13, 3)),
    ((), (), (15, 5), (15, 5)),
])
def test_affine(loc_shape: torch.Size, scale_shape: torch.Size, value_shape: torch.Size,
                expected_shape: torch.Size) -> None:
    # Create module.
    loc = torch.randn(loc_shape)
    scale = torch.randn(scale_shape)
    affine = Affine(loc, scale)

    # Transform and compare with expected shape.
    value = torch.randn(value_shape)
    result = affine(value)
    assert result.shape == expected_shape

    # Verify the shape is the same as if we'd used a linear module.
    if scale.ndim == 0:
        largest = max(value.shape[-1], 0 if loc.ndim == 0 else loc.shape[0])
        in_features = out_features = largest
    else:
        out_features, in_features = scale.shape
    linear = torch.nn.Linear(in_features, out_features)
    assert linear(value).shape == result.shape

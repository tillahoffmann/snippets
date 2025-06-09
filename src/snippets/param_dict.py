from functools import reduce
from operator import mul
from typing import Dict, Tuple, TYPE_CHECKING


if TYPE_CHECKING:
    from .util import TensorLike


ParamDict = Dict[str, "TensorLike"]
ShapeDict = Dict[str, Tuple[int]]


def from_param_dict(params: "ParamDict", shapes: ShapeDict) -> "TensorLike":
    """
    Convert a dictionary of parameters to a tensor of parameters for batch processing.

    Args:
        params: Dictionary mapping parameter names to tensors.
        shapes: Dictionary mapping parameter names to shapes.

    Returns:
        Tensor of concatenated, raveled parameters ordered by parameter name.

    Example:

        .. doctest::

            >>> import torch
            >>> from snippets.param_dict import from_param_dict

            >>> params = {"b": torch.arange(12).reshape((3, 4)), "a": torch.arange(3)}
            >>> shapes = {"b": (3, 4), "a": (3,)}
            >>> from_param_dict(params, shapes)
            tensor([ 0,  1,  2,  0, ..., 11])
    """
    if not params:
        raise ValueError("The parameter dictionary is empty.")
    if set(params) != set(shapes):
        raise ValueError(
            "Parameter dictionary keys and shape dictionary keys do not match."
        )

    batch_shape = None
    batch_shape_param_name = None
    parts = []
    for param, value in sorted(params.items()):
        # Get the shapes and verify the batch shapes match up.
        param_shape = shapes[param]
        param_ndim = len(param_shape)
        param_batch_shape = value.shape[: len(value.shape) - param_ndim]

        if batch_shape is None:
            batch_shape = param_batch_shape
            batch_shape_param_name = param
        elif batch_shape != param_batch_shape:
            raise ValueError(
                f"Expected batch shape {batch_shape} based on "
                f"{batch_shape_param_name}; got {param_batch_shape} for {param}."
            )

        # Collapse the parameter shape.
        value = value.reshape((*batch_shape, -1))
        parts.append(value)

    if value.__class__.__name__ == "ndarray":
        import numpy as np

        return np.concatenate(parts, axis=-1)
    else:
        import torch as th

        return th.concatenate(parts, axis=-1)


def to_param_dict(params: "TensorLike", shapes: ShapeDict) -> "ParamDict":
    """
    Convert a tensor of parameters to a dictionary of parameters.

    Args:
        params: Tensor of concatenated, raveled parameters ordered by parameter name.
        shapes: Dictionary mapping parameter names to shapes.

    Returns:
        Dictionary mapping parameter names to tensors with the specified shape.

    Example:

        .. doctest::

            >>> import torch
            >>> from snippets.param_dict import to_param_dict

            >>> params = torch.arange(15)
            >>> shapes = {"b": (3, 4), "a": (3,)}
            >>> to_param_dict(params, shapes)  # doctest: +NORMALIZE_WHITESPACE
            {'a': tensor([0, 1, 2]), 'b': tensor([[ 3,  4,  5,  6], ...])}
    """
    if not shapes:
        raise ValueError("The shape dictionary is empty.")

    size = sum(reduce(mul, shape, 1) for shape in shapes.values())
    if params.shape[-1] != size:
        raise ValueError(
            f"Expected {size} elements based on shape dictionary; got "
            f"{params.shape[-1]}"
        )

    parts = {}
    offset = 0
    batch_shape = params.shape[:-1]
    for param, shape in sorted(shapes.items()):
        size = reduce(mul, shape, 1)
        parts[param] = params[..., offset : offset + size].reshape(
            (*batch_shape, *shape)
        )
        offset += size

    return parts

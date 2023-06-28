from typing import TYPE_CHECKING, Union


if TYPE_CHECKING:
    import numpy as np
    import torch as th

    TensorLike = Union[np.ndarray, th.Tensor]


class raise_for_missing_modules():
    """
    Helper class providing install instructions if a required package is missing.

    Example:

        .. doctest::

            >>> from snippets.util import raise_for_missing_modules

            >>> with raise_for_missing_modules():
            ...     import xxx
            Traceback (most recent call last):
                ...
            RuntimeError: install module `xxx` to use the snippet at `...`
    """
    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type, exc_val, traceback) -> None:
        return
        # Nothing to be done if there is no exception.
        if not exc_val:
            return

        if isinstance(exc_val, ModuleNotFoundError):

            message = f"install module `{exc_val.name}` to use the snippet at " \
                f"`{traceback.tb_frame.f_code.co_filename}`"
            raise RuntimeError(message) from exc_val
        # Reraise the original exception if not due to a missing module.
        # raise

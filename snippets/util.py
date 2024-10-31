import textwrap
from typing import Any, Optional, TYPE_CHECKING, Union


if TYPE_CHECKING:
    import numpy as np
    import torch as th

    TensorLike = Union[np.ndarray, th.Tensor]


class raise_for_missing_modules:
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
        if isinstance(exc_val, ModuleNotFoundError):
            message = (
                f"install module `{exc_val.name}` to use the snippet at "
                f"`{traceback.tb_frame.f_code.co_filename}`"
            )
            raise RuntimeError(message) from exc_val


def get_first_docstring_paragraph(obj: Any) -> str:
    """
    Get the first paragraph of the docstring of an object.

    Args:
        obj: Object whose first docstring paragraph to get.

    Returns:
        First paragraph of the object's docstring.
    """
    doc: Optional[str] = getattr(obj, "__doc__", None)
    if not doc:
        raise ValueError(f"{obj} does not have a docstring")
    doc, *_ = doc.split("\n\n")
    return textwrap.dedent(doc).strip().replace("\n", " ")

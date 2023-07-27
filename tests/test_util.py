import pytest
from snippets.util import get_first_docstring_paragraph, raise_for_missing_modules


def test_raise_for_missing_modules() -> None:
    with raise_for_missing_modules():
        __import__("sys")
    with pytest.raises(RuntimeError, match="install module `xxx`"), raise_for_missing_modules():
        __import__("xxx")


def test_get_first_docstring_paragraph() -> None:
    """
    This is
    some documentation.

    But there's more to it.
    """
    assert get_first_docstring_paragraph(test_get_first_docstring_paragraph) \
        == "This is some documentation."

    with pytest.raises(ValueError, match="does not have a docstring"):
        get_first_docstring_paragraph(test_raise_for_missing_modules)

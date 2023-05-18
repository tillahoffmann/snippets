import pytest
from snippets.util import raise_for_missing_modules


def test_raise_for_missing_modules() -> None:
    with raise_for_missing_modules():
        import sys  # noqa: F401
    with pytest.raises(RuntimeError, match="install module `xxx`"), raise_for_missing_modules():
        import xxx  # noqa: F401

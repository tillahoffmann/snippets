import multiprocessing
from pathlib import Path
import pytest
from snippets.call_with_timeout import call_with_timeout, _wrapper
import subprocess
import time
from typing import Optional


def test_timeout_expired() -> None:
    with pytest.raises(TimeoutError):
        call_with_timeout(1, time.sleep, 2)


def _target(x: int, error: Optional[Exception] = None) -> int:
    if error:
        raise error
    return x + 5


def test_timeout_met() -> None:
    assert call_with_timeout(1, _target, 37) == 42


def test_timeout_met_with_error() -> None:
    with pytest.raises(RuntimeError, match="custom value error"):
        call_with_timeout(1, _target, 9, ValueError("custom value error"))


def test_wrapper() -> None:
    queue = multiprocessing.Queue()
    _wrapper(queue, _target, 3)
    assert queue.get() == (True, 8)

    ex = ValueError("foo")
    _wrapper(queue, _target, 3, error=ex)
    success, (ex_recovered, tb) = queue.get()
    assert not success
    # These are not the same object because the exception has been pickled; we compare the message.
    assert str(ex_recovered) == str(ex)


def test_subprocess() -> None:
    # Compile the binary.
    here = Path(__file__).parent
    args = ["cc", "-o", "test_call_with_timeout", "test_call_with_timeout.c"]
    subprocess.check_call(args, cwd=here)

    # Make sure we raise a timeout error due to the infinite while loop.
    with pytest.raises(TimeoutError):
        call_with_timeout(1, subprocess.check_call, ["./test_call_with_timeout"], cwd=here)

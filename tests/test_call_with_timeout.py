import multiprocessing
from pathlib import Path
import psutil
import pytest
from snippets.call_with_timeout import _wrapper, call_with_timeout, ZombieProcessError
import subprocess
import time
from typing import List, Optional
from unittest import mock


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
    # These are not the same object because the exception has been pickled; we compare
    # the message.
    assert str(ex_recovered) == str(ex)


@pytest.fixture
def binary_path() -> Path:
    # Compile the binary and yield the path.
    here = Path(__file__).parent
    output = here / "test_call_with_timeout"
    args = ["cc", "-o", output, "test_call_with_timeout.c"]
    subprocess.check_call(args, cwd=here)
    yield output

    # Check if there are any lingering processes.
    process = subprocess.run(
        ["pgrep", "test_call_with_timeout"], text=True, capture_output=True
    )
    if process.returncode == 0:
        raise ZombieProcessError(
            "lingering `test_call_with_timeout` process with pids: %s", process.text
        )


def test_subprocess_success(binary_path: Path) -> None:
    process: subprocess.CompletedProcess = call_with_timeout(
        1, subprocess.run, [binary_path, "foobar"]
    )
    assert process.returncode == 42


def test_subprocess(binary_path: Path) -> None:
    # Make sure we raise a timeout error due to the infinite while loop.
    with pytest.raises(TimeoutError):
        call_with_timeout(1, subprocess.check_call, [binary_path])


def test_subprocess_ignoring_sigterm(binary_path: Path) -> None:
    # Make sure we raise a timeout error due to the infinite while loop.
    with mock.patch("snippets.call_with_timeout.JOIN_TIMEOUT", 1), pytest.raises(
        TimeoutError
    ):
        call_with_timeout(1, subprocess.check_call, [binary_path, "SIGTERM"])


def test_subprocess_ignoring_sigterm_zombie(binary_path: Path) -> None:
    # Make sure we raise a timeout error due to the infinite while loop.
    processes: List[psutil.Process] = []
    with mock.patch("snippets.call_with_timeout.JOIN_TIMEOUT", 0.5), mock.patch(
        "psutil.Process.kill", processes.append
    ), pytest.raises(ZombieProcessError):
        call_with_timeout(1, subprocess.check_call, [binary_path, "SIGTERM"])
    assert len(processes) == 1
    # Kill the process because we didn't due to the patch.
    for process in processes:
        process.kill()
    psutil.wait_procs(processes, 1)

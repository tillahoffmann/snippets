import multiprocessing
from queue import Empty
import traceback
from typing import Any, Callable
from .util import raise_for_missing_modules


with raise_for_missing_modules():
    import psutil


# Time to wait for processes that have been terminated or killed in seconds.
JOIN_TIMEOUT = 3


class ZombieProcessError(RuntimeError):
    """
    A zombie process could not be terminated or killed.
    """


def _wrapper(queue: multiprocessing.Queue, target: Callable, *args, **kwargs) -> None:
    """
    Wrapper to execute a function in a subprocess.
    """
    try:
        result = target(*args, **kwargs)
        success = True
    except Exception as ex:
        result = (ex, traceback.format_exc())
        success = False
    queue.put_nowait((success, result))


def call_with_timeout(timeout: float, target: Callable, *args, **kwargs) -> Any:
    """
    Call a target with a timeout and return its result.

    Args:
        timeout: Number of seconds to wait for a result.
        target: Callable to evaluate which must be serializable with :mod:`pickle`.
        *args: Positional arguments passed to :code:`target`.
        **kwargs: Keyword arguments passed to :code:`target`.

    Returns:
        Value returned by :code:`target(*args, **kwargs)`.

    Raises:
        TimeoutError: If the target does not complete within the timeout.

    Example:

        .. doctest::

            >>> from snippets.call_with_timeout import call_with_timeout
            >>> from time import sleep

            >>> call_with_timeout(1.0, " ".join, ["Hello", "world!"])
            'Hello world!'

            >>> call_with_timeout(1.0, sleep, 2.0)
            Traceback (most recent call last):
                ...
            TimeoutError: call to <built-in function sleep> did not complete in 1.0
            seconds
    """
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_wrapper, args=(queue, target, *args), kwargs=kwargs, daemon=True
    )
    process.start()

    try:
        success, result = queue.get(timeout=timeout)
    except Empty:
        raise TimeoutError(f"call to {target} did not complete in {timeout} seconds")
    finally:
        if process.is_alive():
            # Try to terminate the process and all its children. If not possible, kill
            # them (https://stackoverflow.com/a/4229404/1150961).
            process = psutil.Process(process.pid)
            processes = [process, *process.children(recursive=True)]
            for func in [psutil.Process.terminate, psutil.Process.kill]:
                for process in processes:
                    func(process)
                _, processes = psutil.wait_procs(processes, timeout=JOIN_TIMEOUT)
                if not processes:
                    break
            if processes:
                raise ZombieProcessError(f"processes still alive: {len(processes)}")
    if not success:
        ex, tb = result
        raise RuntimeError(tb) from ex
    return result

import multiprocessing
import psutil
from queue import Empty
import traceback
from typing import Any, Callable


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
        target: Function to call.
        *args: Positional arguments passed to :code:`target`.
        **kwargs: Keyword arguments passed to :code:`target`.

    Returns:
        result: Return value of :code:`target`.

    Raises:
        TimeoutError: If the target does not complete within the timeout.
    """
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_wrapper, args=(queue, target, *args), kwargs=kwargs,
                                      daemon=True)
    process.start()

    try:
        success, result = queue.get(timeout=timeout)
    except Empty:
        raise TimeoutError(f"call to {target} did not complete in {timeout} seconds")
    finally:
        if process.is_alive():
            # Try to terminate the process and all its children. If not possible, kill them.
            # (https://stackoverflow.com/a/4229404/1150961).
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

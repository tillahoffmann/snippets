import multiprocessing
import psutil
from queue import Empty
import traceback
from typing import Any, Callable


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
            # Kill the process and all its children (https://stackoverflow.com/a/4229404/1150961).
            process = psutil.Process(process.pid)
            processes = [process, *process.children(recursive=True)]
            for process in processes:
                process.kill()
            _, still_alive = psutil.wait_procs(processes, timeout=3)
            if still_alive:
                raise RuntimeError(f"{len(still_alive)} processes are still alive")
    if not success:
        ex, tb = result
        raise RuntimeError(tb) from ex
    return result

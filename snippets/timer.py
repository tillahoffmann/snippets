from __future__ import annotations
import time
from typing import Optional


class Timer:
    """
    Timer that can be used as a context.

    Example:

        .. doctest::

            >>> from snippets.timer import Timer
            >>> from time import sleep

            >>> with Timer() as timer:
            ...     sleep(1)
            >>> timer.duration
            1...
    """
    def __init__(self) -> None:
        self.start: Optional[float] = None
        self.end: Optional[float] = None

    def __enter__(self) -> Timer:
        self.start = time.time()
        return self

    def __exit__(self, *args) -> None:
        self.end = time.time()

    @property
    def duration(self) -> float:
        """
        Duration for which the timer was active.
        """
        if not self.start:
            raise RuntimeError("timer has not started")
        return (self.end or time.time()) - self.start

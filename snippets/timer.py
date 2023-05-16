from __future__ import annotations
import time


class Timer:
    def __init__(self) -> None:
        self.start = None
        self.end = None

    def __enter__(self) -> Timer:
        self.start = time.time()
        return self

    def __exit__(self, *args) -> None:
        self.end = time.time()

    @property
    def duration(self) -> float:
        if not self.start:
            raise RuntimeError("timer has not started")
        return (self.end or time.time()) - self.start

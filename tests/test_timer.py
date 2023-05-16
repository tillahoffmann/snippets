from snippets.timer import Timer
import time


def test_timer() -> None:
    import pytest

    with pytest.raises(RuntimeError, match="not started"):
        Timer().duration

    with Timer() as t:
        time.sleep(0.5)

    assert 0.5 < t.duration < 0.6

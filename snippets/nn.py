from typing import Literal, Optional, Type, TypeVar
from .util import raise_for_missing_modules

with raise_for_missing_modules():
    import torch


S = TypeVar("S", bound="StopOnPlateau")


class StopOnPlateau:
    """
    Stop training when a monitored quantity plateaus akin to
    :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`.

    Args:
        mode: One of :code:`min` or :code:`max`. In :code:`min` mode, training will stop
            when the quantity monitored has stopped decreasing; in :code:`max` mode
            training will stop when the quantity monitored has stopped increasing.
        patience: Number of epochs without improvement after which training will stop.
        threshold: Threshold for measuring the new optimum, to only focus on significant
            changes.
        threshold_mode: One of :code:`rel` or :code:`abs`. For :code:`rel`,
            :code:`dynamic_threshold = best * (1 + threshold)` in :code:`max` mode or
            :code:`dynamic_threshold = best * (1 - threshold)` in :code:`min` mode. For
            :code:`abs`, :code:`dynamic_threshold = best + threshold` in :code:`max`
            mode or :code:dynamic_threshold = best - threshold` in :code:`min` mode.

    Example:

        .. doctest::

            >>> from snippets.nn import StopOnPlateau

            >>> stop = StopOnPlateau(patience=3)
            >>> for _ in range(5):
            ...     stop.step(3)
            False
            False
            False
            False
            True
    """

    def __init__(
        self,
        mode: Literal["min", "max"] = "min",
        patience: int = 20,
        threshold: float = 1e-4,
        threshold_mode: Literal["rel", "abs"] = "rel",
    ) -> None:
        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        if self.mode == "min":
            self.best = float("inf")
        elif self.mode == "max":
            self.best = -float("inf")
        else:
            raise ValueError(self.mode)
        self.num_bad_epochs = 0

    @property
    def stop(self) -> bool:
        """
        If the process should stop.
        """
        return self.num_bad_epochs > self.patience

    def step(self, value: float) -> bool:
        """
        Update the state with a new value.

        Args:
            value: Value of the monitored quantity.

        Returns:
            If the process should stop.
        """
        value = float(value)
        if self.is_better(self.best, value):
            self.num_bad_epochs = 0
            self.best = value
        else:
            self.num_bad_epochs += 1
        return self.stop

    def is_better(self, best: float, candidate: float) -> bool:
        """
        Check if the candidate is better than the current best value.

        Args:
            best: Reference value to compare with.
            candidate: Candidate value to check.

        Returns:
            If the candidate is better than the current best value.
        """
        if self.mode == "min" and self.threshold_mode == "rel":
            return candidate < best * (1 - self.threshold)
        elif self.mode == "min" and self.threshold_mode == "abs":
            return candidate < best - self.threshold
        elif self.mode == "max" and self.threshold_mode == "rel":
            return candidate > best * (1 + self.threshold)
        elif self.mode == "max" and self.threshold_mode == "abs":
            return candidate > best + self.threshold
        else:
            raise ValueError(self.mode, self.threshold_mode)

    @classmethod
    def from_scheduler(
        cls: Type[S],
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        patience_factor: float = 1,
        patience: Optional[int] = None,
    ) -> S:
        """
        Create a :class:`.StopOnPlateau` instance configured based on a
        :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`.

        Args:
            scheduler: Learning rate scheduler whose configuration to copy.
            patience_factor: Factor by which to scale the patience of the learning rate
                scheduler.
            patience: Patience of the instance (takes precedence over
                :code:`patience_factor`).

        Returns:
            Instance configured based on supplied
            :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`.
        """
        patience = patience or int(patience_factor * scheduler.patience)
        return cls(
            scheduler.mode, patience, scheduler.threshold, scheduler.threshold_mode
        )


class Affine(torch.nn.Module):
    """
    Apply a *fixed* affine transform :math:`y = x A^\\intercal + b` akin to
    :class:`torch.nn.Linear`.

    Args:
        loc: Offset of the transform.
        scale: Scale (matrix) of the transform.

    Example:

        .. doctest::

            >>> from snippets.nn import Affine
            >>> import torch

            >>> affine = Affine(loc=1, scale=2)
            >>> x = torch.arange(3)
            >>> affine(x)
            tensor([1., 3., 5.])
    """

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        self.loc = torch.as_tensor(loc, dtype=dtype)
        self.scale = torch.as_tensor(scale, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""  # Hide inherited documentation.
        if self.scale.ndim == 0:
            return x * self.scale + self.loc
        return torch.nn.functional.linear(x, self.scale, self.loc)

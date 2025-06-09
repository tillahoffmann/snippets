import math
from .util import raise_for_missing_modules

with raise_for_missing_modules():
    import torch
    import torch.utils.data


class TensorDataLoader:
    """
    Fast data loader for torch tensor datasets, fusing
    :class:`torch.utils.data.TensorDataset` and :class:`torch.utils.data.DataLoader`.

    :class:`torch.utils.data.DataLoader` is slow for
    :class:`torch.utils.data.TensorDataset`\\ s because it iterates over elements of the
    dataset. Tensor-specific slicing implemented by :class:`~.TensorDataLoader` is
    typically orders of magnitude faster for data that fit in memory.

    Args:
        dataset: Dataset to load from.
        batch_size: Number of samples per batch.
        shuffle: Shuffle dataset before batching.

    Example:

        >>> from snippets.tensor_data_loader import TensorDataLoader
        >>> import torch
        >>> from torch.utils.data import TensorDataset

        >>> tensors = torch.randn(13, 5), torch.randn(13)
        >>> dataset = TensorDataset(*tensors)
        >>> loader = TensorDataLoader(dataset, batch_size=7)
        >>> for X, y in loader:
        ...     X.shape, y.shape
        (torch.Size([7, 5]), torch.Size([7]))
        (torch.Size([6, 5]), torch.Size([6]))
    """

    def __init__(
        self,
        dataset: torch.utils.data.TensorDataset,
        batch_size: int = 1,
        shuffle: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        n = len(self.dataset)
        permutation = torch.randperm(n) if self.shuffle else None
        offset = 0
        while offset < n:
            yield tuple(
                (
                    x[permutation[offset : offset + self.batch_size]]
                    if self.shuffle
                    else x[offset : offset + self.batch_size]
                )
                for x in self.dataset.tensors
            )
            offset += self.batch_size

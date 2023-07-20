import math
from .util import raise_for_missing_modules

with raise_for_missing_modules():
    import torch
    import torch.utils.data


class TensorDataLoader:
    """
    Fast data loader for tensor datasets.

    Args:
        dataset: Dataset to load from.
        batch_size: Number of samples per batch.
        shuffle: Shuffle dataset before batching.
    """
    def __init__(self, dataset: torch.utils.data.TensorDataset, batch_size: int = 1,
                 shuffle: bool = False) -> None:
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
                x[permutation[offset:offset + self.batch_size]] if self.shuffle else
                x[offset:offset + self.batch_size] for x in self.dataset.tensors
            )
            offset += self.batch_size

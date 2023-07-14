import pickle
from typing import Any, Iterable, Iterator, List, TypeVar

from .util import raise_for_missing_modules

with raise_for_missing_modules():
    import torch
    import torch.utils.data


T = TypeVar("T")


class BatchedDataset(torch.utils.data.IterableDataset[T]):
    """
    Dataset comoprising batches saved on disk.

    Args:
        filenames: Filenames each corresponding to a batch of data.
        length: Total number of samples in the dataset.
        shuffle: Shuffle filenames for each iteration and, for each indexable batch, elements within
            the bach.
        num_concurrent: Number of concurrent batches to load (only reasonable if shuffling is
            enabled).

    Example:

        .. doctest::

            >>> a
    """
    def __init__(self, filenames: List[str], length: int | None = None,
                 shuffle: bool = False, num_concurrent: int = 1) -> None:
        super().__init__()

        if num_concurrent > 1 and not shuffle:
            raise ValueError("Concurrent batches are only reasonable with shuffling.")

        self.filenames = filenames
        self.length = length
        self.num_concurrent = num_concurrent
        self.shuffle = shuffle

    def __len__(self) -> int:
        if self.length is None:
            raise TypeError(f"TypeError: object of type '{self.__class__}' has no len()")
        return self.length

    def __iter__(self) -> Iterator[T]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise NotImplementedError("Batched datasets do not support multiple workers.")

        # Permute filenames if desired and initialize the iterators.
        filenames = self.filenames
        if self.shuffle:
            filenames = [filenames[i] for i in torch.randperm(len(filenames))]
        iterators = [iter(self.load_dataset(filename)) for filename in
                     filenames[:self.num_concurrent]]
        offset = len(iterators)

        while iterators:
            # Yield from one of the iterators.
            iterator = iterators[torch.randint(0, len(iterators), ()) if self.shuffle else 0]
            try:
                yield next(iterator)
            except StopIteration:
                iterators.remove(iterator)
                if offset < len(filenames):
                    iterators.append(iter(self.load_dataset(filenames[offset])))
                    offset += 1

    def __getitem__(self, __key: Any) -> T:
        raise NotImplementedError("Batched datasets do not support indexing.")

    def load_dataset(self, filename: str) -> Iterable[T]:
        with open(filename, "rb") as fp:
            return pickle.load(fp)

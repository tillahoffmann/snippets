from pathlib import Path
import pickle
import pytest
from snippets.batched_dataset import BatchedDataset
import torch.utils.data
from typing import List


@pytest.fixture
def filenames(tmp_path: Path) -> List[str]:
    sizes = [3, 5, 7, 4, 13]
    offset = 0
    filenames = []
    for i, size in enumerate(sizes):
        dataset = torch.utils.data.TensorDataset(torch.randn(size, 2), torch.arange(size) + offset)
        offset += size

        filename = tmp_path / f"{i}.pkl"
        with open(filename, "wb") as fp:
            pickle.dump(dataset, fp)
        filenames.append(filename)
    return filenames


@pytest.mark.parametrize("shuffle", [False, True])
def test_batched_dataset(filenames: List[str], shuffle: bool, tmp_path: Path) -> None:
    dataset = BatchedDataset(filenames, shuffle=shuffle)

    # Make sure this works a few times.
    for _ in range(3):
        parts = []
        for X, y in dataset:
            assert X.shape == (2,)
            assert y.shape == ()
            parts.append(y)
        parts = torch.as_tensor(parts)

        expected = torch.arange(parts.numel())
        if shuffle:
            with pytest.raises(AssertionError, match="Tensor-likes are not equal!"):
                torch.testing.assert_close(parts, expected)
        else:
            torch.testing.assert_close(parts, expected)


def test_batched_dataset_invalid_shuffle():
    with pytest.raises(ValueError, match="Concurrent batches are only reasonable"):
        BatchedDataset([], shuffle=False, num_concurrent=2)
    BatchedDataset([], shuffle=True, num_concurrent=2)


def test_batched_dataset_length():
    assert len(BatchedDataset([], length=23)) == 23
    with pytest.raises(TypeError, match="has no len()"):
        len(BatchedDataset([]))


def test_batched_dataset_loader(filenames):
    dataset = BatchedDataset(filenames)
    loader = torch.utils.data.DataLoader(dataset, batch_size=7)
    for i, (X, y) in enumerate(loader):
        assert X.shape == (7, 2) or i == 4
    assert i == 4

    with pytest.raises(ValueError, match="expected unspecified shuffle option"):
        torch.utils.data.DataLoader(dataset, shuffle=True)

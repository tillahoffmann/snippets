import itertools as it
import pytest
from snippets.tensor_data_loader import TensorDataLoader
import torch
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture(params=it.product(
    # Batch sizes.
    [10, 17],
    # Tensor shapes.
    [
        ((),),  # Just a scalar.
        ((9,),),  # A single vector per element.
        ((7,), ()),  # A feature-outcome combination.
        ((3, 4), (5, 7, 8), ()),  # A complex combination.
    ],
))
def dataset(request: pytest.FixtureRequest) -> TensorDataset:
    size, shapes = request.param
    tensors = [torch.randn(size, *shape) for shape in shapes]
    return TensorDataset(*tensors)


@pytest.mark.parametrize("batch_size", [1, 2, 13])
def test_tensor_data_loader_equivalence(dataset: TensorDataset, batch_size: int) -> None:
    loader1 = TensorDataLoader(dataset, batch_size)
    loader2 = DataLoader(dataset, batch_size)
    assert len(loader1) == len(loader2), "number of batches does not match"

    for batch1, batch2 in zip(loader1, loader2):
        assert len(batch1) == len(batch2), "number of tensors per batch does not match"

        for tensor1, tensor2 in zip(batch1, batch2):
            torch.testing.assert_close(tensor1, tensor2)


@pytest.mark.parametrize("batch_size", [1, 7])
def test_tensor_data_loader_shuffle(dataset: TensorDataset, batch_size: int) -> None:
    # Create a new dataset with an index prepended.
    dataset = TensorDataset(torch.arange(len(dataset)), *dataset.tensors)
    loader = TensorDataLoader(dataset, batch_size, shuffle=True)

    # Iterate over batches and transpose them to get the tensors back.
    indices, *transposed = [torch.concat(batches, axis=0) for batches in zip(*loader)]
    order = torch.argsort(indices)

    assert torch.diff(indices).unique().numel() > 1, "indices are not shuffled"

    for actual, expected in zip(transposed, dataset.tensors[1:]):
        torch.testing.assert_close(actual[order], expected)

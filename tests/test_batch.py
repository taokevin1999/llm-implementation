import torch

from llm_impl.data import get_batch


def test_batch_shapes():
    data = torch.arange(100)

    x, y = get_batch(
        data,
        batch_size=4,
        block_size=8,
    )

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)


def test_targets_are_shifted_inputs():
    data = torch.arange(100)

    x, y = get_batch(
        data,
        batch_size=4,
        block_size=8,
    )

    assert torch.equal(x[:, 1:], y[:, :-1])


def test_batch_dtype():
    data = torch.arange(100, dtype=torch.long)

    x, y = get_batch(
        data,
        batch_size=4,
        block_size=8,
    )

    assert x.dtype == torch.long
    assert y.dtype == torch.long
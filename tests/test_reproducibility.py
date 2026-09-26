import torch

from llm_impl.utils import set_seed

def test_same_seed_same_tensor():
    set_seed(42)
    x1 = torch.randn(5)

    set_seed(42)
    x2 = torch.randn(5)

    assert torch.equal(x1, x2)


def test_different_seed_different_tensor():
    set_seed(42)
    x1 = torch.randn(5)

    set_seed(43)
    x2 = torch.randn(5)

    assert not torch.equal(x1, x2)
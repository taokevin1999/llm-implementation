import torch

from llm_impl.utils import tensor_mean

def test_tensor_mean():
    x = torch.tensor([1.0, 2.0, 3.0])
    result = tensor_mean(x)

    assert result.item() == 2.0
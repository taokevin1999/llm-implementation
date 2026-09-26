import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)

def system_info() -> dict:
    info = {
        "python_torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda

    return info

def tensor_mean(x: torch.Tensor) -> torch.Tensor:
    return x.mean()
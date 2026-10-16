from pathlib import Path

import torch


def save_checkpoint(
    path,
    model,
    optimizer,
    config,
    stoi,
    step,
    metadata,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "stoi": stoi,
        "step": step,
        "metadata": metadata,
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }

    # Finish writing a temporary file before replacing the destination.
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)
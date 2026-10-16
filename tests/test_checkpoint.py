import copy

import torch

from llm_impl.attention import MultiHeadAttention
from llm_impl.checkpoint import save_checkpoint
from llm_impl.model import DecoderLM


def test_checkpoint_restores_predictions(tmp_path):
    torch.manual_seed(42)

    config = {
        "vocab_size": 10,
        "n_embed": 16,
        "num_heads": 2,
        "n_layers": 2,
        "block_size": 8,
        "dropout": 0.1,
    }

    attentions = [
        MultiHeadAttention(
            d_model=config["n_embed"],
            num_heads=config["num_heads"],
            block_size=config["block_size"],
        )
        for _ in range(config["n_layers"])
    ]

    model = DecoderLM(
        vocab_size=config["vocab_size"],
        n_embed=config["n_embed"],
        block_size=config["block_size"],
        attentions=attentions,
        dropout=config["dropout"],
    )

    optimizer = torch.optim.AdamW(model.parameters())

    probe_x = torch.tensor([
        [2, 5, 2, 7],
        [1, 3, 4, 6],
    ])

    model.eval()

    with torch.no_grad():
        expected_logits, _ = model(probe_x)

    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        stoi={str(i): i for i in range(10)},
        step=0,
        metadata={"purpose": "checkpoint test"},
    )

    saved = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    restored_model = copy.deepcopy(model)

    # Change the parameters so loading must actually restore them.
    with torch.no_grad():
        for parameter in restored_model.parameters():
            parameter.zero_()

        changed_logits, _ = restored_model(probe_x)

    assert not torch.allclose(changed_logits, expected_logits)

    restored_model.load_state_dict(saved["model_state"])
    restored_model.eval()

    with torch.no_grad():
        restored_logits, _ = restored_model(probe_x)

    torch.testing.assert_close(
        restored_logits,
        expected_logits,
    )
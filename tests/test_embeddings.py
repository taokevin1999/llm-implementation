import pytest
import torch

from llm_impl.embeddings import TransformerInput


def test_embedding_lookup_and_positions():
    model = TransformerInput(
        vocab_size=10,
        n_embed=8,
        block_size=16,
    )

    idx = torch.tensor([
        [2, 5, 2, 7],
        [2, 1, 6, 7],
    ], dtype=torch.long)

    h = model(idx)

    assert h.shape == (2, 4, 8)

    # Same token at the same position in different sequences.
    torch.testing.assert_close(h[0, 0], h[1, 0])

    # Token 2 occurs at positions 0 and 2.
    # Its token vector cancels when we subtract the representations.
    expected_difference = (
        model.position_embedding.weight[0]
        - model.position_embedding.weight[2]
    )

    torch.testing.assert_close(
        h[0, 0] - h[0, 2],
        expected_difference,
    )


def test_sequence_too_long():
    model = TransformerInput(
        vocab_size=10,
        n_embed=8,
        block_size=4,
    )

    idx = torch.zeros((2, 5), dtype=torch.long)

    with pytest.raises(ValueError, match="Sequence length exceeds block_size"):
        model(idx)
import torch

from llm_impl.attention import AttentionHead, MultiHeadAttention

def test_attention_output_shape():
    B = 4
    T = 8
    d_model = 32
    head_size = 16

    x = torch.randn(B, T, d_model)

    head = AttentionHead(
        d_model=d_model,
        head_size=head_size,
        block_size=T,
    )

    out = head(x)

    assert out.shape == (
        B,
        T,
        head_size,
    )


def test_attention_is_causal():
    torch.manual_seed(0)

    B = 1
    T = 5
    d_model = 8
    head_size = 4

    x = torch.randn(B, T, d_model)

    head = AttentionHead(
        d_model=d_model,
        head_size=head_size,
        block_size=T,
    )

    out1 = head(x)

    x_modified = x.clone()

    x_modified[:, 4, :] += 1000

    out2 = head(x_modified)

    assert torch.allclose(
        out1[:, :4, :],
        out2[:, :4, :],
        atol=1e-5,
    )

def test_attention_gradients():
    x = torch.randn(
        2,
        5,
        16,
        requires_grad=True,
    )

    mha = MultiHeadAttention(
        d_model=16,
        num_heads=4,
        block_size=5,
    )

    out = mha(x)

    loss = out.sum()
    loss.backward()

    for parameter in mha.parameters():
        assert parameter.grad is not None

def test_multihead_output_shape():
    B = 4
    T = 8
    d_model = 32
    num_heads = 4

    x = torch.randn(
        B,
        T,
        d_model,
    )

    mha = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        block_size=T,
    )

    out = mha(x)

    assert out.shape == (
        B,
        T,
        d_model,
    )

def test_multihead_is_causal():
    torch.manual_seed(0)

    B = 1
    T = 5
    d_model = 16
    num_heads = 4

    x = torch.randn(
        B,
        T,
        d_model,
    )

    mha = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        block_size=T,
    )

    out1 = mha(x)

    x_modified = x.clone()
    x_modified[:, -1, :] += 1000

    out2 = mha(x_modified)

    assert torch.allclose(
        out1[:, :-1, :],
        out2[:, :-1, :],
        atol=1e-5,
    )
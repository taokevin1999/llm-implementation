from llm_impl.data import build_vocab, encode, decode, tokenize
import torch

def test_encode_decode():
    text = "hello world"
    stoi, itos = build_vocab(text)

    assert decode(encode(text, stoi), itos) == text


def test_token_dtype():
    text = "hello world"
    stoi, _ = build_vocab(text)

    tokens = tokenize(text, stoi)

    assert tokens.dtype == torch.long
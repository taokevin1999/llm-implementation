import torch

text = "the quick brown fox jumps over the lazy dog"

chars = sorted(set(text))

vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)} #map ch to id
itos = {i: ch for i, ch in enumerate(chars)} #map id to ch

print(stoi)
print(itos)

def encode(s):
    return [stoi[ch] for ch in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

ids = encode("the")

print(ids)
print(decode(ids))

data = torch.tensor(
    encode(text),
    dtype=torch.long,
)

print(data)
print(data.shape)
print(data.dtype)
import torch

def build_vocab(text):
    chars = sorted(set(text))

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    return stoi, itos


def encode(text, stoi):
    return [stoi[ch] for ch in text]


def decode(ids, itos):
    return "".join(itos[i] for i in ids)


def tokenize(text, stoi):
    return torch.tensor(
        encode(text, stoi),
        dtype=torch.long,
    )

def get_batch(data, batch_size, block_size):
    starts = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,),
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in starts
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in starts
    ])

    return x, y

def get_batch_from_split(
    split,
    train_data,
    val_data,
    batch_size,
    block_size,
):
    data = train_data if split == "train" else val_data

    return get_batch(
        data,
        batch_size,
        block_size,
    )
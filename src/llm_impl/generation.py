import torch


def generate(model, idx, max_new_tokens, temperature=1.0):
    if idx.ndim != 2 or idx.shape[1] == 0:
        raise ValueError("idx must have shape (B, T) with T >= 1")

    if temperature < 0:
        raise ValueError("temperature must be nonnegative")

    was_training = model.training
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Use only the most recent supported context.
            context = idx[:, -model.block_size:]

            logits, _ = model(context)       # (B, T_context, V)
            next_logits = logits[:, -1, :]   # (B, V)

            if temperature == 0:
                next_id = next_logits.argmax(
                    dim=-1,
                    keepdim=True,
                )                            # (B, 1)
            else:
                probabilities = torch.softmax(
                    next_logits / temperature,
                    dim=-1,
                )                            # (B, V)

                next_id = torch.multinomial(
                    probabilities,
                    num_samples=1,
                )                            # (B, 1)

            idx = torch.cat((idx, next_id), dim=1)

    model.train(was_training)
    return idx
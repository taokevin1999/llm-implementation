import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import torch
import argparse

from llm_impl.attention import MultiHeadAttention
from llm_impl.data import build_vocab, tokenize, get_batch
from llm_impl.model import DecoderLM
from llm_impl.checkpoint import save_checkpoint

#determine if resuming or training from 0
parser = argparse.ArgumentParser()

parser.add_argument(
    "--resume",
    type=Path,
    default=None,
)

parser.add_argument(
    "--steps",
    type=int,
    default=None,
)

args = parser.parse_args()

saved = None

if args.resume is not None:
    saved = torch.load(
        args.resume,
        map_location="cpu",
        weights_only=True,
    )

    required = {
        "cpu_rng_state",
        "cuda_rng_states",
        "metadata",
    }

    missing = required - saved.keys()

    if missing:
        raise ValueError(
            f"Checkpoint lacks recovery information: {sorted(missing)}. "
            "Use a checkpoint created with the Part A helper."
        )
    
torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

#get data
data_path = Path("storage/datasets/tinyshakespeare/input.txt")
text = data_path.read_text(encoding="utf-8")

split_index = int(0.9 * len(text))
train_text = text[:split_index]
val_text = text[split_index:]

# Build the vocabulary from training text only.
stoi, itos = build_vocab(train_text)

unseen = set(val_text) - set(stoi)
if unseen:
    raise ValueError(f"Validation contains unseen characters: {unseen!r}")

train_data = tokenize(train_text, stoi)
val_data = tokenize(val_text, stoi)

V = len(stoi)

print("Device:", device)
print("Vocabulary size:", V)
print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))

#model settings
config = {
    "vocab_size": V,
    "n_embed": 128,
    "num_heads": 4,
    "n_layers": 3,
    "block_size": 64,
    "dropout": 0.1,
    "batch_size": 16,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "steps": 1000,
    "eval_interval": 100,
    "eval_batches": 20,
}

# restore from previous training
if saved is not None:
    config = saved["config"].copy()

    if stoi != saved["stoi"]:
        raise ValueError("Tokenizer mapping differs from checkpoint")

    if V != config["vocab_size"]:
        raise ValueError("Vocabulary size differs from checkpoint")

start_step = 0 if saved is None else saved["step"]

target_steps = (
    config["steps"]
    if args.steps is None
    else args.steps
)

if target_steps <= start_step:
    raise ValueError(
        "Target steps must exceed the number of completed updates"
    )

config["steps"] = target_steps

#declaring model
attentions = [
    MultiHeadAttention(
        d_model=config["n_embed"],
        num_heads=config["num_heads"],
        block_size=config["block_size"],
    )
    for _ in range(config["n_layers"])
]

model = DecoderLM(
    vocab_size=V,
    n_embed=config["n_embed"],
    block_size=config["block_size"],
    attentions=attentions,
    dropout=config["dropout"],
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
)

#restore optimizer state
if saved is not None:
    model.load_state_dict(saved["model_state"])
    optimizer.load_state_dict(saved["optimizer_state"])

print(f"Training from step {start_step} to {target_steps}")

#fixing evaluation batches
def make_eval_batches(data, seed):
    generator = torch.Generator().manual_seed(seed)
    batches = []

    B = config["batch_size"]
    T = config["block_size"]

    for _ in range(config["eval_batches"]):
        starts = torch.randint(
            len(data) - T,
            (B,),
            generator=generator,
        )

        x = torch.stack([data[i:i + T] for i in starts])
        y = torch.stack([data[i + 1:i + T + 1] for i in starts])

        batches.append((x, y))

    return batches


train_eval_batches = make_eval_batches(train_data, seed=123)
val_eval_batches = make_eval_batches(val_data, seed=456)

#pertoken loss calculation
def evaluate(model, batches):
    was_training = model.training
    model.eval()

    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in batches:
            x = x.to(device)
            y = y.to(device)

            _, loss = model(x, y)

            n_tokens = y.numel()
            total_nll += loss.item() * n_tokens
            total_tokens += n_tokens

    model.train(was_training)

    mean_loss = total_nll / total_tokens

    return mean_loss, math.exp(mean_loss), total_tokens


#saving metadata and training
run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
run_dir = Path("storage/outputs/week5") / run_name
run_dir.mkdir(parents=True, exist_ok=False)
checkpoint_dir = Path("storage/checkpoints/week5") / run_name

metadata = {
    "config": config,
    "seed": 42,
    "torch_version": str(torch.__version__),
    "tokenizer": "character",
    "stoi": stoi,
    "dataset": str(data_path),
    "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    "split_index": split_index,
    "train_tokens": len(train_data),
    "val_tokens": len(val_data),
    "evaluation_seeds": {"train": 123, "validation": 456},
}

#add information about which run we are resuming from
if saved is not None:
    previous_metadata = saved["metadata"]

    for key in ("dataset_sha256", "split_index"):
        if metadata[key] != previous_metadata[key]:
            raise ValueError(f"Dataset setting differs: {key}")

metadata["start_step"] = start_step
metadata["target_steps"] = target_steps
metadata["resumed_from"] = (
    None if args.resume is None else str(args.resume)
)

#write the actual json metadata
(run_dir / "metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

#restore cuda RNG before training

if saved is not None:
    cuda_states = saved["cuda_rng_states"]

    if device.type == "cuda":
        if cuda_states is None:
            raise ValueError(
                "Checkpoint has no CUDA RNG state; "
                "this recovery exercise requires the original device setup"
            )

        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from checkpoint")

        torch.cuda.set_rng_state_all(cuda_states)

    elif cuda_states is not None:
        raise ValueError(
            "Checkpoint used CUDA; resume on CUDA for this recovery exercise"
        )

    torch.set_rng_state(saved["cpu_rng_state"])

model.train()

#actual training loop

with (run_dir / "metrics.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "step", "train_loss", "val_loss",
        "val_ppl", "val_tokens_scored",
    ])

    for step in range(start_step, target_steps + 1):
        if (
            step == start_step
            or step % config["eval_interval"] == 0
            or step == target_steps
        ):
            train_loss, _, _ = evaluate(model, train_eval_batches)
            val_loss, val_ppl, val_tokens = evaluate(model, val_eval_batches)

            writer.writerow([
                step, train_loss, val_loss, val_ppl, val_tokens,
            ])
            file.flush()

            print(
                f"Step {step:4d} | "
                f"train {train_loss:.3f} | "
                f"val {val_loss:.3f} | "
                f"val PPL {val_ppl:.2f}"
            )

            save_checkpoint(
                path=checkpoint_dir / f"step_{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                stoi=stoi,
                step=step,
                metadata=metadata,
            )

        if step == target_steps:
            break

        x, y = get_batch(
            train_data,
            config["batch_size"],
            config["block_size"],
        )
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")

        loss.backward()
        optimizer.step()

#saved trained model
save_checkpoint(
    path=checkpoint_dir / "last.pt",
    model=model,
    optimizer=optimizer,
    config=config,
    stoi=stoi,
    step=config["steps"],
    metadata=metadata,
)

print("Metrics:", run_dir)
print("Checkpoint:", checkpoint_dir / "last.pt")

import copy

# Use a fixed validation input.
probe_x = val_eval_batches[0][0].to(device)

model.eval()

with torch.no_grad():
    expected_logits, _ = model(probe_x)

saved = torch.load(
    checkpoint_dir / "last.pt",
    map_location="cpu",
    weights_only=True,
)

restored_model = copy.deepcopy(model)

# Deliberately change its parameters so the check tests restoration.
with torch.no_grad():
    for parameter in restored_model.parameters():
        parameter.zero_()

restored_model.load_state_dict(saved["model_state"])
restored_model.eval()

with torch.no_grad():
    restored_logits, _ = restored_model(probe_x)

torch.testing.assert_close(restored_logits, expected_logits)

print("Checkpoint prediction recovery passed.")
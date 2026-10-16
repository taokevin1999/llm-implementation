import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from llm_impl.attention import MultiHeadAttention
from llm_impl.data import encode, decode
from llm_impl.generation import generate
from llm_impl.model import DecoderLM


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
args = parser.parse_args()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint = torch.load(
    args.checkpoint,
    map_location="cpu",
    weights_only=True,
)

config = checkpoint["config"]
stoi = checkpoint["stoi"]
itos = {index: character for character, index in stoi.items()}

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

model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

prompts = ["ROMEO:\n", "JULIET:\n"]
temperatures = [0.0, 0.7, 1.0]
max_new_tokens = 300
seed = 42

results = []

for prompt in prompts:
    for temperature in temperatures:
        # Reset immediately before generation.
        torch.manual_seed(seed)

        idx = torch.tensor(
            [encode(prompt, stoi)],
            dtype=torch.long,
            device=device,
        )

        output = generate(
            model,
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        generated_text = decode(output[0].cpu().tolist(), itos)

        results.append({
            "prompt": prompt,
            "temperature": temperature,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "text": generated_text,
        })

        print(f"\nTemperature: {temperature}")
        print(generated_text)


run_name = args.checkpoint.parent.name
output_dir = Path("storage/outputs/week5") / run_name
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
output_path = output_dir / f"generation_{timestamp}.json"

output_path.write_text(
    json.dumps(
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_step": checkpoint["step"],
            "device": str(device),
            "torch_version": str(torch.__version__),
            "samples": results,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print("\nSaved samples:", output_path)
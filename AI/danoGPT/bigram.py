#!/usr/bin/env python3
"""
Simple bigram language model, implemented using pytorch embeddings.
A helper first level to build up to understanding GPTs.
"""

import argparse
import os
from time import perf_counter
from typing import Optional, Tuple

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.nn import functional as F
from tqdm import tqdm

import utils

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# hyperparameters
batch_size = 10  # num independent sequences to process in parallel
block_size = 8  # max context length for predictions
device = "cpu"
lr = 1e-3


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--input-path",
        "-i",
        type=str,
        help="Path to .txt file for training/evaluation",
        default=os.path.join(SCRIPT_DIR, "input.txt"),
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        help="Path to .pdf file for plot",
        default="bigram_loss.pdf",
    )
    parser.add_argument(
        "--steps", "-n", type=int, help="Number of training steps", default=10_000
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
        default=42,
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        help="Device to use for training (auto detects if not provided)",
        default=None,
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    global device
    if args.device is None:
        device = utils.get_device()
    print("using device=", device)

    with open(args.input_path, "r") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"{vocab_size=}")

    encode, decode = utils.get_encoder_decoder(chars)

    # let's tokenize our dataset
    all_data = torch.tensor(encode(text), dtype=torch.long).to(device)
    print(f"full dataset: {all_data.shape=}, {all_data.dtype=}")
    train_data, val_data, test_data = split_dataset(all_data)
    data_map = {
        "train": train_data,
        "test": test_data,
        "val": val_data,
    }

    xb, yb = get_batch(data_map, "train")
    # print(f"{xb.shape=}")  # inputs: (4, 8)
    # print(f"{yb.shape=}")  # target: (4, 8)

    # for example
    # print(f"inputs: '{decode(xb[0].tolist())}'")
    # print(f"target: '{decode(yb[0].tolist())}'")

    # (22:45 in tutorial)
    model = BigramLangModel(vocab_size).to(device)
    logits, loss = model(xb, yb)
    print(f"initial loss example: {loss:.4f}")
    # batch, time (block_size), channel (embedding_dim)
    print(f"{logits.shape=}")  # (B=4, T=8, C=65)

    # generate text starting with newline char
    idx = torch.zeros((1, 1), dtype=torch.long, device=device).fill_(encode("\n")[0])
    print(f"\n{idx=}")
    print(f"{type(idx)=}")
    res = model.generate(idx, 25)
    print(res.shape)
    text = decode(res[0].tolist())
    print(f"text='{text}'")

    @torch.no_grad()
    def eval_model(split: str):
        """return average loss across "some part" of a given dataset split (e.g. 'val')."""
        total_loss = 0.0
        samples = 100
        model.eval()
        for _ in range(samples):
            xb, yb = get_batch(
                data_map, split, batch_size=batch_size, block_size=block_size
            )
            _, loss = model(xb, yb)
            total_loss += loss.item()
        return total_loss / samples

    # higher learning rate cause this network is tiny
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    stats = {"val": [], "train": [], "step": []}

    text_before = decode(model.generate(idx, 100)[0].tolist())
    print(f"\ntext before: \n'{text_before}'")

    utils.count_params(model)
    print(f"\n\ntraining for {args.steps} steps (device={device})")
    start_time = perf_counter()
    accelerator = Accelerator(cpu=device == "cpu")
    model, optimizer, train_data = accelerator.prepare(model, optimizer, train_data)
    for step in tqdm(range(args.steps), dynamic_ncols=True):
        if step % 100 == 0:
            stats["step"].append(step)
            stats["val"].append(eval_model("val"))
            stats["train"].append(eval_model("train"))

        xb, yb = get_batch(
            data_map, "train", batch_size=batch_size, block_size=block_size
        )
        model.train()
        optimizer.zero_grad()
        _, loss = model(xb, yb)
        accelerator.backward(loss)
        optimizer.step()

    dur = perf_counter() - start_time
    print(f"training complete in {dur:.2f} seconds ({args.steps/dur:.2f} steps/sec)")

    # print(stats)
    # generate text starting with newline char
    idx = torch.ones((1, 1), dtype=torch.long, device=device).fill_(encode("\n")[0])
    text_after = decode(model.generate(idx, 500)[0].tolist())
    print(f"\ntext after: \n'{text_after}'")

    utils.plot_stats(stats, "Bigram Model Training", args.output_path)


class BigramLangModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()

        # in this case, embeddings have the same length as the vocab size
        # because for a bigram model, one char should map to a probability distribution of the other chars :)
        num_embeddings = vocab_size
        embedding_dim = vocab_size
        self.token_embedding_table = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """idx and target are both (B,T) tensor of integers."""

        # map tokens to their embeddings
        logits = self.token_embedding_table(idx)

        B, T, C = logits.shape

        if targets is None:
            return logits, None

        # restructure for cross_entropy function
        logits = logits.view((B * T, C))
        targets = targets.view(B * T)
        # reorder (B,T,C) -> (B,C,T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        """
        Given context (idx tokens), predict max_new_tokens.
        idx is (B,T) of indieces in current context.
        """

        for _ in range(max_new_tokens):
            logits, _ = self(idx)  # predict
            # this is a bigram model, so only need to look at last context token (-1)
            logits = logits[:, -1, :]  # (B,C)

            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from dist
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


def split_dataset(
    data: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split"""
    n = len(data)
    s1 = int(n * 0.8)
    s2 = int(n * 0.9)
    return (data[:s1], data[s1:s2], data[s2:])


def get_batch(
    data_map: dict,
    split: str,
    batch_size: int = batch_size,
    block_size: int = block_size,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Produce batch of inputs and target sequences.
    E.g. for a given batch item, x[item]: "hello wo",  y[item]: "ello wor",
    # so later input "h" -> target "e", input "he" -> "l", input "hel" -> "o", ...
    x and y both have dimensions: (batch_size, block_size)
    """
    data = data_map[split]

    # print(f"generating random int in [0, {len(data) - block_size})")
    # generate random start index batch_size separate batches
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[(i + 1) : i + block_size + 1] for i in ix])
    return x, y
    # return x.to(device), y.to(device)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import os
from time import perf_counter
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.nn import functional as F
from tqdm import tqdm

import utils
from bigram import get_batch, split_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# hyperparameters
batch_size = 10  # num independent sequences to process in parallel
block_size = 8  # max context length for predictions
device = "cpu"
n_embed = 32
lr = 1e-3


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-path",
        "-i",
        type=str,
        help="Path to .txt file for training/evaluation",
        default="input.txt",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        help="Path to .pdf file for plot",
        default="v2_loss.pdf",
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

    # simple tokenizer
    # string to int
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ilist: "".join([itos[n] for n in ilist])

    assert decode(encode("hello world")) == "hello world"

    # let's tokenize our dataset
    all_data = torch.tensor(encode(text), dtype=torch.long).to(device)
    print(f"full dataset: {all_data.shape=}, {all_data.dtype=}")
    # print(f"{all_data[:50]=}")
    # print(decode(all_data[:50].tolist()))

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
    model = LangModel(vocab_size, n_embed, block_size).to(device)
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
            xb, yb = get_batch(data_map, split, batch_size=8, block_size=block_size)
            _, loss = model(xb, yb)
            total_loss += loss.item()
        return total_loss / samples

    # higher learning rate cause this network is tiny
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    stats = {"val": [], "train": [], "step": []}

    text_before = decode(model.generate(idx, 100)[0].tolist())
    print(f"\ntext before: \n'{text_before}'")

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

    plt.title("V2 Model Training")
    plt.plot(stats["step"], stats["val"], label="val")
    plt.plot(stats["step"], stats["train"], label="train")
    plt.legend()
    # axis labels
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.savefig(args.output_path)
    print(f"saved plot to '{args.output_path}'")


class LangModel(nn.Module):
    def __init__(self, vocab_size: int, n_embed: int, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # position embeddings: define embedding vectors for indices [0, block_size)
        self.pos_embedding_table = nn.Embedding(block_size, n_embed)

        # self-attention head
        self.sa_head = Head(n_embed, block_size, head_size=n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """idx and target are both (B,T) tensor of integers."""
        device = next(self.parameters()).device
        B, T = idx.shape

        # map tokens to their embeddings
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.sa_head(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

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
        Crops context as necessary to ensure it doesn't exceed block_size (the max context window).
        idx is (B,T) of indieces in current context.
        """

        # TODO: continue from 1:21
        for _ in range(max_new_tokens):
            # crop idx to last block_size tokens
            idx_cur = idx[:, -self.block_size :]
            logits, _ = self(idx_cur)  # predict
            logits = logits[:, -1, :]  # (B,C)

            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from dist
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class Head(nn.Module):
    """
    Single head of self-attention.
    Specifically for a decoder block as masks are used to prevent tokens from attending to future tokens.
    See from https://youtu.be/kCc8FmEb1nY?t=4752
    """

    def __init__(
        self,
        n_embed: int,
        block_size: int,
        head_size: int,
        value_size: Optional[int] = None,
    ):
        """
        params:
            n_embed:   size of input tokens' embeddings
            head_size: size of resulting key and query vectors
            value_size: size of value representations created from each token (defaults to head_size)
        """
        super().__init__()
        value_size = head_size if value_size is None else value_size
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, value_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones((block_size, block_size))))

    def forward(self, x: torch.Tensor):
        head_size = self.key.weight.shape[1]
        # value_size = self.value.weight.shape[1]

        B, T, C = x.shape
        k = self.key(x)
        q = self.key(x)

        weights = q @ k.transpose(-2, -1)  # swaps dims -2 and -1 -> (B, head_size, T)
        # weights now specify for each sequence the attention affinities between all token combinations

        # now we use a mask to prevent tokens from attending to future tokens in their sequence
        weights = (
            weights.masked_fill(self.tril[:T, :T] == 0, float == ("-inf"))
            * head_size**-0.5
        )
        weights = F.softmax(weights, dim=-1)

        # let values combine according to query:key affinities
        out = weights @ self.value(x)  # (B,T,T) @ (B,T,value_size) = (B,T,value_size)
        return out


if __name__ == "__main__":
    main()

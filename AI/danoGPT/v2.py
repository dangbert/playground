#!/usr/bin/env python3

import argparse
import json
import os
from time import perf_counter
from typing import Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.nn import functional as F
from tqdm import tqdm

import utils
from bigram import add_common_args, get_batch, split_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# hyperparameters
batch_size = 10
block_size = 128
n_heads = 4
device = "cpu"
n_embed = 32
lr = 1e-3
n_blocks = 3


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    args = parser.parse_args()
    if os.path.isdir(args.output_path):
        args.output_path = os.path.join(args.output_path, "out")

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

    model = LangModel(
        vocab_size, n_embed, block_size, n_heads=n_heads, n_blocks=n_blocks
    ).to(device)

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

    idx = torch.zeros((1, 1), dtype=torch.long, device=device).fill_(encode("\n")[0])
    text_before = decode(model.generate(idx, 100)[0].tolist())
    print(f"\ntext before: \n'{text_before}'")

    def dump_text():
        idx = torch.ones((1, 1), dtype=torch.long, device=device).fill_(encode("\n")[0])
        # print(f"\ntext after: \n'{text_after}'")
        text_after = decode(model.generate(idx, 10_000)[0].tolist())
        text_path = args.output_path + ".output.txt"
        with open(text_path, "w") as f:
            f.write(text_after)
        print(f"wrote generated text to '{text_path}'")

        # dump model
        model_path = args.output_path + ".model"
        torch.save(model.state_dict(), model_path)

        with open(args.output_path + ".stats.json", "w") as f:
            json.dump(stats, f, indent=2)

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
            utils.plot_stats(stats, "V2 Model Training", args.output_path + "loss.pdf")
        if step % 500 == 0:
            dump_text()

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

    dump_text()

    utils.plot_stats(
        stats, "V2 Model Training", args.output_path + "loss.pdf", verbose=True
    )


class LangModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        block_size: int,
        n_heads: int = 4,
        n_blocks: int = n_blocks,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[Block(n_embed, block_size, n_heads) for _ in range(n_blocks)]
        )
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # position embeddings: define embedding vectors for indices [0, block_size)
        self.pos_embedding_table = nn.Embedding(block_size, n_embed)

        # self-attention head
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """idx and target are both (B,T) tensor of integers."""
        device = next(self.parameters()).device
        B, T = idx.shape

        # map tokens to their embeddings
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)
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


class Block(nn.Module):
    def __init__(self, n_embed: int, block_size: int, n_heads: int = 4):
        super().__init__()
        # self attention
        self.sa = MultiHeadAttention(n_heads, n_embed, block_size)
        self.norm1 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor):
        x = x + self.sa(self.norm1(x))  # skip connection
        x = x + self.ffwd(self.norm2(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads: int,
        n_embed: int,
        block_size: int,
    ):
        """Inefficient implementation of Multi-Headed attention."""
        super().__init__()
        assert n_embed % n_heads == 0, f"n_embed must be multiple of {n_heads}"
        head_size = n_embed // n_heads
        self.heads = nn.ModuleList(
            [Head(n_embed, block_size, head_size) for _ in range(n_heads)]
        )
        # TODO: understand "projection back into residual pathway" ??
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x: torch.Tensor):
        # compute and concatenate over channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class Head(nn.Module):
    """
    Single head of self-attention.
    Specifically for a decoder block as masks are used to prevent tokens from attending to future tokens.
    See from https://youtu.be/kCc8FmEb1nY?t=4752

    TODO: this can be reimplemented to add another dimension to support multi-headed attention.
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


class FeedForward(nn.Module):
    def __init__(self, n_embed: int):
        super().__init__()
        self.net = nn.Sequential(
            # using factor of 4 (see "attention is all you need" section 3.3)
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            # "projection layer going back into the residual pathway"
            nn.Linear(4 * n_embed, n_embed),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


if __name__ == "__main__":
    main()

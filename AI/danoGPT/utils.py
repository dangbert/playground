from typing import Callable, Tuple

import matplotlib.pyplot as plt
import torch


def get_encoder_decoder(chars: list[str]) -> Tuple[Callable, Callable]:
    """
    Given list of unique chars in corpus, return encoder and decoder functions.
    (simple tokenizer, maps chars to ints)
    """
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ilist: "".join([itos[n] for n in ilist])

    assert decode(encode("hello world")) == "hello world"
    return encode, decode


def plot_stats(stats: dict, title: str, fname: str, verbose: bool = False):
    plt.clf()
    plt.title(title)
    plt.plot(stats["step"], stats["val"], label="val")
    plt.plot(stats["step"], stats["train"], label="train")
    plt.legend()
    # axis labels
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.savefig(fname)
    if verbose:
        print(f"saved plot to '{fname}'")


def get_device() -> str:
    """Returns the device for PyTorch to use."""
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    # mac MPS support: https://pytorch.org/docs/stable/notes/mps.html
    elif torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not built with MPS enabled."
            )
        else:
            device = "mps"
    return device


def count_params(model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nmodel info:")
    print(f"{total_params=:_}")
    print(f"{trainable_params=:_}")
    return total_params, trainable_params

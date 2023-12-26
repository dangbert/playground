import torch


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

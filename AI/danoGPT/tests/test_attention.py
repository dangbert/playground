import torch

from v2 import Head, LangModel


def test_head__can_run():
    """Verify Head module can be used for forward pass without raising exception."""
    n_embed = 7
    head_size = 12
    block_size = 8

    head = Head(n_embed, block_size, head_size)

    batch_size = 3
    x = torch.randint(0, 10, (batch_size, block_size, n_embed), dtype=torch.float)
    out = head(x)
    assert out.shape == (batch_size, block_size, head_size)

    # should also support smaller block sizes
    x = torch.randint(0, 10, (batch_size, block_size - 2, n_embed), dtype=torch.float)
    out = head(x)
    assert out.shape == (batch_size, block_size - 2, head_size)


def test_langmodel():
    torch.manual_seed(1337)
    for _ in range(4):
        n_embed = torch.randint(5, 50, (1,)).item()
        block_size = torch.randint(8, 20, (1,)).item()
        vocab_size = torch.randint(50, 100, (1,)).item()
        model = LangModel(vocab_size, n_embed, block_size)

        start_char_idx = 4  # arbitrary here
        idx = torch.zeros((1, 1), dtype=torch.long).fill_(start_char_idx)

        max_new_tokens = torch.randint(50, 100, (1,)).item()
        res = model.generate(idx, max_new_tokens)
        assert res.shape == (1, max_new_tokens + 1)

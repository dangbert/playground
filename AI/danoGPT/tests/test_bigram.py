import torch

from bigram import BigramLangModel


def test_can_generate():
    model = BigramLangModel(vocab_size=50)

    start_char_idx = 4  # arbitrary here
    idx = torch.zeros((1, 1), dtype=torch.long).fill_(start_char_idx)
    res = model.generate(idx, 25)
    assert res.shape == (1, 25 + 1)

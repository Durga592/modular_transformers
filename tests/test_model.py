import torch
from src.models.transformer import make_model


def test_make_model():
    model = make_model(src_vocab=100, tgt_vocab=100, N=2)
    src = torch.randint(1, 100, (2, 10))
    tgt = torch.randint(1, 100, (2, 10))
    src_mask = torch.ones(2, 1, 10)
    tgt_mask = torch.ones(2, 10, 10)

    out = model(src, tgt, src_mask, tgt_mask)
    assert out.shape == (2, 10, 512)
    print("Full Transformer Model test passed successfully!")


if __name__ == "__main__":
    test_make_model()

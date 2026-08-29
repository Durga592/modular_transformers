import torch
from src.modules.attention import MultiHeadedAttention


def test_multi_headed_attention():
    h = 8
    d_model = 512
    batch_size = 2
    seq_len = 10

    mha = MultiHeadedAttention(h=h, d_model=d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    out = mha(x, x, x)

    assert out.shape == (batch_size, seq_len, d_model)
    print("MultiHeadedAttention test passed successfully!")


if __name__ == "__main__":
    test_multi_headed_attention()

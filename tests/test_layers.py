import torch
from src.modules.layers import LayerNorm, SublayerConnection


def test_layer_norm():
    ln = LayerNorm(features=512)
    x = torch.randn(2, 10, 512)
    out = ln(x)
    assert out.shape == (2, 10, 512)
    print("LayerNorm test passed successfully!")


if __name__ == "__main__":
    test_layer_norm()

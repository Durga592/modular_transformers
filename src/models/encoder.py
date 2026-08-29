import torch
import torch.nn as nn
from src.modules.layers import LayerNorm, SublayerConnection, clones


class Encoder(nn.Module):
    """Core encoder is a stack of N layers."""

    def __init__(self, layer: nn.Module, N: int):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """Encoder sub-layer combining self-attention and feed-forward networks."""

    def __init__(self, size: int, self_attn: nn.Module, feed_forward: nn.Module, dropout: float):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

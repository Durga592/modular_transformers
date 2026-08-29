import copy
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax

from src.modules.attention import MultiHeadedAttention
from src.modules.feed_forward import PositionwiseFeedForward
from src.modules.embeddings import Embeddings, PositionalEncoding
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer


class Generator(nn.Module):
    """Standard linear projection + log_softmax generation layer."""

    def __init__(self, d_model: int, vocab: int):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    """Standard Encoder-Decoder architecture."""

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: nn.Module, tgt_embed: nn.Module, generator: Generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """Take in and process masked src and target sequences."""
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def make_model(src_vocab: int, tgt_vocab: int, N: int = 6, d_model: int = 512, d_ff: int = 2048, h: int = 8, dropout: float = 0.1) -> EncoderDecoder:
    """Helper construct a full Transformer model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # Initialize parameters with Glorot / Xavier Uniform
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

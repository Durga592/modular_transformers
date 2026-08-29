import torch


def subsequent_mask(size: int) -> torch.Tensor:
    """Mask out subsequent target positions to preserve autoregressive flow."""
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


class DummyOptimizer(torch.optim.Optimizer):
    """Dummy optimizer used for execution checks and dry runs."""

    def __init__(self):
        self.param_groups = [{"lr": 0}]

    def step(self):
        pass

    def zero_grad(self, set_to_none=False):
        pass


class DummyScheduler:
    """Dummy learning rate scheduler for dry runs."""

    def step(self):
        pass

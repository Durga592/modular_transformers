import torch


def rate(step: int, model_size: int, factor: float, warmup: int) -> float:
    """
    Compute learning rate using inverse square root decay schedule with warmup.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class SimpleLossCompute:
    """A simple loss compute and train loop abstraction."""

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss

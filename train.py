import torch
from torch.optim.lr_scheduler import LambdaLR

from src.models.transformer import make_model
from src.data.dataset import data_gen
from src.training.label_smoothing import LabelSmoothing
from src.training.lr_scheduler import rate, SimpleLossCompute
from src.training.trainer import run_epoch, TrainState
from src.training.decode import greedy_decode


def run_copy_task():
    """Runs a complete end-to-end training task on synthetic copy data."""
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    print("--- Starting Synthetic Training Task ---")
    for epoch in range(10):
        model.train()
        run_epoch(
            data_gen(V, 30, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, 30, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )

    # Inference test run
    model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)
    print("\n--- Testing Greedy Decoding ---")
    print("Source Tensor: ", src)
    print("Decoded Output:", greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]

    def step(self):
        pass

    def zero_grad(self, set_to_none=False):
        pass


class DummyScheduler:
    def step(self):
        pass


if __name__ == "__main__":
    run_copy_task()

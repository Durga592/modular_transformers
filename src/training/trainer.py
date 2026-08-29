import time
import torch


class TrainState:
    """Track metrics throughout training iterations."""

    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0


def run_epoch(data_iter, model, loss_compute, optimizer, scheduler, mode="train", accum_iter=1, train_state=TrainState()):
    """Train or evaluate a single epoch."""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        if mode == "train" or mode == "train_accum":
            loss_node.backward()
            train_state.samples += batch.src.size(0)
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                train_state.step += 1
            train_state.accum_step += 1

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 40 == 1 and (mode == "train" or mode == "train_accum"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        n_accum += 1

    return total_loss / total_tokens, train_state

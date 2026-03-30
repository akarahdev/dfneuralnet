from typing import Sized

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import util

class WorldPredictorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_relu_stack(x)


def train(
        dataloader: DataLoader[Sized],
        model: WorldPredictorModel,
        loss_fn: nn.Module,
        optimizer: Optimizer
):
    model.train()
    for inp, output in dataloader:
        inp: torch.Tensor = inp.to(util.device, non_blocking=True)
        output: torch.Tensor = output.to(util.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        pred: torch.Tensor = model(inp)
        loss: torch.Tensor = loss_fn(pred, output)

        loss.backward()
        optimizer.step()


def test(
        dataloader: DataLoader[Sized],
        model: WorldPredictorModel,
        loss_fn: nn.Module,
) -> float:
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for inp, out in dataloader:
            inp = inp.to(util.device, non_blocking=True)
            out.to(util.device, non_blocking=True)

            pred = model(inp)
            test_loss += loss_fn(pred, out).item()

    return test_loss / len(dataloader)


def test_with_output(
        dataloader: DataLoader[Sized],
        model: WorldPredictorModel,
        loss_fn: nn.Module,
) -> tuple[set[tuple[float, float, float]], float]:
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0.0
    ret = set()

    with torch.no_grad():
        for inp, out in dataloader:
            inp = inp.to(util.device, non_blocking=True)
            out = out.to(util.device, non_blocking=True)

            pred = model(inp)
            test_loss += loss_fn(pred, out).item()

            mask = pred.squeeze() > 0.5
            if mask.dim() == 0:
                mask = mask.unsqueeze(0)
            passing = inp[mask]
            for row in passing.cpu().tolist():
                ret.add((row[0], row[1], row[2]))

    test_loss /= num_batches
    # print(f"Test Error: \n Avg loss: {test_loss:>8f}")
    # print(ret)
    # print()
    return ret, test_loss
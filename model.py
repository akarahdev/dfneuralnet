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
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = x.to(util.device)
        logits: torch.Tensor = self.linear_relu_stack(x)
        return logits


def train(
        dataloader: DataLoader[Sized],
        model: WorldPredictorModel,
        loss_fn: nn.Module,
        optimizer: Optimizer
):
    model.train()
    for inp, output in dataloader:
        inp: torch.Tensor = inp.to(util.device)
        output: torch.Tensor = output.to(util.device)

        pred: torch.Tensor = model(inp)
        loss: torch.Tensor = loss_fn(pred, output)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(
        dataloader: DataLoader[Sized],
        model: WorldPredictorModel,
        loss_fn: nn.Module,
) -> set[tuple[float, float, float]]:
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    ret = set()
    with torch.no_grad():
        for inp, out in dataloader:
            inp, out = inp.to(util.device), out.to(util.device)
            pred = model(inp)
            if pred.item() > 0.5:
                ret.add((inp[0][0].item(), inp[0][1].item(), inp[0][2].item()))
            test_loss += loss_fn(pred, out).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f}")
    print(ret)
    print()
    return ret
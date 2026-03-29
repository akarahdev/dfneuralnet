from torch.utils.data import DataLoader

import model
import util
import torch.nn as nn
import torch
import data
from data import WorldDataset

if __name__ == "__main__":
    print("Hello, world!")

    state = set()
    dataset = WorldDataset(state)
    dataloader = DataLoader(dataset, batch_size=1)
    network = model.WorldPredictorModel().to(util.device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters())

    for epoch in range(10000):
        print(f"Epoch {epoch}")
        model.train(dataloader, network, loss_fn, optimizer)
        model.test(dataloader, network, loss_fn)
    print("Done!")
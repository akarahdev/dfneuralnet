from flask import Flask
from flask import request
from torch.utils.data import DataLoader

import model
import data
import torch
import torch.nn as nn
import util

app = Flask(__name__)

state: set[tuple[int, int, int]] = set()
network = model.WorldPredictorModel().to(util.device)

@app.get("/train_epoch")
def train_epoch():
    global network
    global state
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(network.parameters())
    print(state)
    dataset = data.WorldDataset(state)
    dataloader = DataLoader(dataset, batch_size=1)
    for epoch in range(30):
        print(f"Epoch {epoch}")
        model.train(dataloader, network, loss_fn, optimizer)
        model.test(dataloader, network, loss_fn)
    out = model.test(dataloader, network, loss_fn)
    ret = "["
    for x, y, z in out:
        ret += f"[{x:.2f},{y:.2f},{z:.2f}], "
    ret += "]"
    print(ret)
    print(out)
    print(state)
    return ret


current_model = model.WorldPredictorModel()


@app.get("/reset_network")
def reset_model():
    global current_model
    current_model = model.WorldPredictorModel()


@app.get("/create_network")
def create_network():
    return "Network created"


@app.post("/update_dataset")
def update_dataset():
    print(f"Data: {request.get_data()}")
    json_data = request.get_json()
    print(f"Updated dataset: {json_data}")
    if not json_data or "dataset" not in json_data:
        return "Invalid JSON", 400
    global state
    state = set()
    value = json_data["dataset"]
    if not isinstance(value, list):
        return "Dataset must be a list", 400
    for element in value:
        if not isinstance(element, list) or len(element) != 3:
            continue
        state.add((element[0], element[1], element[2]))

    global network
    network = model.WorldPredictorModel().to(util.device)

    print(state)
    return "Dataset updated"


@app.get("/get_network")
def get_network():
    return ""

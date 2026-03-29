from flask import Flask
from flask import request
from torch.utils.data import DataLoader

import model
import data
import torch
import torch.nn as nn
import util
import threading

app = Flask(__name__)

state: set[tuple[int, int, int]] = set()
network = model.WorldPredictorModel().to(util.device)
network_lock = threading.Lock()
dataset = data.WorldDataset(state)
dataloader = DataLoader(dataset, batch_size=max(len(dataset), 1), pin_memory=(util.device != "cpu"), num_workers=0)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

training_thread: threading.Thread | None = None
training_active = False


def make_dataloader() -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=max(len(dataset), 1),
        pin_memory=(util.device != "cpu"),
        num_workers=0,
    )


def eternal_train():
    global training_active
    training_active = True
    print(f"Training started with state: {state}")
    epoch = 0
    while training_active:
        with network_lock:
            model.train(dataloader, network, loss_fn, optimizer)
            _, test_error = model.test(dataloader, network, loss_fn)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss {test_error:.5f}")
        epoch += 1


@app.get("/train_epoch")
def train_epoch():
    with network_lock:
        out, _ = model.test(dataloader, network, loss_fn)
    ret = "["
    for x, y, z in out:
        ret += f"[{x:.2f},{y:.2f},{z:.2f}], "
    ret += "]"
    # print(ret)
    # print(out)
    # print(state)
    return ret


@app.get("/reset_network")
def reset_model():
    global network, optimizer
    with network_lock:
        network = model.WorldPredictorModel().to(util.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    return "Network reset"


@app.get("/create_network")
def create_network():
    return "Network created"


@app.post("/update_dataset")
def update_dataset():
    global state, network, optimizer, training_active, training_thread, dataset, dataloader

    training_active = False
    if training_thread and training_thread.is_alive():
        training_thread.join()

    print(f"Data: {request.get_data()}")
    json_data = request.get_json()
    print(f"Updated dataset: {json_data}")
    if not json_data or "dataset" not in json_data:
        return "Invalid JSON", 400

    new_state = set()
    value = json_data["dataset"]
    if not isinstance(value, list):
        return "Dataset must be a list", 400
    for element in value:
        if not isinstance(element, list) or len(element) != 3:
            continue
        new_state.add((element[0], element[1], element[2]))

    state.clear()
    state.update(new_state)

    dataset = data.WorldDataset(state)
    dataloader = make_dataloader()

    network = model.WorldPredictorModel().to(util.device)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    print(state)

    training_thread = threading.Thread(target=eternal_train, daemon=True)
    training_thread.start()

    return "Dataset updated"


@app.get("/get_network")
def get_network():
    return ""
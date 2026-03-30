from torch.utils.data import DataLoader

import model
import util
import torch.nn as nn
import torch
import data
from data import StandardWorldDataset
import webserver
if __name__ == "__main__":
    webserver.app.run(host="0.0.0.0", port=8080)

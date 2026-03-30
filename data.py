import torch
from torch import Tensor
from torch.utils.data import Dataset

import util


class StandardWorldDataset(Dataset):
    def area_side_length(self) -> int:
        return 9

    def mark_dirty(self):
        self.cache.clear()

    def __init__(self, state: set[tuple[int, int, int]]):
        self.state: set[tuple[int, int, int]] = state
        self.cache: dict[int, tuple[Tensor, Tensor]] = {}

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor]:
        if item in self.cache:
            return self.cache[item]

        side_length = self.area_side_length() + 1
        x = (item // (side_length * side_length)) % side_length
        y = (item // side_length) % side_length
        z = item % side_length
        self.cache[item] = (
                Tensor([x, y, z]),
                Tensor([1.0 if (x, y, z) in self.state else 0.0])
        )
        return self.cache[item]

    def __len__(self):
        axis_size = self.area_side_length() + 1
        return axis_size ** 3

class ExtrapolatedWorldDataset(StandardWorldDataset):
    def area_side_length(self) -> int:
        return 19
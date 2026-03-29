from torch import Tensor
from torch.utils.data import Dataset

class WorldDataset(Dataset):
    state: set[tuple[int, int]]

    def __init__(self, state: set[tuple[int, int]]):
        self.state = state

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor]:
        x = (item // 10)
        y = (item % 10)
        return Tensor([float(x / 10), float(y / 10)]), Tensor([1.0 if (x, y) in self.state else 0.0])

    def __len__(self):
        return 100
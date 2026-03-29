import torch

device: str = ""
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "cpu"
else:
    device = "cpu"

def get_from_range(index: float, minimum: float, maximum: float) -> float:
    """
    Returns the value in the specified range at the specified index.
    :param index: Must be 0-1. Specifies where in the minimum and maximum to return.
    :param minimum: The minimum value.
    :param maximum: The maximum value.
    :return: The value in the range.
    """
    return ((maximum - minimum) * index) + minimum

def get_in_range(index: float, minimum: float, maximum: float) -> float:
    """
    Returns 0-1, based on where the index lies in the range.
    :param index: The number to find relative to the range.
    :param minimum: The minimum of the range.
    :param maximum: The maximum of the range.
    :return: Where the index lies relative to the range.
    """
    return (-minimum / index) + minimum - maximum
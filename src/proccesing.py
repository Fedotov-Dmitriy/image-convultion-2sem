import numpy as np


def reflect_index(index: int, size: int) -> np.ndarray:
    return np.where(
        index < 0, -index, np.where(index >= size, 2 * (size - 1) - index, index)
    )


def replicate(index: int, size: int) -> np.ndarray:
    return np.clip(index, 0, size - 1)


def wrap(index: int, size: int) -> int:
    return index % size


modes = {"1": reflect_index, "2": replicate, "3": wrap}

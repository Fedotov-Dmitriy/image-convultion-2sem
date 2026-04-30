import numpy as np
from typing import Callable


def convolution(image: np.ndarray, kernel: np.ndarray, pad: Callable) -> np.ndarray:
    height = image.shape[0]
    width = image.shape[1]
    col_index = np.arange(width)
    row_index = np.arange(height)
    row_index, col_index = np.meshgrid(row_index, col_index, indexing="ij")

    kernel_row_index = np.arange(kernel.shape[0]) - kernel.shape[0] // 2
    kernel_col_index = np.arange(kernel.shape[1]) - kernel.shape[1] // 2
    kernel_row_index, kernel_col_index = np.meshgrid(
        kernel_row_index, kernel_col_index, indexing="ij"
    )

    row_index = pad(row_index[..., np.newaxis, np.newaxis] + kernel_row_index, height)
    col_index = pad(col_index[..., np.newaxis, np.newaxis] + kernel_col_index, width)

    matrix = np.sum(
        kernel[np.newaxis, np.newaxis, ..., np.newaxis]
        * image[row_index, col_index, :],
        axis=(2, 3),
    )
    return np.clip(matrix, 0, 255).astype(np.uint8)

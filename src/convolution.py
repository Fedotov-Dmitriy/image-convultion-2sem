def convolution(image: np.ndarray, kernel: np.ndarray, pad: Callable) -> np.ndarray:
    if image.size == 0 or kernel.size == 0:
        raise ValueError("Изображение должно быть пустыми")
    if kernel.size == 0:
        raise ValueError("Ядро свертки не должно быть пустым")
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("Размер ядра должен быть нечетным")

    height = image.shape[0]
    width = image.shape[1]
    is_grayscale = image.ndim == 2
    if is_grayscale:
        image = image[..., np.newaxis]

    col_range = np.arange(width)
    row_range = np.arange(height)
    row_index, col_index = np.meshgrid(row_range, col_range, indexing="ij")

    kernel_row_range = np.arange(kernel.shape[0]) - kernel.shape[0] // 2
    kernel_col_range = np.arange(kernel.shape[1]) - kernel.shape[1] // 2
    kernel_row_index, kernel_col_index = np.meshgrid(
        kernel_row_range, kernel_col_range, indexing="ij"
    )

    row_index = pad(row_index[..., np.newaxis, np.newaxis] + kernel_row_index, height)
    col_index = pad(col_index[..., np.newaxis, np.newaxis] + kernel_col_index, width)

    matrix = np.sum(
        kernel[np.newaxis, np.newaxis, ..., np.newaxis].astype(np.float32)
        * image[row_index, col_index, :].astype(np.float32),
        axis=(2, 3),
        dtype=np.float32,
    )

    result = np.clip(matrix, 0, 255).astype(np.uint8)

    if is_grayscale:
        result = result[..., 0]

    return result
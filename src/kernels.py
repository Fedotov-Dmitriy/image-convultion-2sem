from numpy import array

kernels = {
    # Размытие
    "box_blur": array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
    "gaussian_blur": array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
    # Обнаружение границ
    "sobel_x": array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    "prewitt_y": array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
    "laplacian": array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
    # Резкость
    "sharpen": array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
}

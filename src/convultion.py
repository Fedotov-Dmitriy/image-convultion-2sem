import numpy as np
from proccesing import reflect_index


def convultion(image: np.ndarray, kernles: np.ndarray) -> np.ndarray:
    height, width = image.shape
    kernel_half = kernles.shape[0] // 2
    picture = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            pixel = 0.0
            for moving_x in range(-kernel_half, kernel_half + 1):
                for moving_y in range(-kernel_half, kernel_half + 1):
                    px = reflect_index(i + moving_x, image.shape[0])
                    py = reflect_index(j + moving_y, image.shape[1])
                    pixel += (
                        image[px][py]
                        * kernles[kernel_half + moving_x][kernel_half + moving_y]
                    )
            picture[i, j] = pixel
    return np.clip(picture, 0, 255).astype(np.uint8)

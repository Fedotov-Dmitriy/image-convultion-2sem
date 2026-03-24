import numpy as np
from proccesing import reflect_index


def convultion(image: np.ndarray, kernles: np.ndarray) -> np.ndarray:
    height, width = image.shape
    kernel_half = kernles.shape[0] // 2
    picture = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            pixel = 0
            for sosed_x in range(-kernel_half, kernel_half + 1):
                for sosed_y in range(-kernel_half, kernel_half + 1):
                    px = reflect_index(i + sosed_x, image.shape[0])
                    py = reflect_index(j + sosed_y, image.shape[1])
                    pixel += (
                        image[px][py]
                        * kernles[kernel_half + sosed_x][kernel_half + sosed_y]
                    )
            picture[i, j] = pixel
    return np.clip(picture, 0, 255).astype(np.uint8)

from PIL import Image
import pytest
from convolution import convolution
from kernels import kernels
import numpy as np
from proccesing import reflect_index
import os

BASE_DIR = os.path.dirname(__file__)
img_path = os.path.join(BASE_DIR, "..", "input", "night_city.jpg")


@pytest.mark.parametrize("kernel_name", kernels.keys())
def test_convolution_matches_reference(kernel_name):
    reference_img_path = os.path.join(BASE_DIR, "reference", f"{kernel_name}.png")
    img = Image.open(img_path)
    ref_img = Image.open(reference_img_path)
    result_1 = convolution(np.array(img), kernels[kernel_name], reflect_index)
    result_2 = np.array(ref_img)
    kernel_size = kernels[kernel_name].shape[0]
    pad = kernel_size // 2
    np.testing.assert_allclose(
        result_1[pad:-pad, pad:-pad].astype(float),
        result_2[pad:-pad, pad:-pad].astype(float),
        atol=1,
    )

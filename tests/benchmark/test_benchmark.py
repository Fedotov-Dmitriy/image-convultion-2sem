import numpy as np
import pytest
from PIL import Image, ImageFilter
from src.convolution import convolution
from src.kernels import kernels
from src.proccesing import reflect_index, replicate

cv2 = pytest.importorskip("cv2")

# размеры, типы, ядра
IMAGE_SIZES = [
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
]

IMAGE_TYPES = ["grayscale", "rgb"]

KERNEL_NAMES = ["box_blur", "gaussian_blur", "sobel_x", "sharpen"]

EDGE_MODES = {
    "reflect": {
        "my_pad": reflect_index,
        "numpy_pad": "symmetric",
        "opencv_border": cv2.BORDER_REFLECT,
    },
    "replicate": {
        "my_pad": replicate,
        "numpy_pad": "edge",
        "opencv_border": cv2.BORDER_REPLICATE,
    },
}


# fixtures
@pytest.fixture(params=IMAGE_SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def image_size(request):
    return request.param


@pytest.fixture(params=IMAGE_TYPES)
def image_type(request):
    return request.param


@pytest.fixture(params=KERNEL_NAMES)
def kernel_name(request):
    return request.param


@pytest.fixture(params=list(EDGE_MODES.keys()))
def edge_mode_name(request):
    return request.param


@pytest.fixture
def image(image_size, image_type):
    width, height = image_size
    rng = np.random.default_rng(42)
    if image_type == "grayscale":
        return rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


@pytest.fixture
def kernel(kernel_name):
    return kernels[kernel_name]


@pytest.fixture
def edge_mode(edge_mode_name):
    return EDGE_MODES[edge_mode_name]


# OpenCV filter
def opencv_filter(image, kernel, edge_mode_name):
    mode = EDGE_MODES[edge_mode_name]
    border = mode["opencv_border"]
    if border is None:
        pytest.skip(f"OpenCV не поддерживает режим края {edge_mode_name}")
    return cv2.filter2D(image, -1, kernel.astype(np.float32), borderType=border)


# Pillow filter
def pillow_filter_channel(channel, kernel, numpy_pad_mode):
    kh, kw = kernel.shape
    pad_y, pad_x = kh // 2, kw // 2
    padded = np.pad(channel, ((pad_y, pad_y), (pad_x, pad_x)), mode=numpy_pad_mode)
    pil_image = Image.fromarray(padded)
    filtered = pil_image.filter(
        ImageFilter.Kernel(
            size=(kw, kh),
            kernel=kernel.astype(float).flatten().tolist(),
            scale=1,
            offset=0,
        )
    )
    result = np.asarray(filtered)
    return result[pad_y:-pad_y, pad_x:-pad_x].astype(np.uint8)


def pillow_filter(image, kernel, edge_mode_name):
    mode = EDGE_MODES[edge_mode_name]
    numpy_pad_mode = mode["numpy_pad"]
    if numpy_pad_mode not in ["symmetric", "edge"]:
        pytest.skip(f"Pillow не поддерживает режим края {edge_mode_name}")
    if image.ndim == 2:
        return pillow_filter_channel(image, kernel, numpy_pad_mode)
    channels = [
        pillow_filter_channel(image[:, :, c], kernel, numpy_pad_mode)
        for c in range(image.shape[2])
    ]
    return np.stack(channels, axis=2)


# Benchmarks
def test_my_convolution_benchmark(benchmark, image, kernel, edge_mode):
    result = benchmark(convolution, image, kernel, edge_mode["my_pad"])
    assert result.shape == image.shape


def test_opencv_benchmark(benchmark, image, kernel, edge_mode_name):
    result = benchmark(opencv_filter, image, kernel, edge_mode_name)
    assert result.shape == image.shape


def test_pillow_benchmark(benchmark, image, kernel, edge_mode_name):
    result = benchmark(pillow_filter, image, kernel, edge_mode_name)
    assert result.shape == image.shape

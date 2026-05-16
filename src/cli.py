import os
from typing import Callable
from kernels import kernels
from proccesing import modes
from PIL import Image


def kernels_output() -> str:
    kernel_names = list(kernels.keys())

    while True:
        print("Выберите номер ядра свертки")
        for i, name in enumerate(kernel_names):
            print(f"{i + 1}. {name}")

        try:
            number = int(input()) - 1
        except ValueError:
            print("Ошибка: нужно ввести число")
            continue

        if 0 <= number < len(kernel_names):
            return kernel_names[number]

        print("Ошибка: такого номера ядра нет")


def picture_output() -> str:
    folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "input")

    if not os.path.exists(folder_path):
        raise FileNotFoundError("Папка input не найдена")

    pictures = os.listdir(folder_path)

    if not pictures:
        raise FileNotFoundError("В папке input нет изображений")

    while True:
        print("Выберите картинку")
        for i, name in enumerate(pictures):
            print(f"{i + 1}. {name}")

        try:
            number = int(input()) - 1
        except ValueError:
            print("Ошибка: нужно ввести число")
            continue

        if 0 <= number < len(pictures):
            return os.path.join(folder_path, pictures[number])

        print("Ошибка: такого номера картинки нет")


def select() -> Callable:
    while True:
        choice = input(
            "Выберите режим обработки края:\n1. reflect\n2. replicate\n3. wrap\n"
        )

        if choice in modes:
            return modes[choice]

        print("Ошибка: нужно выбрать 1, 2 или 3")


def grayscale(img: Image.Image) -> Image.Image:
    while True:
        choice = input("Выберите цвет изображения:\n1. grayscale\n2. RGB\n")

        if choice == "1":
            return img.convert("L")

        if choice == "2":
            return img.convert("RGB")

        print("Ошибка: нужно выбрать 1 или 2")

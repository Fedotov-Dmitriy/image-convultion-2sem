import os

from kernels import kernels


def kernels_output() -> str:
    print("Выберите номер ядра свертки")
    for i, name in enumerate(kernels.keys()):
        print(f"{i + 1}.{name}")
    number = int(input()) - 1
    return list(kernels.keys())[number]


def picture_output() -> str:
    folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "input")
    pictures = os.listdir(folder_path)
    print("Выберите картинку")
    for i, name in enumerate(pictures):
        print(f"{i + 1}.{name}")
    number = int(input()) - 1
    return os.path.join(folder_path, pictures[number])

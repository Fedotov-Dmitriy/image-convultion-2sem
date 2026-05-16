from kernels import kernels
from convolution import convolution
from PIL import Image, UnidentifiedImageError
from numpy import array
from cli import kernels_output, picture_output, select, grayscale


def main() -> None:
    try:
        img = Image.open(picture_output())
        img = grayscale(img)

        result = convolution(array(img), kernels[kernels_output()], select())

        new_img = Image.fromarray(result)
        new_img.show()

    except FileNotFoundError as error:
        print(f"Ошибка: {error}")

    except UnidentifiedImageError:
        print("Ошибка: выбранный файл не является изображением")

    except ValueError as error:
        print(f"Ошибка: {error}")

    except KeyboardInterrupt:
        print("\nПрограмма остановлена пользователем")


if __name__ == "__main__":
    main()

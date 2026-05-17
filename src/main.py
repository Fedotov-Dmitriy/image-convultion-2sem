from src.kernels import kernels
from src.convolution import convolution
from PIL import Image, UnidentifiedImageError
from numpy import array
from src.cli import kernels_output, picture_output, select, grayscale


def main() -> None:
    try:
        opened_img = Image.open(picture_output())
        processed_img = grayscale(opened_img)

        result = convolution(array(processed_img), kernels[kernels_output()], select())

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

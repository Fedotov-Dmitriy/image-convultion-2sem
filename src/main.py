from kernels import kernels
from convolution import convolution
from PIL import Image
from numpy import array
from cli import kernels_output, picture_output, select

if __name__ == "__main__":
    img = Image.open(picture_output())
    result = convolution(array(img), kernels[kernels_output()], select())
    new_img = Image.fromarray(result)
    new_img.show()

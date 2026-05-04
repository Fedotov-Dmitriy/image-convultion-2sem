from kernels import kernels
from convultion import convultion
from PIL import Image
from numpy import array
from cli import kernels_output, picture_output

if __name__ == "__main__":
    img = Image.open(picture_output()).convert("L")
    result = convultion(array(img), kernels[kernels_output()])
    new_img = Image.fromarray(result)
    new_img.show()

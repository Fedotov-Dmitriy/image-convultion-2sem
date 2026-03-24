from kernels import kernels
from convultion import convultion
from PIL import Image
from numpy import array

if __name__ == "__main__":
    img = Image.open("input/b9f91113198e9e6c0a0a0415af19d2b8.jpg").convert("L")
    result = convultion(array(img), kernels["sharpen"])
    new_img = Image.fromarray(result)
    new_img.save("output/rabbit_sharpen.jpg")

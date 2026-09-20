import numpy as np
import cv2

def hs_colorfulness(image: cv2.typing.MatLike) -> float:
    #takes in a standard cv2 BGR image in the format of a numpy array and calculates
    #the colorfulness metric as defined in https://doi.org/10.1117/12.477378
    b, g, r = cv2.split(image)

    rg = r - g
    yb = 0.5 * (r + g) - b

    std_rg = rg.std()
    std_yb = yb.std()
    mean_rg = rg.mean()
    mean_yb=yb.mean()

    std_rgyb = np.sqrt(std_rg ** 2 + std_yb ** 2)
    mean_rgyb = np.sqrt(mean_rg ** 2 + mean_yb ** 2)

    colorfulness = std_rgyb + 0.3 * mean_rgyb
    return colorfulness

def hs_colorfulness_smoothed(image: cv2.typing.MatLike) -> float:
    #same as hs_colorfulness but it applies a minor 3x3 gaussian blur to reduce effects
    #of noise, compression, or dithering
    image = cv2.GaussianBlur(image, (3,3),0)
    return hs_colorfulness(image)
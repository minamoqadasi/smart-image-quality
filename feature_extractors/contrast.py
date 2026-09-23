import numpy as np
import cv2

def rms_contrast(image: cv2.typing.MatLike) -> float:

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    contrast = image_gray.std()
    return contrast

def michelson_contrast(image: cv2.typing.MatLike) -> float:
    # this is based on the brightest and darkest pixels
    # formula: (I_max - I_min) / (I_max + I_min)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image_float = image_gray.astype(float)
    
    i_min = image_float.min()
    i_max = image_float.max()
    
    denominator = i_max + i_min
    if denominator == 0:
        return 0.0 # prevents division by zero on completely black images
        
    contrast = (i_max - i_min) / denominator
    return contrast

def rms_contrast_smoothed(image: cv2.typing.MatLike) -> float:
    # 3x3 gaussian blur beforehand
    image_smoothed = cv2.GaussianBlur(image, (3, 3), 0)
    return rms_contrast(image_smoothed)
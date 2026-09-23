import numpy as np
import cv2

def grayscale_brightness(image: cv2.typing.MatLike) -> float:

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    mean_brightness = image_gray.mean()
    return mean_brightness

def hsv_brightness(image: cv2.typing.MatLike) -> float:

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    h, s, v = cv2.split(hsv_image)
    
    mean_brightness = v.mean()
    return mean_brightness

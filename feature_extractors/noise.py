import numpy as np
import cv2

def guassian_noise(image: cv2.typing.MatLike) -> float:

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    noise = gray.astype(float) - blur.astype(float)
    return float(np.std(noise))

def median_noise(image: cv2.typing.MatLike) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 3)

    noise = np.abs(gray.astype(float) - median.astype(float))
    return float(np.mean(noise))

def snr(image: cv2.typing.MatLike) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise_level = guassian_noise(image)

    if noise_level == 0: 
        return 0.0

    return float(np.mean(gray) / noise_level)
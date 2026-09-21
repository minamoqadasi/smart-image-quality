import cv2
import numpy as np
from skimage.measure import shannon_entropy

def laplacian_sharpness(image: cv2.typing.MatLike) -> float:
    #calculates the variance of the Laplacian as a proxy for sharpness
    #It's a kind of crude approximation of perceptual sharpness, but it's in wide use
    #so I figured it would be important to include it

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale because color isn't usually a component of sharpness perception
    image_laplacian = cv2.Laplacian(image_gray, ddepth=cv2.CV_16S, ksize=3) #ddepth CV_16S is used so that it's stored as a 16-bit signed integer

    laplacian_variance = image_laplacian.var()
    return laplacian_variance

def laplacian_sharpness_bilateral(image: cv2.typing.MatLike) -> float:
    #similar to laplacian_sharpness but it applies a minor bilateral filter to reduce effects of noise while preserving edges
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_smoothed = cv2.bilateralFilter(image_gray, d=5, sigmaColor=50, sigmaSpace=50) #setting d=5 and both sigmas to 50 as a measure to ensure that oversmoothing doesn't occur
    image_laplacian = cv2.Laplacian(image_smoothed, ddepth=cv2.CV_16S, ksize=3)

    laplacian_variance = image_laplacian.var()
    return laplacian_variance

def tenengrad_sharpness(image: cv2.typing.MatLike) -> float:
    #the tenengrad sharpness metric is the magnitude of the image gradient as calculated by the sobel operator
    #it's a widely used sharpness metric for tasks like autofocus detection

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #the following two calls take the gradient in only one direction each
    gradient_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.hypot(gradient_x, gradient_y) #just now learned there's a native numpy function for calculating sqrt(x^2 + y^2)
    return np.mean(gradient_magnitude)

def tenengrad_variance_sharpness(image:cv2.typing.MatLike) -> float:
    #a minor variation on the tenengrad sharpness calculation that incorporates the variance of the image in order to reduce the effect of noise
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return tenengrad_sharpness(image) + np.var(image_gray)

def entropic_sharpness(image: cv2.typing.MatLike) -> float:
    #another interesting measure of the sharpness of an image is calculating the entropy of its intensity value distribution
    #the idea is that sharper images have more information in them and therefore more entropy
    #and this metric should be better at detecting sharpness in terms of textural attributes than edges
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return shannon_entropy(image_gray)





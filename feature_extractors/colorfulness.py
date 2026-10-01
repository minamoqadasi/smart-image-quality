import math

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

def lab_colorfulness(image: cv2.typing.MatLike) -> float:
    #takes in cv2 BGR image and converts it into CIELAB to calculate perceptual chroma
    #because CIELAB is perceptul, this chroma should closely correlate to human colorfulness perception
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image.astype(int))
    chroma = np.sqrt(a**2 + b**2)
    mean_chroma = chroma.mean()
    std_chroma = chroma.std()
    if(math.isinf(std_chroma)):
        #for some reason some of the images are returning infinite colorfulness so. stop that
        raise ValueError

    colorfulness = mean_chroma + 0.3 * std_chroma #0.3 seems to be standard weighting among papers
    return colorfulness

def hsv_colorfulness(image: cv2.typing.MatLike, k:float = 0.3) -> float:
    #HSV naturally breaks an image apart into hue, saturation, and value (brightness)
    # so it makes a natural choice for calculating colorfulness which is normally tied to saturation in human vision

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h,s,v = cv2.split(hsv_image)

    saturation_mean = s.mean()
    saturation_std = s.std()
    value_mean = v.mean()

    colorfulness = saturation_std + 0.3 * saturation_mean - k * np.abs(value_mean-128) #the last bit at the end is to make darker images less "colorful"

    return colorfulness

def entropic_colorfulness(image: cv2.typing.MatLike) -> float:
    #this is the most unique colorfulness metric, as it relies on creating color histograms
    #and calculating the entropy of them
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) #convert to LAB so we're in a perceptual color space
    histogram = cv2.calcHist([lab_image], [0,1,2],None, [32,32,32], [0,256,0,256,0,256] )

    total_pixels = np.sum(histogram)
    if total_pixels==0:
        return 0

    probabilities = histogram / total_pixels #this basically transforms it into a probability distribution
    non_zero_probs = probabilities[probabilities > 0] #filter out all the 0 probabilities to avoid log of 0

    entropy = -1 * np.sum(non_zero_probs * np.log2(non_zero_probs))
    normalized_entropy = entropy /  15.0 #dividing by the log_2 of the total number of bins to get the normalized entropy, and log_2 of 32**3 is 15
    return normalized_entropy

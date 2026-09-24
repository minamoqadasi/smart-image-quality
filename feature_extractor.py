from pathlib import Path
import pandas as pd
import cv2

from feature_extractors import *


def process_dataset(image_dir: str, output_csv: str = "dataset_features.csv"):
    image_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))

    data = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue

        features = {
            "filename": path.name,
            "brightness": brightness.grayscale_brightness(image),
            "brightness_hsv": brightness.hsv_brightness(image),
            "colorfulness_hs": colorfulness.hs_colorfulness(image),
            "colorfulness_hs_smoothed": colorfulness.hs_colorfulness_smoothed(image),
            "colorfulness_lab": colorfulness.lab_colorfulness(image),
            "colorfulness_hsv": colorfulness.hsv_colorfulness(image),
            "colorfulness_entropy": colorfulness.entropic_colorfulness(image),
            "contrast_rms": contrast.rms_contrast(image),
            "contrast_michelson": contrast.michelson_contrast(image),
            "contrast_rms_smoothed": contrast.rms_contrast_smoothed(image),
            "sharpness_laplacian": sharpness.laplacian_sharpness(image),
            "sharpness_laplacian_bilateral": sharpness.laplacian_sharpness_bilateral(image),
            "sharpness_tenengrad": sharpness.tenengrad_sharpness(image),
            "sharpness_tenengrad_variance": sharpness.tenengrad_variance_sharpness(image),
            "sharpness_entropy": sharpness.entropic_sharpness(image),
            "sharpness_hfer": sharpness.hfer_sharpness(image),
            "gaussian_noise": noise.guassian_noise(image),
            "median_noise": noise.median_noise(image),
            "gaussian_snr": noise.snr(image)
        }
        data.append(features)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    return df

def extract_image_features(image: cv2.typing.MatLike) -> dict[str, float]:
    features = {
        "brightness": brightness.grayscale_brightness(image),
        "brightness_hsv": brightness.hsv_brightness(image),
        "colorfulness_hs": colorfulness.hs_colorfulness(image),
        "colorfulness_hs_smoothed": colorfulness.hs_colorfulness_smoothed(image),
        "colorfulness_lab": colorfulness.lab_colorfulness(image),
        "colorfulness_hsv": colorfulness.hsv_colorfulness(image),
        "colorfulness_entropy": colorfulness.entropic_colorfulness(image),
        "contrast_rms": contrast.rms_contrast(image),
        "contrast_michelson": contrast.michelson_contrast(image),
        "contrast_rms_smoothed": contrast.rms_contrast_smoothed(image),
        "sharpness_laplacian": sharpness.laplacian_sharpness(image),
        "sharpness_laplacian_bilateral": sharpness.laplacian_sharpness_bilateral(image),
        "sharpness_tenengrad": sharpness.tenengrad_sharpness(image),
        "sharpness_tenengrad_variance": sharpness.tenengrad_variance_sharpness(image),
        "sharpness_entropy": sharpness.entropic_sharpness(image),
        "sharpness_hfer": sharpness.hfer_sharpness(image),
        "gaussian_noise": noise.guassian_noise(image),
        "median_noise": noise.median_noise(image),
        "gaussian_snr": noise.snr(image)
    }
    return features
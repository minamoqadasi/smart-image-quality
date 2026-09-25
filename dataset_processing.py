import pathlib
from pathlib import Path
from feature_extractors import extract_image_features

def process_biq(dataset_root: pathlib.Path = Path("datasets/")):
    koniq_root = dataset_root / "koniq"
    koniq_csv = koniq_root / "koniq10k_scores_and_distributions.csv"
    koniq_images = koniq_root / "images"

def process_koniq(dataset_root: pathlib.Path = Path("datasets/")):
    biq_root = dataset_root / "biq2021"
    biq_csv = biq_root / "BIQ2021.csv"
    biq_images = biq_root / "images"

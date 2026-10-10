import os
import pathlib
from pathlib import Path
import cv2
from tqdm import tqdm

from feature_extractors import extract_image_features
import pandas as pd
import numpy as np

def process_koniq(dataset_root: pathlib.Path = Path("datasets/")) -> pd.DataFrame:
    koniq_root: pathlib.Path = dataset_root / "koniq"
    koniq_score_csv: pathlib.Path = koniq_root / "koniq10k_scores_and_distributions.csv"
    koniq_indicator_csv: pathlib.Path = koniq_root / "koniq10k_indicators.csv"
    koniq_images_path: pathlib.Path = koniq_root / "images"

    score_df = pd.read_csv(koniq_score_csv) #load the MOS score CSV to get the file paths and the MOS score

    indicator_df = pd.read_csv(koniq_indicator_csv)

    score_df['MOS'] = (score_df['MOS'] - 1) / (5 - 1) #min-max norm but based on the absolute range of 1-5 instead of the empirical range
    score_df['SD'] = score_df['SD'] / (5 - 1) # laws of standard deviation tells us that if Y = aX+b then std(Y) = |a|std(X)
    score_df = score_df.drop(columns=["c1", "c2", 'c3', 'c4', 'c5', "c_total", 'MOS_zscore']) # get rid of the unnecessary columns preemptively

    indicator_df['image_id'] = indicator_df['image_id'].astype(str) + '.jpg' #i kinda feel bad doing it like this, but it's the only way i can think of to convert the ideas to the file paths
    indicator_df = indicator_df.drop(columns=['quality_factor', 'bitrate','hxw','deep_feature'])
    indicator_df = indicator_df.rename(columns={"brightness":"brightness_subjective","contrast":"contrast_subjective","colorfulness":"colorfulness_subjective","sharpness":"sharpness_subjective"})

    joint_df = score_df.set_index('image_name').join(indicator_df.set_index('image_id'))
    joint_df['image_name'] = joint_df.index

    image_features_list = []
    for image_name in tqdm(joint_df['image_name']):
        image = cv2.imread(koniq_images_path/image_name)
        if image is None:
            #this is just a graceful way of handling if an image is, for some reason, not actually there
            #just skip it entirely
            continue
        image_features = extract_image_features(image)
        image_features['image_name'] = image_name
        image_features_list.append(image_features)

    image_feature_df = pd.DataFrame(image_features_list)
    # print(image_feature_df)

    joint_df = joint_df.set_index('image_name').join(image_feature_df.set_index('image_name'))

    return joint_df

def process_biq(dataset_root: pathlib.Path = Path("datasets/")) -> pd.DataFrame:
    biq_root = dataset_root / "biq2021"
    biq_csv = biq_root / "BIQ2021.csv"
    biq_image_path = biq_root / "images"

    score_df = pd.read_csv(biq_csv)
    score_df=score_df.rename(columns={"StandardDeviation":"SD","Images":"image_name"})
    #biq doesn't have any unnecessary columns so no need to drop anything, and it's all in [0,1] already

    image_features_list = []
    for image_name in tqdm(score_df['image_name']):
        image = cv2.imread(biq_image_path/image_name)
        if image is None:
            continue
        image_features = extract_image_features(image)
        image_features['image_name'] = image_name
        image_features_list.append(image_features)
    image_feature_df = pd.DataFrame(image_features_list)

    joint_df = score_df.set_index('image_name').join(image_feature_df.set_index('image_name'))

    return joint_df

def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    excluded_columns = ['brightness_subjective','contrast_subjective','colorfulness_subjective','sharpness_subjective','MOS','SD']
    numerical_columns = df.select_dtypes(include="number") #only makes sense to normalize on numeric columns
    for column in numerical_columns:
        if column not in excluded_columns: #no need to normalize the subjective columns, they're not even in both datasets
            #basic min-max normalization to get to [0,1]
            #NOT normalizing MOS and SD because those are already in [0,1] (and normalizing SD doesn't make sense anyways)
            column_range = df[column].max() - df[column].min()
            df[column] = (df[column] - df[column].min()) / column_range

    return df

def clean_merged_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # remove duplicate columns
    df = df.loc[:,~df.columns.duplicated()].copy()

    # remove redundant columns that are created when reading CSVs with pandas
    redundant_index_cols = [col for col in df.columns if col.startswith('Unnamed:')]
    df = df.drop(columns=redundant_index_cols, errors='ignore')

    # remove cols with no info
    df = df.dropna(axis=1, how='all')

    # remove constant columns except for identifiers
    identifiers = {"image_name", "dataset"}
    constant_cols = [col for col in df.columns if col not in identifiers and df[col].nunique(dropna=False) <= 1]
    df = df.drop(columns=constant_cols)

    return df


if __name__ == "__main__":

    output_folder = Path('output/')
    output_folder.mkdir(parents=True, exist_ok=True)

    koniq_df = process_koniq()
    biq_df = process_biq()

    koniq_df.to_csv(output_folder/'processed_koniq.csv')
    biq_df.to_csv(output_folder/'processed_biq.csv')

    koniq_df['dataset'] = 'koniq'
    biq_df['dataset'] = 'biq2021'

    joint_df = normalize_dataset(pd.concat([koniq_df, biq_df], sort=False))
    joint_df.to_csv(output_folder / 'processed_merged.csv')

    joint_df = clean_merged_dataset(joint_df)


    joint_df.to_csv(output_folder/'cleaned_merged.csv')

    print(f"\nMerged dataset shape: {joint_df.shape}")
    print("\nColumns: ")
    print(joint_df.columns.tolist())

    print("\nMissing values per column: ")
    print(joint_df.isna().sum().sort_values(ascending=False))

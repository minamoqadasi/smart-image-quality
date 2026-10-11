import pandas as pd
import numpy as np
from pathlib import Path

input_path = Path("output/processed_merged.csv")
df = pd.read_csv(input_path)

# missing values
print("\n--- Missing Values ---")

missing_count = df.isna().sum()
missing_percent = df.isna().mean() * 100

missing_summary = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing Percentage": missing_percent.round(2)
    })

print(missing_summary[
        missing_summary["Missing Count"] > 0
    ].sort_values("Missing Count", ascending=False))

# unshared columns
subjective_cols = [
        "brightness_subjective",
        "contrast_subjective",
        "colorfulness_subjective",
        "sharpness_subjective"
    ]

# Remove unshared columns and MOS/SD from the feature set
excluded = set(subjective_cols + ["MOS", "SD"])

feature_cols = [
        col for col in df.select_dtypes(include="number").columns
        if col not in excluded
    ]

# Remove rows without a valid target or extracted features
before = len(df)

df = df.dropna(subset=["MOS"] + feature_cols)

print("\nRows before cleaning:", before)
print("Rows after cleaning:", len(df))
print("Rows removed:", before - len(df))



# duplicate
print("\n--- Duplicate Data ---")

before = len(df)

# completely identical rows
exact_duplicates = df.duplicated().sum()
print("Exact duplicate rows:", exact_duplicates)

# duplicate image
duplicate_images = df.duplicated(subset=["dataset", "image_name"]).sum()

print("Duplicate image identifiers:", duplicate_images)

# Display duplicated image records
if duplicate_images > 0:
    duplicates = df[df.duplicated(subset=["dataset", "image_name"], keep=False)]

    print("\nDuplicate image records:")
    print(duplicates[["dataset", "image_name", "MOS"]])

# Remove duplicates
df = df.drop_duplicates(subset=["dataset", "image_name"], keep="first").copy()

print("\nRows before:", before)
print("Rows after:", len(df))
print("Rows removed:", before - len(df))



# outliers
print("\n--- Outliers ---")

outlier_summary = []

for col in feature_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    outlier_summary.append({"Feature": col, "Outlier Count": len(outliers), "Outlier Percentage": round(len(outliers) / len(df) * 100, 2)})

outlier_df = pd.DataFrame(outlier_summary)

outlier_df = outlier_df.sort_values("Outlier Count", ascending=False)

print(outlier_df.to_string(index=False))

# outlier report
outlier_df.to_csv("output/outlier_summary.csv", index=False)

print("\nOutlier report saved.")




# correlation
print("\n--- Correlation ---")

# extracted numerical features and MOS
correlation_cols = feature_cols + ["MOS"]

correlation_data = df[correlation_cols]

# Pearson correlation
pearson_corr = correlation_data.corr(method="pearson")

# Spearman correlation
spearman_corr = correlation_data.corr(method="spearman")

# Compare correlations with MOS
mos_correlation = pd.DataFrame({"Pearson": pearson_corr["MOS"].drop("MOS"), 
                                "Spearman": spearman_corr["MOS"].drop("MOS")})

mos_correlation["Absolute Pearson"] = (mos_correlation["Pearson"].abs())

mos_correlation = mos_correlation.sort_values("Absolute Pearson", ascending=False)

print("\nFeature Correlation with MOS:")
print(mos_correlation.to_string())

# Save results
pearson_corr.to_csv("output/pearson_correlation.csv")

spearman_corr.to_csv("output/spearman_correlation.csv")

mos_correlation.to_csv("output/mos_feature_correlations.csv")

print("\nCorrelation reports saved.")


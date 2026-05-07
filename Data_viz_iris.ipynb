# ==========================================
# DATA VISUALIZATION III
# IRIS DATASET ANALYSIS
# ==========================================

# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# LOAD IRIS DATASET
# ==========================================

df = sns.load_dataset('iris')

# ==========================================
# DISPLAY FIRST 5 ROWS
# ==========================================

print("\nFirst 5 Rows:\n")

print(df.head())

# ==========================================
# DATASET INFORMATION
# ==========================================

print("\nDataset Information:\n")

print(df.info())

# ==========================================
# FEATURE TYPES
# ==========================================

print("\nFeature Types:\n")

print(df.dtypes)

# ==========================================
# NUMERIC AND NOMINAL FEATURES
# ==========================================

numeric_features = [
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width'
]

nominal_features = [
    'species'
]

print("\nNumeric Features:\n")

print(numeric_features)

print("\nNominal Features:\n")

print(nominal_features)

# ==========================================
# DESCRIPTIVE STATISTICS
# ==========================================

print("\nDescriptive Statistics:\n")

print(df.describe())

# ==========================================
# HISTOGRAM FOR EACH FEATURE
# ==========================================

plt.figure(figsize=(12,8))

for i, col in enumerate(numeric_features):

    plt.subplot(2,2,i+1)

    sns.histplot(
        data=df,
        x=col,
        kde=True,
        bins=15
    )

    plt.title(f'Histogram of {col}')

plt.tight_layout()

plt.show()

# ==========================================
# BOXPLOT FOR EACH FEATURE
# ==========================================

plt.figure(figsize=(12,8))

for i, col in enumerate(numeric_features):

    plt.subplot(2,2,i+1)

    sns.boxplot(
        data=df,
        y=col
    )

    plt.title(f'Boxplot of {col}')

plt.tight_layout()

plt.show()

# ==========================================
# OUTLIER DETECTION USING IQR
# ==========================================

Q1 = df[numeric_features].quantile(0.25)

Q3 = df[numeric_features].quantile(0.75)

IQR = Q3 - Q1

LB = Q1 - (1.5 * IQR)

UB = Q3 + (1.5 * IQR)

outliers_detected = (
    (df[numeric_features] < LB) |
    (df[numeric_features] > UB)
)

outlier_count = outliers_detected.sum()

# ==========================================
# SUMMARY TABLE
# ==========================================

summary = pd.DataFrame({
    'Q1': Q1,
    'Q3': Q3,
    'Lower Bound': LB,
    'Upper Bound': UB,
    'Outlier Count': outlier_count
})

print("\nOutlier Summary:\n")

print(summary)

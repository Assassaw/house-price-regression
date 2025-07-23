import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# === Set the folder containing your data files ===
base_path = "C:/Users/Astevenson/Downloads/house-prices-advanced-regression-techniques"

# === Safely build file paths ===
train_path = os.path.join(base_path, "train.csv")
test_path = os.path.join(base_path, "test.csv")
submission_path = os.path.join(base_path, "submission_ridge.csv")

# === Optional: Print for debugging ===
print("TRAIN PATH:", train_path)

# === Load data ===
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
test_ids = test["Id"]

# === Log-transform SalePrice for normality ===
train["SalePrice"] = np.log1p(train["SalePrice"])

# === Combine train/test for uniform preprocessing ===
all_data = pd.concat([train.drop("SalePrice", axis=1), test], axis=0)

# === Feature engineering ===
all_data["TotalSF"] = all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + all_data["2ndFlrSF"]
all_data["Age"] = all_data["YrSold"] - all_data["YearBuilt"]
all_data["RemodAge"] = all_data["YrSold"] - all_data["YearRemodAdd"]

# === Handle missing data ===
num_cols = all_data.select_dtypes(include=[np.number]).columns
cat_cols = all_data.select_dtypes(include=["object"]).columns

all_data[num_cols] = all_data[num_cols].fillna(all_data[num_cols].median())
all_data[cat_cols] = all_data[cat_cols].fillna("Missing")

# === One-hot

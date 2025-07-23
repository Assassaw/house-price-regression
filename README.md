import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

base_path = "C:/Users/Astevenson/Downloads/house-prices-advanced-regression-techniques"
train_path = os.path.join(base_path, "train.csv")
test_path = os.path.join(base_path, "test.csv")
submission_path = os.path.join(base_path, "submission_ridge.csv")

print("TRAIN PATH:", train_path)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
test_ids = test["Id"]

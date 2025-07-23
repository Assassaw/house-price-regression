import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

train = pd.read_csv(r"C:\Users\Astevenson\Downloads\house-price-advanced-regression-techniques\train.csv")
test = pd.read_csv(r"C:\Users\Astevenson\Downloads\house-price-advanced-regression-techniques\test.csv")
test_ids = test["Id"]


train["SalePrice"] = np.log1p(train["SalePrice"])


all_data = pd.concat([train.drop("SalePrice", axis=1), test], axis=0)

all_data["TotalSF"] = all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + all_data["2ndFlrSF"]
all_data["Age"] = all_data["YrSold"] - all_data["YearBuilt"]
all_data["RemodAge"] = all_data["YrSold"] - all_data["YearRemodAdd"]

num_cols = all_data.select_dtypes(include=[np.number]).columns
cat_cols = all_data.select_dtypes(include=["object"]).columns

all_data[num_cols] = all_data[num_cols].fillna(all_data[num_cols].median())
all_data[cat_cols] = all_data[cat_cols].fillna("Missing")

all_data = pd.get_dummies(all_data, columns=cat_cols, drop_first=True)


X_train = all_data.iloc[:train.shape[0]].drop("Id", axis=1)
X_test = all_data.iloc[train.shape[0]:].drop("Id", axis=1)
y_train = train["SalePrice"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


alphas = [0.1, 1, 10, 30, 100]
ridge = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=5)
ridge.fit(X_train_scaled, y_train)


train_preds = ridge.predict(X_train_scaled)
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
print("Best alpha:", ridge.alpha_)
print("Train RMSE (log scale):", round(train_rmse, 4))

test_preds = np.expm1(ridge.predict(X_test_scaled))


submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": test_preds
})
submission.to_csv(r"C:\Users\Astevenson\Downloads\house-price-advanced-regression-techniques\submission_ridge.csv", index=False)
print("Submission saved.")

# SalePrice distribution
plt.figure(figsize=(8,5))
plt.hist(np.expm1(y_train), bins=40, color="skyblue", edgecolor="black")
plt.title("Distribution of Sale Prices")
plt.xlabel("Sale Price ($)")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(X_train["TotalSF"], np.expm1(y_train), alpha=0.5)
plt.title("Total Square Footage vs Sale Price")
plt.xlabel("TotalSF")
plt.ylabel("Sale Price ($)")
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np

BASE = "/data/weizhen/code/math"
df_train = pd.read_csv(f"{BASE}/application_train.csv")

print("--- 1. Target Distribution ---")
print(df_train["TARGET"].value_counts(normalize=True) * 100)

print("\n--- 2. Top 10 Missing Values ---")
print(df_train.isna().mean().sort_values(ascending=False).head(10) * 100)

print("\n--- 3 & 4. Age and Employment Stats ---")
df_train["AGE_YEARS"] = -df_train["DAYS_BIRTH"] / 365.0
print("Age Correlation:", df_train["AGE_YEARS"].corr(df_train["TARGET"]))

df_train["DAYS_EMPLOYED_ANOM"] = df_train["DAYS_EMPLOYED"] == 365243
df_train["DAYS_EMPLOYED"] = df_train["DAYS_EMPLOYED"].replace(365243, np.nan)
df_train["EMP_YEARS"] = -df_train["DAYS_EMPLOYED"] / 365.0
print("Employment Correlation:", df_train["EMP_YEARS"].corr(df_train["TARGET"]))

print("\n--- 6. EXT_SOURCE Correlations ---")
print(df_train[["TARGET", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].corr()["TARGET"])

print("\n--- 12. Categorical Stats ---")
print(df_train.groupby("CODE_GENDER", observed=True)["TARGET"].mean() * 100)
print(df_train.groupby("NAME_CONTRACT_TYPE", observed=True)["TARGET"].mean() * 100)






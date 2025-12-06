import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")

BASE = "/data/weizhen/code/math"
OUTPUT_DIR = "/data/weizhen/code/math/plots"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Loading application_train.csv...")
df_train = pd.read_csv(f"{BASE}/application_train.csv")

# --- 1. Target Distribution ---
print("Analyzing 1. Target Distribution...")
vc = df_train["TARGET"].value_counts().sort_index()
rate = (vc / vc.sum()) * 100
plt.figure(figsize=(6, 4))
bars = plt.bar(rate.index.astype(str), rate.values)
plt.title("Target Distribution (Default Rate %)")
plt.xlabel("Target (0=Non-Default, 1=Default)")
plt.ylabel("Percentage (%)")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_target_distribution.png")
plt.close()

# --- 2. Missing Values (Top 30) ---
print("Analyzing 2. Missing Values...")
na_rate = df_train.isna().mean().sort_values(ascending=False).head(30)
plt.figure(figsize=(10, 8))
na_rate.sort_values().plot(kind="barh")
plt.title("Top 30 Features with Highest Missing Rates")
plt.xlabel("Missing Fraction")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_missing_values.png")
plt.close()

# --- 3. Age vs Default Rate ---
print("Analyzing 3. Age vs Default Rate...")
df_train["AGE_YEARS"] = -df_train["DAYS_BIRTH"] / 365.0
mask = df_train["AGE_YEARS"].notna()
bins = pd.qcut(df_train.loc[mask, "AGE_YEARS"], q=10, duplicates="drop")
rate = df_train.loc[mask].groupby(bins, observed=True)["TARGET"].mean() * 100
plt.figure(figsize=(10, 5))
rate.plot(kind="bar")
plt.title("Default Rate by Age Decile")
plt.ylabel("Default Rate (%)")
plt.xlabel("Age Group (Years)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_age_vs_default.png")
plt.close()

# --- 4. Years Employed vs Default Rate ---
print("Analyzing 4. Years Employed vs Default Rate...")
df_train["DAYS_EMPLOYED_ANOM"] = df_train["DAYS_EMPLOYED"] == 365243
df_train["DAYS_EMPLOYED"] = df_train["DAYS_EMPLOYED"].replace(365243, np.nan)
df_train["EMP_YEARS"] = -df_train["DAYS_EMPLOYED"] / 365.0
mask = df_train["EMP_YEARS"].notna()
bins = pd.qcut(df_train.loc[mask, "EMP_YEARS"], q=10, duplicates="drop")
rate = df_train.loc[mask].groupby(bins, observed=True)["TARGET"].mean() * 100
plt.figure(figsize=(10, 5))
rate.plot(kind="bar")
plt.title("Default Rate by Employment Length Decile")
plt.ylabel("Default Rate (%)")
plt.xlabel("Employment Length (Years)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_employment_vs_default.png")
plt.close()

# --- 5. Credit/Income Ratio vs Default Rate ---
print("Analyzing 5. Credit/Income Ratio vs Default Rate...")
df_train["CREDIT_INCOME_RATIO"] = df_train["AMT_CREDIT"] / (df_train["AMT_INCOME_TOTAL"] + 1.0)
upper = df_train["CREDIT_INCOME_RATIO"].quantile(0.99)
df_train["CREDIT_INCOME_RATIO"] = df_train["CREDIT_INCOME_RATIO"].clip(upper=upper)
mask = df_train["CREDIT_INCOME_RATIO"].notna()
bins = pd.qcut(df_train.loc[mask, "CREDIT_INCOME_RATIO"], q=10, duplicates="drop")
rate = df_train.loc[mask].groupby(bins, observed=True)["TARGET"].mean() * 100
plt.figure(figsize=(10, 5))
rate.plot(kind="bar")
plt.title("Default Rate by Credit/Income Ratio Decile")
plt.ylabel("Default Rate (%)")
plt.xlabel("Credit/Income Ratio")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_credit_income_ratio.png")
plt.close()

# --- 6. EXT_SOURCE_2 vs Default Rate ---
print("Analyzing 6. EXT_SOURCE_2 vs Default Rate...")
mask = df_train["EXT_SOURCE_2"].notna()
bins = pd.qcut(df_train.loc[mask, "EXT_SOURCE_2"], q=10, duplicates="drop")
rate = df_train.loc[mask].groupby(bins, observed=True)["TARGET"].mean() * 100
plt.figure(figsize=(10, 5))
rate.plot(kind="bar")
plt.title("Default Rate by EXT_SOURCE_2 Decile")
plt.ylabel("Default Rate (%)")
plt.xlabel("EXT_SOURCE_2 Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_ext_source_2.png")
plt.close()

# --- 7. Previous Application Refusals ---
print("Analyzing 7. Previous Application Refusals...")
prev = pd.read_csv(f"{BASE}/previous_application.csv", usecols=["SK_ID_CURR", "NAME_CONTRACT_STATUS"])
refused_cnt = (prev["NAME_CONTRACT_STATUS"] == "Refused").groupby(prev["SK_ID_CURR"]).sum()
df_merged = df_train[["SK_ID_CURR", "TARGET"]].merge(refused_cnt.rename("REFUSED_CNT"), left_on="SK_ID_CURR", right_index=True, how="left")
df_merged["REFUSED_CNT"] = df_merged["REFUSED_CNT"].fillna(0)
df_merged["REFUSED_BIN"] = pd.cut(df_merged["REFUSED_CNT"], bins=[-1, 0, 1, 2, 100], labels=["0", "1", "2", "3+"])
rate = df_merged.groupby("REFUSED_BIN", observed=True)["TARGET"].mean() * 100
plt.figure(figsize=(8, 5))
rate.plot(kind="bar")
plt.title("Default Rate by Number of Refused Previous Applications")
plt.ylabel("Default Rate (%)")
plt.xlabel("Count of Refusals")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_prev_app_refusals.png")
plt.close()
del prev, df_merged

# --- 8. Active Bureau Credits ---
print("Analyzing 8. Active Bureau Credits...")
bureau = pd.read_csv(f"{BASE}/bureau.csv", usecols=["SK_ID_CURR", "CREDIT_ACTIVE"])
active_cnt = (bureau["CREDIT_ACTIVE"] == "Active").groupby(bureau["SK_ID_CURR"]).sum()
df_merged = df_train[["SK_ID_CURR", "TARGET"]].merge(active_cnt.rename("ACTIVE_CNT"), left_on="SK_ID_CURR", right_index=True, how="left")
df_merged["ACTIVE_CNT"] = df_merged["ACTIVE_CNT"].fillna(0)
df_merged["ACTIVE_BIN"] = pd.cut(df_merged["ACTIVE_CNT"], bins=[-1, 0, 1, 2, 100], labels=["0", "1", "2", "3+"])
rate = df_merged.groupby("ACTIVE_BIN", observed=True)["TARGET"].mean() * 100
plt.figure(figsize=(8, 5))
rate.plot(kind="bar")
plt.title("Default Rate by Number of Active Bureau Credits")
plt.ylabel("Default Rate (%)")
plt.xlabel("Count of Active Credits")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_active_bureau_credits.png")
plt.close()
del bureau, df_merged

# --- 9. Installments Payments (Overdue) ---
print("Analyzing 9. Installments Payments...")
ins = pd.read_csv(f"{BASE}/installments_payments.csv", usecols=["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"])
ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
dpd_mean = ins.groupby("SK_ID_CURR")["DPD"].mean().rename("DPD_MEAN").reset_index()
df_merged = df_train[["SK_ID_CURR", "TARGET"]].merge(dpd_mean, on="SK_ID_CURR", how="left")
mask = df_merged["DPD_MEAN"].notna() & (df_merged["DPD_MEAN"] > 0) 
# Only plot for those who have some history, and potentially split 0 vs >0 or bins
# For simplicity, following the prompt's quantile approach but handling many 0s
# If many 0s, qcut might fail with duplicate edges if not handled.
# Let's bin: 0, and then quantiles for >0
df_merged["DPD_BIN"] = pd.cut(df_merged["DPD_MEAN"], bins=[-1, 0, 1, 5, 1000], labels=["0", "0-1", "1-5", ">5"])
rate = df_merged.groupby("DPD_BIN", observed=True)["TARGET"].mean() * 100
plt.figure(figsize=(8, 5))
rate.plot(kind="bar")
plt.title("Default Rate by Average Days Past Due (DPD)")
plt.ylabel("Default Rate (%)")
plt.xlabel("Average DPD Groups")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_installments_dpd.png")
plt.close()
del ins, df_merged, dpd_mean

# --- 10. Credit Card Utilization ---
print("Analyzing 10. Credit Card Utilization...")
cc = pd.read_csv(f"{BASE}/credit_card_balance.csv", usecols=["SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"])
cc["UTIL"] = cc["AMT_BALANCE"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1.0)
util_mean = cc.groupby("SK_ID_CURR")["UTIL"].mean().rename("CC_UTIL_MEAN").reset_index()
df_merged = df_train[["SK_ID_CURR", "TARGET"]].merge(util_mean, on="SK_ID_CURR", how="left")
mask = df_merged["CC_UTIL_MEAN"].notna()
bins = pd.qcut(df_merged.loc[mask, "CC_UTIL_MEAN"], q=10, duplicates="drop")
rate = df_merged.loc[mask].groupby(bins, observed=True)["TARGET"].mean() * 100
plt.figure(figsize=(10, 5))
rate.plot(kind="bar")
plt.title("Default Rate by Credit Card Utilization Decile")
plt.ylabel("Default Rate (%)")
plt.xlabel("Utilization Ratio")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_cc_utilization.png")
plt.close()
del cc, df_merged, util_mean

# --- 11. Correlation Heatmap (Supplement) ---
print("Analyzing 11. Correlation Heatmap...")
corr_cols = ["TARGET", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", 
             "DAYS_BIRTH", "DAYS_EMPLOYED", "AMT_CREDIT", "AMT_GOODS_PRICE", "AMT_INCOME_TOTAL"]
corr_data = df_train[corr_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_data, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Key Features")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/11_correlation_heatmap.png")
plt.close()

# --- 12. Categorical Features (Supplement) ---
print("Analyzing 12. Categorical Features...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gender
rate_gender = df_train.groupby("CODE_GENDER", observed=True)["TARGET"].mean() * 100
rate_gender.plot(kind="bar", ax=axes[0], color='skyblue')
axes[0].set_title("Default Rate by Gender")
axes[0].set_ylabel("Default Rate (%)")

# Contract Type
rate_contract = df_train.groupby("NAME_CONTRACT_TYPE", observed=True)["TARGET"].mean() * 100
rate_contract.plot(kind="bar", ax=axes[1], color='salmon')
axes[1].set_title("Default Rate by Contract Type")
axes[1].set_ylabel("Default Rate (%)")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/12_categorical_features.png")
plt.close()

print("All analysis completed. Plots saved to:", OUTPUT_DIR)


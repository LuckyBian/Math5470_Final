"""
Home Credit Default Risk - Exploratory Data Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'home-credit-default-risk/')
if not os.path.exists(DATA_PATH):
    DATA_PATH = '/home/liyiming/home-credit-default-risk/'

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def basic_info():
    print("=" * 80)
    print("Dataset Basic Info")
    print("=" * 80)
    
    train = pd.read_csv(DATA_PATH + 'application_train.csv')
    test = pd.read_csv(DATA_PATH + 'application_test.csv')
    
    print(f"\nTrain shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    print(f"\nTarget Distribution:")
    print(train['TARGET'].value_counts())
    print(f"\nDefault Rate: {train['TARGET'].mean():.4%}")
    
    return train, test

def missing_values(train):
    print("\n" + "=" * 80)
    print("Missing Values Analysis")
    print("=" * 80)
    
    missing = train.isnull().sum()
    missing_pct = 100 * missing / len(train)
    missing_df = pd.DataFrame({
        'Count': missing,
        'Percent(%)': missing_pct
    })
    missing_df = missing_df[missing_df['Count'] > 0].sort_values('Percent(%)', ascending=False)
    
    print(f"\nTotal {len(missing_df)} features have missing values")
    print(f"\nTop 10 missing features:")
    print(missing_df.head(10))
    
    return missing_df

def numerical_features_stats(train):
    print("\n" + "=" * 80)
    print("Numerical Features Stats")
    print("=" * 80)
    
    key_features = [
        'AMT_INCOME_TOTAL',
        'AMT_CREDIT',
        'AMT_ANNUITY',
        'AMT_GOODS_PRICE',
        'DAYS_BIRTH',
        'DAYS_EMPLOYED',
        'EXT_SOURCE_1',
        'EXT_SOURCE_2',
        'EXT_SOURCE_3',
    ]
    
    stats_df = train[key_features].describe()
    print("\n" + stats_df.to_string())
    
    print(f"\n\nDAYS_EMPLOYED outliers:")
    print(f"Max: {train['DAYS_EMPLOYED'].max()} (365243 is outlier)")
    print(f"Count: {(train['DAYS_EMPLOYED'] == 365243).sum()}")

def categorical_features_analysis(train):
    print("\n" + "=" * 80)
    print("Categorical Features Analysis")
    print("=" * 80)
    
    cat_features = [
        'NAME_CONTRACT_TYPE',
        'CODE_GENDER',
        'FLAG_OWN_CAR',
        'FLAG_OWN_REALTY',
        'NAME_INCOME_TYPE',
        'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE',
    ]
    
    for feat in cat_features:
        if feat in train.columns:
            print(f"\n{feat}:")
            value_counts = train[feat].value_counts()
            print(value_counts)
            
            default_rate = train.groupby(feat)['TARGET'].mean().sort_values(ascending=False)
            print(f"\nDefault rate by category:")
            for cat, rate in default_rate.items():
                print(f"  {cat}: {rate:.4%}")

def target_correlation(train):
    print("\n" + "=" * 80)
    print("Target Correlations")
    print("=" * 80)
    
    numeric_features = train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features.remove('SK_ID_CURR')
    numeric_features.remove('TARGET')
    
    correlations = train[numeric_features + ['TARGET']].corr()['TARGET'].drop('TARGET')
    correlations = correlations.abs().sort_values(ascending=False)
    
    print("\nTop 20 correlated features:")
    for feat, corr in correlations.head(20).items():
        print(f"{feat:40s}: {corr:.4f}")

def external_sources_analysis(train):
    print("\n" + "=" * 80)
    print("External Sources Analysis")
    print("=" * 80)
    
    ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    
    print("\nExternal sources vs Default rate:")
    for source in ext_sources:
        train[f'{source}_RANGE'] = pd.qcut(train[source], q=10, duplicates='drop')
        default_rate = train.groupby(f'{source}_RANGE')['TARGET'].mean()
        
        print(f"\n{source}:")
        print(default_rate.to_string())

def bureau_analysis():
    print("\n" + "=" * 80)
    print("Bureau Data Analysis")
    print("=" * 80)
    
    bureau = pd.read_csv(DATA_PATH + 'bureau.csv')
    
    print(f"\nShape: {bureau.shape}")
    print(f"Unique clients: {bureau['SK_ID_CURR'].nunique()}")
    print(f"Avg credits per client: {len(bureau) / bureau['SK_ID_CURR'].nunique():.2f}")
    
    print(f"\nCredit Active status:")
    print(bureau['CREDIT_ACTIVE'].value_counts())
    
    print(f"\nCredit Type:")
    print(bureau['CREDIT_TYPE'].value_counts())

def previous_app_analysis():
    print("\n" + "=" * 80)
    print("Previous Application Analysis")
    print("=" * 80)
    
    prev = pd.read_csv(DATA_PATH + 'previous_application.csv')
    
    print(f"\nShape: {prev.shape}")
    print(f"Unique clients: {prev['SK_ID_CURR'].nunique()}")
    print(f"Avg apps per client: {len(prev) / prev['SK_ID_CURR'].nunique():.2f}")
    
    print(f"\nContract Status:")
    print(prev['NAME_CONTRACT_STATUS'].value_counts())
    
    print(f"\nContract Type:")
    print(prev['NAME_CONTRACT_TYPE'].value_counts())

def generate_summary():
    print("\n" + "=" * 80)
    print("Data Summary")
    print("=" * 80)
    
    files = [
        'application_train.csv',
        'application_test.csv',
        'bureau.csv',
        'bureau_balance.csv',
        'previous_application.csv',
        'credit_card_balance.csv',
        'POS_CASH_balance.csv',
        'installments_payments.csv'
    ]
    
    print("\nFile sizes:")
    for file in files:
        filepath = Path(DATA_PATH) / file
        if filepath.exists():
            df = pd.read_csv(filepath)
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"{file:30s}: {df.shape[0]:>10,} rows x {df.shape[1]:>4} cols  ({size_mb:>8.2f} MB)")

def main():
    print("\n" + "=" * 80)
    print("Home Credit Default Risk - EDA")
    print("=" * 80)
    
    train, test = basic_info()
    
    missing_values(train)
    
    numerical_features_stats(train)
    
    categorical_features_analysis(train)
    
    target_correlation(train)
    
    external_sources_analysis(train)
    
    bureau_analysis()
    
    previous_app_analysis()
    
    generate_summary()
    
    print("\n" + "=" * 80)
    print("EDA Completed!")
    print("=" * 80)

if __name__ == '__main__':
    main()

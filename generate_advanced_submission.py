"""
Quick generation of advanced model submission
Uses single validation instead of full K-Fold
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
import warnings
import sys
import os

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from home_credit_advanced import (
    load_data, advanced_feature_engineering, 
    process_bureau_advanced, process_previous_application_advanced,
    process_installments_advanced, process_credit_card_advanced,
    process_pos_cash_advanced, encode_categorical_features
)

def prepare_data():
    train, test = load_data()
    
    train = advanced_feature_engineering(train, is_train=True)
    test = advanced_feature_engineering(test, is_train=False)
    
    bureau_agg = process_bureau_advanced()
    prev_agg = process_previous_application_advanced()
    ins_agg = process_installments_advanced()
    cc_agg = process_credit_card_advanced()
    pos_agg = process_pos_cash_advanced()
    
    print("Merging features...")
    train = train.merge(bureau_agg, on='SK_ID_CURR', how='left')
    train = train.merge(prev_agg, on='SK_ID_CURR', how='left')
    train = train.merge(ins_agg, on='SK_ID_CURR', how='left')
    train = train.merge(cc_agg, on='SK_ID_CURR', how='left')
    train = train.merge(pos_agg, on='SK_ID_CURR', how='left')
    
    test = test.merge(bureau_agg, on='SK_ID_CURR', how='left')
    test = test.merge(prev_agg, on='SK_ID_CURR', how='left')
    test = test.merge(ins_agg, on='SK_ID_CURR', how='left')
    test = test.merge(cc_agg, on='SK_ID_CURR', how='left')
    test = test.merge(pos_agg, on='SK_ID_CURR', how='left')
    
    train, test = encode_categorical_features(train, test)
    
    print(f"Final train shape: {train.shape}")
    print(f"Final test shape: {test.shape}")
    
    return train, test

def quick_train_and_predict():
    print("="*80)
    print("Quick Advanced Model Training")
    print("="*80)
    
    train, test = prepare_data()
    
    target = train['TARGET']
    features = [col for col in train.columns if col not in ['TARGET', 'SK_ID_CURR']]
    X = train[features]
    y = target
    X_test = test[features]
    test_id = test['SK_ID_CURR']
    
    print(f"\nNumber of features: {len(features)}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_leaves': 48,
        'max_depth': 8,
        'min_child_samples': 70,
        'subsample': 0.85,
        'subsample_freq': 1,
        'colsample_bytree': 0.85,
        'reg_alpha': 0.05,
        'reg_lambda': 0.05,
        'min_split_gain': 0.02,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    print("\nTraining...")
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=5000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=200)
        ]
    )
    
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    val_auc = roc_auc_score(y_val, val_pred)
    print(f"\nValidation AUC: {val_auc:.6f}")
    
    print("\nPredicting...")
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    submission = pd.DataFrame({
        'SK_ID_CURR': test_id,
        'TARGET': test_pred
    })
    
    submission.to_csv('submission_advanced.csv', index=False)
    print("\nSubmission saved: submission_advanced.csv")
    print(f"\nStats:\n{submission['TARGET'].describe()}")
    
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 features:")
    print(feature_importance.head(20).to_string(index=False))
    
    print("\n="*80)
    print("Done!")
    print("="*80)

if __name__ == '__main__':
    quick_train_and_predict()

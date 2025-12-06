"""
Home Credit Default Risk - Baseline Model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
import warnings
import os

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'home-credit-default-risk/')
if not os.path.exists(DATA_PATH):
    DATA_PATH = '/home/liyiming/home-credit-default-risk/'

def load_data():
    print("Loading data...")
    train = pd.read_csv(DATA_PATH + 'application_train.csv')
    test = pd.read_csv(DATA_PATH + 'application_test.csv')
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    return train, test

def feature_engineering(df, is_train=True):
    print("Feature engineering...")
    
    # Handle outliers
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    
    # Create new features
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['GOODS_PRICE_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
    
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
    df['EMPLOYED_YEARS'] = -df['DAYS_EMPLOYED'] / 365
    
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
    df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
    
    docs = [col for col in df.columns if 'FLAG_DOCUMENT' in col]
    df['DOCUMENT_COUNT'] = df[docs].sum(axis=1)
    
    df['EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['EXT_SOURCE_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    
    return df

def process_bureau_data():
    print("Processing bureau data...")
    
    bureau = pd.read_csv(DATA_PATH + 'bureau.csv')
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({
        'DAYS_CREDIT': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum', 'mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
        'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],
        'AMT_ANNUITY': ['max', 'mean']
    })
    
    bureau_agg.columns = ['BUREAU_' + '_'.join(col).upper() for col in bureau_agg.columns]
    bureau_agg.reset_index(inplace=True)
    
    bureau_counts = bureau.groupby('SK_ID_CURR').size().reset_index(name='BUREAU_COUNT')
    
    bureau_active = bureau[bureau['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR').size().reset_index(name='BUREAU_ACTIVE_COUNT')
    bureau_closed = bureau[bureau['CREDIT_ACTIVE'] == 'Closed'].groupby('SK_ID_CURR').size().reset_index(name='BUREAU_CLOSED_COUNT')
    
    bureau_agg = bureau_agg.merge(bureau_counts, on='SK_ID_CURR', how='left')
    bureau_agg = bureau_agg.merge(bureau_active, on='SK_ID_CURR', how='left')
    bureau_agg = bureau_agg.merge(bureau_closed, on='SK_ID_CURR', how='left')
    
    del bureau
    gc.collect()
    
    return bureau_agg

def process_previous_application():
    print("Processing previous_application data...")
    
    prev = pd.read_csv(DATA_PATH + 'previous_application.csv')
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum']
    })
    
    prev_agg.columns = ['PREV_' + '_'.join(col).upper() for col in prev_agg.columns]
    prev_agg.reset_index(inplace=True)
    
    prev_counts = prev.groupby('SK_ID_CURR').size().reset_index(name='PREV_COUNT')
    prev_approved = prev[prev['NAME_CONTRACT_STATUS'] == 'Approved'].groupby('SK_ID_CURR').size().reset_index(name='PREV_APPROVED_COUNT')
    prev_refused = prev[prev['NAME_CONTRACT_STATUS'] == 'Refused'].groupby('SK_ID_CURR').size().reset_index(name='PREV_REFUSED_COUNT')
    
    prev_agg = prev_agg.merge(prev_counts, on='SK_ID_CURR', how='left')
    prev_agg = prev_agg.merge(prev_approved, on='SK_ID_CURR', how='left')
    prev_agg = prev_agg.merge(prev_refused, on='SK_ID_CURR', how='left')
    
    del prev
    gc.collect()
    
    return prev_agg

def process_installments():
    print("Processing installments_payments data...")
    
    ins = pd.read_csv(DATA_PATH + 'installments_payments.csv')
    
    ins['PAYMENT_DIFF'] = ins['AMT_PAYMENT'] - ins['AMT_INSTALMENT']
    ins['PAYMENT_RATIO'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['DAYS_LATE'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DAYS_LATE'] = ins['DAYS_LATE'].apply(lambda x: x if x > 0 else 0)
    
    ins_agg = ins.groupby('SK_ID_CURR').agg({
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DAYS_INSTALMENT': ['min', 'max', 'mean'],
        'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean'],
        'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'PAYMENT_DIFF': ['min', 'max', 'mean', 'sum'],
        'PAYMENT_RATIO': ['min', 'max', 'mean'],
        'DAYS_LATE': ['max', 'mean', 'sum']
    })
    
    ins_agg.columns = ['INSTAL_' + '_'.join(col).upper() for col in ins_agg.columns]
    ins_agg.reset_index(inplace=True)
    
    ins_counts = ins.groupby('SK_ID_CURR').size().reset_index(name='INSTAL_COUNT')
    ins_agg = ins_agg.merge(ins_counts, on='SK_ID_CURR', how='left')
    
    del ins
    gc.collect()
    
    return ins_agg

def process_credit_card():
    print("Processing credit_card_balance data...")
    
    cc = pd.read_csv(DATA_PATH + 'credit_card_balance.csv')
    
    cc_agg = cc.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE': ['min', 'max', 'mean'],
        'AMT_BALANCE': ['min', 'max', 'mean'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean'],
        'AMT_DRAWINGS_ATM_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_INST_MIN_REGULARITY': ['min', 'max', 'mean'],
        'AMT_PAYMENT_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_RECEIVABLE_PRINCIPAL': ['min', 'max', 'mean'],
        'AMT_RECIVABLE': ['min', 'max', 'mean'],
        'AMT_TOTAL_RECEIVABLE': ['min', 'max', 'mean'],
        'CNT_DRAWINGS_ATM_CURRENT': ['min', 'max', 'mean', 'sum'],
        'CNT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'CNT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    })
    
    cc_agg.columns = ['CC_' + '_'.join(col).upper() for col in cc_agg.columns]
    cc_agg.reset_index(inplace=True)
    
    cc_counts = cc.groupby('SK_ID_CURR').size().reset_index(name='CC_COUNT')
    cc_agg = cc_agg.merge(cc_counts, on='SK_ID_CURR', how='left')
    
    del cc
    gc.collect()
    
    return cc_agg

def process_pos_cash():
    print("Processing POS_CASH_balance data...")
    
    pos = pd.read_csv(DATA_PATH + 'POS_CASH_balance.csv')
    
    pos_agg = pos.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE': ['min', 'max', 'mean'],
        'CNT_INSTALMENT': ['min', 'max', 'mean'],
        'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    })
    
    pos_agg.columns = ['POS_' + '_'.join(col).upper() for col in pos_agg.columns]
    pos_agg.reset_index(inplace=True)
    
    pos_counts = pos.groupby('SK_ID_CURR').size().reset_index(name='POS_COUNT')
    pos_agg = pos_agg.merge(pos_counts, on='SK_ID_CURR', how='left')
    
    del pos
    gc.collect()
    
    return pos_agg

def encode_categorical_features(train, test):
    print("Encoding categorical features...")
    
    categorical_cols = [col for col in train.columns if train[col].dtype == 'object']
    
    le = LabelEncoder()
    for col in categorical_cols:
        train[col].fillna('missing', inplace=True)
        test[col].fillna('missing', inplace=True)
        
        le.fit(list(train[col].values) + list(test[col].values))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
    
    return train, test

def prepare_data():
    train, test = load_data()
    
    train = feature_engineering(train, is_train=True)
    test = feature_engineering(test, is_train=False)
    
    bureau_agg = process_bureau_data()
    prev_agg = process_previous_application()
    ins_agg = process_installments()
    cc_agg = process_credit_card()
    pos_agg = process_pos_cash()
    
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

def train_model(train):
    print("\nTraining model...")
    
    target = train['TARGET']
    
    features = [col for col in train.columns if col not in ['TARGET', 'SK_ID_CURR']]
    X = train[features]
    y = target
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train size: {X_train.shape}, Val size: {X_val.shape}")
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=10000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=100)
        ]
    )
    
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    val_auc = roc_auc_score(y_val, val_pred)
    print(f"\nValidation AUC: {val_auc:.6f}")
    
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 features:")
    print(feature_importance.head(20))
    
    return model, features

def make_predictions(model, test, features):
    print("\nMaking predictions...")
    
    test_id = test['SK_ID_CURR']
    X_test = test[features]
    
    predictions = model.predict(X_test, num_iteration=model.best_iteration)
    
    submission = pd.DataFrame({
        'SK_ID_CURR': test_id,
        'TARGET': predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    print("Predictions saved to: submission.csv")
    print(f"Prediction stats:\n{submission['TARGET'].describe()}")
    
    return submission

def main():
    print("=" * 60)
    print("Home Credit Default Risk - Baseline Model")
    print("=" * 60)
    
    train, test = prepare_data()
    
    model, features = train_model(train)
    
    make_predictions(model, test, features)
    
    print("\n" + "=" * 60)
    print("Baseline Finished!")
    print("=" * 60)

if __name__ == '__main__':
    main()

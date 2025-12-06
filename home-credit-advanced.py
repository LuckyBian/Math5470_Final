"""
Home Credit Default Risk - Advanced Model with Feature Engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
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

def advanced_feature_engineering(df, is_train=True):
    print("Feature engineering...")
    
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
    df['EMPLOYED_YEARS'] = -df['DAYS_EMPLOYED'] / 365
    df['REGISTRATION_YEARS'] = -df['DAYS_REGISTRATION'] / 365
    df['ID_PUBLISH_YEARS'] = -df['DAYS_ID_PUBLISH'] / 365
    
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['GOODS_PRICE_INCOME_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['GOODS_PRICE_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
    df['ANNUITY_CREDIT_RATIO'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['CREDIT_GOODS_DIFF'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
    df['CREDIT_GOODS_DIFF_RATIO'] = df['CREDIT_GOODS_DIFF'] / df['AMT_GOODS_PRICE']
    
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
    df['INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (df['CNT_CHILDREN'] + 1)
    df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
    df['ADULTS_IN_FAMILY'] = df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN']
    df['HAS_CHILDREN'] = (df['CNT_CHILDREN'] > 0).astype(int)
    df['LARGE_FAMILY'] = (df['CNT_FAM_MEMBERS'] > 4).astype(int)
    
    df['AGE_GROUP'] = pd.cut(df['AGE_YEARS'], bins=[0, 25, 35, 45, 55, 100], 
                              labels=['very_young', 'young', 'middle', 'senior', 'old'])
    df['YOUNG_AGE'] = (df['AGE_YEARS'] < 30).astype(int)
    df['RETIREMENT_AGE'] = (df['AGE_YEARS'] > 60).astype(int)
    
    df['EMPLOYMENT_RATIO'] = df['EMPLOYED_YEARS'] / df['AGE_YEARS']
    df['INCOME_PER_EMPLOYED_YEAR'] = df['AMT_INCOME_TOTAL'] / (df['EMPLOYED_YEARS'] + 1)
    df['RECENTLY_EMPLOYED'] = (df['EMPLOYED_YEARS'] < 2).astype(int)
    df['LONG_EMPLOYED'] = (df['EMPLOYED_YEARS'] > 10).astype(int)
    df['UNEMPLOYED'] = df['DAYS_EMPLOYED'].isna().astype(int)
    
    df['HIGH_CREDIT'] = (df['AMT_CREDIT'] > df['AMT_CREDIT'].median()).astype(int)
    df['LOW_INCOME'] = (df['AMT_INCOME_TOTAL'] < df['AMT_INCOME_TOTAL'].median()).astype(int)
    df['HIGH_CREDIT_LOW_INCOME'] = df['HIGH_CREDIT'] * df['LOW_INCOME']
    
    docs = [col for col in df.columns if 'FLAG_DOCUMENT' in col]
    df['DOCUMENT_COUNT'] = df[docs].sum(axis=1)
    df['NO_DOCUMENTS'] = (df['DOCUMENT_COUNT'] == 0).astype(int)
    df['MANY_DOCUMENTS'] = (df['DOCUMENT_COUNT'] > 3).astype(int)
    
    df['CONTACT_INFO_COUNT'] = (
        df['FLAG_MOBIL'].fillna(0) + 
        df['FLAG_EMP_PHONE'].fillna(0) + 
        df['FLAG_WORK_PHONE'].fillna(0) + 
        df['FLAG_CONT_MOBILE'].fillna(0) + 
        df['FLAG_PHONE'].fillna(0) + 
        df['FLAG_EMAIL'].fillna(0)
    )
    df['NO_CONTACT'] = (df['CONTACT_INFO_COUNT'] == 0).astype(int)
    
    df['EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['EXT_SOURCE_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['EXT_SOURCE_MIN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
    df['EXT_SOURCE_MAX'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
    df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_WEIGHTED'] = (df['EXT_SOURCE_1'] * 0.3 + 
                                   df['EXT_SOURCE_2'] * 0.5 + 
                                   df['EXT_SOURCE_3'] * 0.2)
    
    df['EXT_SOURCE_1_2_MUL'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['EXT_SOURCE_1_3_MUL'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_2_3_MUL'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_1_2_DIFF'] = df['EXT_SOURCE_1'] - df['EXT_SOURCE_2']
    df['EXT_SOURCE_1_3_DIFF'] = df['EXT_SOURCE_1'] - df['EXT_SOURCE_3']
    df['EXT_SOURCE_2_3_DIFF'] = df['EXT_SOURCE_2'] - df['EXT_SOURCE_3']
    
    df['EXT_SOURCE_MEAN_INCOME_RATIO'] = df['EXT_SOURCE_MEAN'] * df['AMT_INCOME_TOTAL']
    df['EXT_SOURCE_MEAN_CREDIT_RATIO'] = df['EXT_SOURCE_MEAN'] * df['AMT_CREDIT']
    df['EXT_SOURCE_MEAN_AGE'] = df['EXT_SOURCE_MEAN'] * df['AGE_YEARS']
    
    df['REGION_RATING_MEAN'] = (df['REGION_RATING_CLIENT'] + 
                                 df['REGION_RATING_CLIENT_W_CITY']) / 2
    df['REGION_RATING_DIFF'] = df['REGION_RATING_CLIENT'] - df['REGION_RATING_CLIENT_W_CITY']
    
    df['YEARS_BEGINEXPLUATATION'] = -df['YEARS_BEGINEXPLUATATION_AVG']
    df['BUILDING_AGE'] = df['YEARS_BEGINEXPLUATATION']
    df['OLD_BUILDING'] = (df['BUILDING_AGE'] > 50).astype(int)
    df['NEW_BUILDING'] = (df['BUILDING_AGE'] < 10).astype(int)
    
    social_cols = [col for col in df.columns if 'SOCIAL_CIRCLE' in col]
    if len(social_cols) > 0:
        df['SOCIAL_CIRCLE_DEFAULT_RATIO'] = (
            df['DEF_30_CNT_SOCIAL_CIRCLE'] / (df['OBS_30_CNT_SOCIAL_CIRCLE'] + 1)
        )
        df['SOCIAL_CIRCLE_DEFAULT_60_RATIO'] = (
            df['DEF_60_CNT_SOCIAL_CIRCLE'] / (df['OBS_60_CNT_SOCIAL_CIRCLE'] + 1)
        )
    
    enquiry_cols = [col for col in df.columns if 'AMT_REQ_CREDIT_BUREAU' in col]
    if len(enquiry_cols) > 0:
        df['TOTAL_CREDIT_ENQUIRIES'] = df[enquiry_cols].sum(axis=1)
        df['RECENT_CREDIT_ENQUIRIES'] = (
            df['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(0) + 
            df['AMT_REQ_CREDIT_BUREAU_DAY'].fillna(0) + 
            df['AMT_REQ_CREDIT_BUREAU_WEEK'].fillna(0)
        )
        df['NO_ENQUIRIES'] = (df['TOTAL_CREDIT_ENQUIRIES'] == 0).astype(int)
        df['MANY_ENQUIRIES'] = (df['TOTAL_CREDIT_ENQUIRIES'] > 5).astype(int)
    
    df['INCOME_SQUARED'] = df['AMT_INCOME_TOTAL'] ** 2
    df['CREDIT_SQUARED'] = df['AMT_CREDIT'] ** 2
    df['AGE_SQUARED'] = df['AGE_YEARS'] ** 2
    df['EXT_SOURCE_MEAN_SQUARED'] = df['EXT_SOURCE_MEAN'] ** 2
    
    df['INCOME_LOG'] = np.log1p(df['AMT_INCOME_TOTAL'])
    df['CREDIT_LOG'] = np.log1p(df['AMT_CREDIT'])
    df['ANNUITY_LOG'] = np.log1p(df['AMT_ANNUITY'])
    df['GOODS_PRICE_LOG'] = np.log1p(df['AMT_GOODS_PRICE'])
    
    df['RISK_SCORE'] = (
        df['EXT_SOURCE_MEAN'].fillna(0.5) * 0.4 +
        (1 - df['CREDIT_INCOME_RATIO'] / df['CREDIT_INCOME_RATIO'].max()) * 0.3 +
        (df['AGE_YEARS'] / 100) * 0.2 +
        (df['EMPLOYED_YEARS'] / df['EMPLOYED_YEARS'].max()).fillna(0) * 0.1
    )
    
    df['FINANCIAL_HEALTH'] = (
        (df['AMT_INCOME_TOTAL'] > df['AMT_INCOME_TOTAL'].median()).astype(int) +
        (df['CREDIT_INCOME_RATIO'] < 3).astype(int) +
        (df['ANNUITY_INCOME_RATIO'] < 0.3).astype(int) +
        (df['EMPLOYED_YEARS'] > 2).fillna(0).astype(int)
    )
    
    print(f"Shape after FE: {df.shape}")
    
    return df

def process_bureau_advanced():
    print("Processing bureau data...")
    
    bureau = pd.read_csv(DATA_PATH + 'bureau.csv')
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({
        'DAYS_CREDIT': ['min', 'max', 'mean', 'std'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean', 'sum'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum', 'mean', 'max'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum', 'std'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum', 'std'],
        'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
        'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],
        'AMT_ANNUITY': ['max', 'mean', 'sum']
    })
    
    bureau_agg.columns = ['BUREAU_' + '_'.join(col).upper() for col in bureau_agg.columns]
    bureau_agg.reset_index(inplace=True)
    
    bureau_counts = bureau.groupby('SK_ID_CURR').size().reset_index(name='BUREAU_COUNT')
    bureau_active = bureau[bureau['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR').size().reset_index(name='BUREAU_ACTIVE_COUNT')
    bureau_closed = bureau[bureau['CREDIT_ACTIVE'] == 'Closed'].groupby('SK_ID_CURR').size().reset_index(name='BUREAU_CLOSED_COUNT')
    
    bureau_agg = bureau_agg.merge(bureau_counts, on='SK_ID_CURR', how='left')
    bureau_agg = bureau_agg.merge(bureau_active, on='SK_ID_CURR', how='left')
    bureau_agg = bureau_agg.merge(bureau_closed, on='SK_ID_CURR', how='left')
    
    bureau_agg['BUREAU_ACTIVE_RATIO'] = bureau_agg['BUREAU_ACTIVE_COUNT'] / bureau_agg['BUREAU_COUNT']
    bureau_agg['BUREAU_CLOSED_RATIO'] = bureau_agg['BUREAU_CLOSED_COUNT'] / bureau_agg['BUREAU_COUNT']
    
    bureau_agg['BUREAU_DEBT_CREDIT_RATIO'] = (
        bureau_agg['BUREAU_AMT_CREDIT_SUM_DEBT_SUM'] / bureau_agg['BUREAU_AMT_CREDIT_SUM_SUM']
    )
    
    bureau_agg['BUREAU_OVERDUE_RATIO'] = (
        bureau_agg['BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM'] / bureau_agg['BUREAU_AMT_CREDIT_SUM_SUM']
    )
    
    bureau_agg['BUREAU_AVG_CREDIT_LENGTH'] = (
        bureau_agg['BUREAU_DAYS_CREDIT_ENDDATE_MEAN'] - bureau_agg['BUREAU_DAYS_CREDIT_MEAN']
    )
    
    bureau_type_counts = bureau.groupby('SK_ID_CURR')['CREDIT_TYPE'].nunique().reset_index(name='BUREAU_CREDIT_TYPE_COUNT')
    bureau_agg = bureau_agg.merge(bureau_type_counts, on='SK_ID_CURR', how='left')
    
    for credit_type in ['Consumer credit', 'Credit card', 'Mortgage', 'Car loan']:
        bureau_type = bureau[bureau['CREDIT_TYPE'] == credit_type]
        if len(bureau_type) > 0:
            type_count = bureau_type.groupby('SK_ID_CURR').size().reset_index(name=f'BUREAU_{credit_type.upper().replace(" ", "_")}_COUNT')
            bureau_agg = bureau_agg.merge(type_count, on='SK_ID_CURR', how='left')
    
    del bureau
    gc.collect()
    
    return bureau_agg

def process_previous_application_advanced():
    print("Processing previous_application data...")
    
    prev = pd.read_csv(DATA_PATH + 'previous_application.csv')
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({
        'AMT_ANNUITY': ['min', 'max', 'mean', 'std'],
        'AMT_APPLICATION': ['min', 'max', 'mean', 'sum', 'std'],
        'AMT_CREDIT': ['min', 'max', 'mean', 'sum', 'std'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean', 'sum'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum', 'std']
    })
    
    prev_agg.columns = ['PREV_' + '_'.join(col).upper() for col in prev_agg.columns]
    prev_agg.reset_index(inplace=True)
    
    prev_counts = prev.groupby('SK_ID_CURR').size().reset_index(name='PREV_COUNT')
    prev_approved = prev[prev['NAME_CONTRACT_STATUS'] == 'Approved'].groupby('SK_ID_CURR').size().reset_index(name='PREV_APPROVED_COUNT')
    prev_refused = prev[prev['NAME_CONTRACT_STATUS'] == 'Refused'].groupby('SK_ID_CURR').size().reset_index(name='PREV_REFUSED_COUNT')
    prev_canceled = prev[prev['NAME_CONTRACT_STATUS'] == 'Canceled'].groupby('SK_ID_CURR').size().reset_index(name='PREV_CANCELED_COUNT')
    
    prev_agg = prev_agg.merge(prev_counts, on='SK_ID_CURR', how='left')
    prev_agg = prev_agg.merge(prev_approved, on='SK_ID_CURR', how='left')
    prev_agg = prev_agg.merge(prev_refused, on='SK_ID_CURR', how='left')
    prev_agg = prev_agg.merge(prev_canceled, on='SK_ID_CURR', how='left')
    
    prev_agg['PREV_APPROVED_RATIO'] = prev_agg['PREV_APPROVED_COUNT'] / prev_agg['PREV_COUNT']
    prev_agg['PREV_REFUSED_RATIO'] = prev_agg['PREV_REFUSED_COUNT'] / prev_agg['PREV_COUNT']
    prev_agg['PREV_CANCELED_RATIO'] = prev_agg['PREV_CANCELED_COUNT'] / prev_agg['PREV_COUNT']
    
    prev_agg['PREV_APP_CREDIT_RATIO'] = (
        prev_agg['PREV_AMT_APPLICATION_MEAN'] / prev_agg['PREV_AMT_CREDIT_MEAN']
    )
    
    prev_agg['PREV_DOWN_PAYMENT_RATIO'] = (
        prev_agg['PREV_AMT_DOWN_PAYMENT_MEAN'] / prev_agg['PREV_AMT_APPLICATION_MEAN']
    )
    
    prev_recent = prev.sort_values('DAYS_DECISION').groupby('SK_ID_CURR').head(3)
    prev_recent_agg = prev_recent.groupby('SK_ID_CURR').agg({
        'AMT_APPLICATION': 'mean',
        'AMT_CREDIT': 'mean',
        'DAYS_DECISION': 'mean'
    })
    prev_recent_agg.columns = ['PREV_RECENT_' + col.upper() + '_MEAN' for col in prev_recent_agg.columns]
    prev_recent_agg.reset_index(inplace=True)
    prev_agg = prev_agg.merge(prev_recent_agg, on='SK_ID_CURR', how='left')
    
    del prev, prev_recent
    gc.collect()
    
    return prev_agg

def process_installments_advanced():
    print("Processing installments_payments data...")
    
    ins = pd.read_csv(DATA_PATH + 'installments_payments.csv')
    
    ins['PAYMENT_DIFF'] = ins['AMT_PAYMENT'] - ins['AMT_INSTALMENT']
    ins['PAYMENT_RATIO'] = ins['AMT_PAYMENT'] / (ins['AMT_INSTALMENT'] + 1)
    ins['DAYS_LATE'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DAYS_LATE'] = ins['DAYS_LATE'].apply(lambda x: x if x > 0 else 0)
    ins['DAYS_EARLY'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DAYS_EARLY'] = ins['DAYS_EARLY'].apply(lambda x: x if x > 0 else 0)
    ins['IS_LATE'] = (ins['DAYS_LATE'] > 0).astype(int)
    ins['IS_UNDERPAID'] = (ins['PAYMENT_DIFF'] < 0).astype(int)
    ins['IS_OVERPAID'] = (ins['PAYMENT_DIFF'] > 0).astype(int)
    
    ins_agg = ins.groupby('SK_ID_CURR').agg({
        'NUM_INSTALMENT_VERSION': ['nunique', 'mean'],
        'DAYS_INSTALMENT': ['min', 'max', 'mean'],
        'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean'],
        'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum', 'std'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum', 'std'],
        'PAYMENT_DIFF': ['min', 'max', 'mean', 'sum', 'std'],
        'PAYMENT_RATIO': ['min', 'max', 'mean', 'std'],
        'DAYS_LATE': ['max', 'mean', 'sum'],
        'DAYS_EARLY': ['max', 'mean', 'sum'],
        'IS_LATE': ['sum', 'mean'],
        'IS_UNDERPAID': ['sum', 'mean'],
        'IS_OVERPAID': ['sum', 'mean']
    })
    
    ins_agg.columns = ['INSTAL_' + '_'.join(col).upper() for col in ins_agg.columns]
    ins_agg.reset_index(inplace=True)
    
    ins_counts = ins.groupby('SK_ID_CURR').size().reset_index(name='INSTAL_COUNT')
    ins_agg = ins_agg.merge(ins_counts, on='SK_ID_CURR', how='left')
    
    ins_agg['INSTAL_LATE_RATIO'] = ins_agg['INSTAL_IS_LATE_SUM'] / ins_agg['INSTAL_COUNT']
    ins_agg['INSTAL_UNDERPAID_RATIO'] = ins_agg['INSTAL_IS_UNDERPAID_SUM'] / ins_agg['INSTAL_COUNT']
    ins_agg['INSTAL_OVERPAID_RATIO'] = ins_agg['INSTAL_IS_OVERPAID_SUM'] / ins_agg['INSTAL_COUNT']
    ins_agg['INSTAL_PAYMENT_COMPLETION'] = ins_agg['INSTAL_AMT_PAYMENT_SUM'] / ins_agg['INSTAL_AMT_INSTALMENT_SUM']
    
    del ins
    gc.collect()
    
    return ins_agg

def process_credit_card_advanced():
    print("Processing credit_card_balance data...")
    
    cc = pd.read_csv(DATA_PATH + 'credit_card_balance.csv')
    
    cc['BALANCE_LIMIT_RATIO'] = cc['AMT_BALANCE'] / (cc['AMT_CREDIT_LIMIT_ACTUAL'] + 1)
    cc['MIN_PAYMENT_RATIO'] = cc['AMT_INST_MIN_REGULARITY'] / (cc['AMT_BALANCE'] + 1)
    cc['PAYMENT_MIN_DIFF'] = cc['AMT_PAYMENT_CURRENT'] - cc['AMT_INST_MIN_REGULARITY']
    cc['DRAWING_LIMIT_RATIO'] = cc['AMT_DRAWINGS_CURRENT'] / (cc['AMT_CREDIT_LIMIT_ACTUAL'] + 1)
    
    cc_agg = cc.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE': ['min', 'max', 'mean'],
        'AMT_BALANCE': ['min', 'max', 'mean', 'sum', 'std'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean', 'std'],
        'AMT_DRAWINGS_ATM_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'sum', 'std'],
        'AMT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_INST_MIN_REGULARITY': ['min', 'max', 'mean'],
        'AMT_PAYMENT_CURRENT': ['min', 'max', 'mean', 'sum', 'std'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['min', 'max', 'mean', 'sum'],
        'AMT_RECEIVABLE_PRINCIPAL': ['min', 'max', 'mean'],
        'AMT_RECIVABLE': ['min', 'max', 'mean', 'sum'],
        'AMT_TOTAL_RECEIVABLE': ['min', 'max', 'mean', 'sum'],
        'CNT_DRAWINGS_ATM_CURRENT': ['min', 'max', 'mean', 'sum'],
        'CNT_DRAWINGS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'CNT_DRAWINGS_POS_CURRENT': ['min', 'max', 'mean', 'sum'],
        'SK_DPD': ['max', 'mean', 'sum'],
        'SK_DPD_DEF': ['max', 'mean', 'sum'],
        'BALANCE_LIMIT_RATIO': ['mean', 'max', 'std'],
        'DRAWING_LIMIT_RATIO': ['mean', 'max']
    })
    
    cc_agg.columns = ['CC_' + '_'.join(col).upper() for col in cc_agg.columns]
    cc_agg.reset_index(inplace=True)
    
    cc_counts = cc.groupby('SK_ID_CURR').size().reset_index(name='CC_COUNT')
    cc_agg = cc_agg.merge(cc_counts, on='SK_ID_CURR', how='left')
    
    cc_agg['CC_AVG_BALANCE_LIMIT_RATIO'] = cc_agg['CC_AMT_BALANCE_MEAN'] / cc_agg['CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN']
    cc_agg['CC_DRAWING_PAYMENT_RATIO'] = cc_agg['CC_AMT_DRAWINGS_CURRENT_SUM'] / (cc_agg['CC_AMT_PAYMENT_CURRENT_SUM'] + 1)
    cc_agg['CC_DPD_RATIO'] = cc_agg['CC_SK_DPD_SUM'] / cc_agg['CC_COUNT']
    
    del cc
    gc.collect()
    
    return cc_agg

def process_pos_cash_advanced():
    print("Processing POS_CASH_balance data...")
    
    pos = pd.read_csv(DATA_PATH + 'POS_CASH_balance.csv')
    
    pos['REMAINING_INSTALMENT_RATIO'] = pos['CNT_INSTALMENT_FUTURE'] / (pos['CNT_INSTALMENT'] + 1)
    pos['COMPLETED_INSTALMENT_RATIO'] = 1 - pos['REMAINING_INSTALMENT_RATIO']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE': ['min', 'max', 'mean'],
        'CNT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
        'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean', 'sum'],
        'SK_DPD': ['max', 'mean', 'sum'],
        'SK_DPD_DEF': ['max', 'mean', 'sum'],
        'REMAINING_INSTALMENT_RATIO': ['mean', 'max'],
        'COMPLETED_INSTALMENT_RATIO': ['mean', 'min']
    })
    
    pos_agg.columns = ['POS_' + '_'.join(col).upper() for col in pos_agg.columns]
    pos_agg.reset_index(inplace=True)
    
    pos_counts = pos.groupby('SK_ID_CURR').size().reset_index(name='POS_COUNT')
    pos_agg = pos_agg.merge(pos_counts, on='SK_ID_CURR', how='left')
    
    pos_agg['POS_REMAINING_TOTAL_RATIO'] = pos_agg['POS_CNT_INSTALMENT_FUTURE_SUM'] / pos_agg['POS_CNT_INSTALMENT_SUM']
    pos_agg['POS_DPD_RATIO'] = pos_agg['POS_SK_DPD_SUM'] / pos_agg['POS_COUNT']
    
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

def train_model_with_kfold(train):
    print("\nTraining model with K-Fold CV...")
    
    target = train['TARGET']
    train_id = train['SK_ID_CURR']
    
    features = [col for col in train.columns if col not in ['TARGET', 'SK_ID_CURR']]
    X = train[features]
    y = target
    
    print(f"Number of features: {len(features)}")
    
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
    
    n_folds = 5
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(len(X))
    feature_importance_df = pd.DataFrame()
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        print(f"Fold {fold}/{n_folds}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
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
        
        oof_predictions[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        fold_score = roc_auc_score(y_val, oof_predictions[val_idx])
        cv_scores.append(fold_score)
        print(f"Fold {fold} AUC: {fold_score:.6f}")
        
        fold_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importance(importance_type='gain'),
            'fold': fold
        })
        feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)
    
    oof_score = roc_auc_score(y, oof_predictions)
    print(f"\nOverall OOF AUC: {oof_score:.6f}")
    print(f"CV Mean AUC: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")
    
    feature_importance = feature_importance_df.groupby('feature')['importance'].mean().reset_index()
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    print("\nTop 30 features:")
    print(feature_importance.head(30).to_string(index=False))
    
    print("\nTraining final model...")
    avg_best_iteration = int(np.mean([score for score in cv_scores]) * 2000)
    train_data = lgb.Dataset(X, label=y)
    final_model = lgb.train(
        params,
        train_data,
        num_boost_round=avg_best_iteration,
        valid_sets=[train_data],
        valid_names=['train'],
        callbacks=[lgb.log_evaluation(period=500)]
    )
    
    return final_model, features, feature_importance

def make_predictions(model, test, features):
    print("\nMaking predictions...")
    
    test_id = test['SK_ID_CURR']
    X_test = test[features]
    
    predictions = model.predict(X_test, num_iteration=model.best_iteration)
    
    submission = pd.DataFrame({
        'SK_ID_CURR': test_id,
        'TARGET': predictions
    })
    
    submission.to_csv('submission_advanced.csv', index=False)
    print("Predictions saved to: submission_advanced.csv")
    print(f"Prediction stats:\n{submission['TARGET'].describe()}")
    
    return submission

def main():
    print("=" * 80)
    print("Home Credit Default Risk - Advanced Model")
    print("=" * 80)
    
    train, test = prepare_data()
    
    model, features, feature_importance = train_model_with_kfold(train)
    
    submission = make_predictions(model, test, features)
    
    print("\n" + "=" * 80)
    print("Advanced Model Finished!")
    print("=" * 80)

if __name__ == '__main__':
    main()

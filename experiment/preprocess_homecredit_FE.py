import pandas as pd
import numpy as np
import gc
import re

def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def reduce_memory_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory reduced: {start_mem:.2f} MB -> {end_mem:.2f} MB "
              f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    return df

def replace_365243_with_nan(df):
    days_cols = [c for c in df.columns if 'DAYS' in c and df[c].dtype != 'object']
    for c in days_cols:
        df[c].replace(365243, np.nan, inplace=True)
    return df

def process_full_data():
    print("Processing Application Train/Test ...")
    train = pd.read_csv('application_train.csv')
    test = pd.read_csv('application_test.csv')

    df = pd.concat([train, test], ignore_index=True, sort=False)
    del train, test; gc.collect()

    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    df = replace_365243_with_nan(df)

    if 'DAYS_LAST_PHONE_CHANGE' in df.columns:
        df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)

    df, _ = one_hot_encoder(df)

    print("Processing Bureau & Balance ...")
    bureau = pd.read_csv('bureau.csv')
    bb = pd.read_csv('bureau_balance.csv')

    bureau = replace_365243_with_nan(bureau)
    bb = replace_365243_with_nan(bb)

    bb, _ = one_hot_encoder(bb)
    bb_agg = bb.groupby('SK_ID_BUREAU').agg({
        'MONTHS_BALANCE': ['min', 'max', 'size']
    })
    bb_agg.columns = ['BB_' + '_'.join(col).upper() for col in bb_agg.columns]
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    del bb, bb_agg; gc.collect()

    bureau, _ = one_hot_encoder(bureau)

    if 'SK_ID_BUREAU' in bureau.columns:
        bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)

    buro_num_agg = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_ANNUITY': ['mean']
    }
    buro_num_agg['CREDIT_DAY_OVERDUE'] = ['max', 'mean']
    buro_num_agg['AMT_CREDIT_SUM_DEBT'] = ['max', 'mean', 'sum']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg(buro_num_agg)
    bureau_agg.columns = ['BURO_' + '_'.join(col).upper() for col in bureau_agg.columns]

    df = df.join(bureau_agg, how='left', on='SK_ID_CURR')
    del bureau, bureau_agg; gc.collect()

    print("Processing Previous Applications ...")
    prev = pd.read_csv('previous_application.csv')
    prev = replace_365243_with_nan(prev)

    prev, _ = one_hot_encoder(prev)
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

    prev_num_agg = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum']
    }
    prev_agg = prev.groupby('SK_ID_CURR').agg(prev_num_agg)
    prev_agg.columns = ['PREV_' + '_'.join(col).upper() for col in prev_agg.columns]

    df = df.join(prev_agg, how='left', on='SK_ID_CURR')
    del prev, prev_agg; gc.collect()

    print("Processing Installments ...")
    ins = pd.read_csv('installments_payments.csv')
    ins = replace_365243_with_nan(ins)

    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

    ins_agg = ins.groupby('SK_ID_CURR').agg({
        'PAYMENT_PERC': ['mean', 'var'],
        'PAYMENT_DIFF': ['mean', 'var'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum']
    })
    ins_agg.columns = ['INSTAL_' + '_'.join(col).upper() for col in ins_agg.columns]

    df = df.join(ins_agg, how='left', on='SK_ID_CURR')
    del ins, ins_agg; gc.collect()

    print("Reducing memory usage ...")
    df = reduce_memory_usage(df)

    print("Saving data ...")
    df_train = df[df['TARGET'].notnull()].copy()
    df_test = df[df['TARGET'].isnull()].copy()

    def clean_columns(cols):
        return [re.sub('[^A-Za-z0-9_]+', '', c) for c in cols]

    df_train.columns = clean_columns(df_train.columns)
    df_test.columns = clean_columns(df_test.columns)

    print("Train final shape:", df_train.shape)
    print("Test final shape:", df_test.shape)

    df_train.to_pickle('train_final.pkl')
    df_test.to_pickle('test_final.pkl')

    print("Done. Data saved to train_final.pkl and test_final.pkl")

if __name__ == "__main__":
    process_full_data()

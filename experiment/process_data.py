import pandas as pd
import numpy as np
import gc

# ==========================================
# 特征工程核心函数 (多表聚合)
# ==========================================
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def process_full_data():
    print(">>> 1. 读取主表 Application Train/Test ...")
    df = pd.read_csv('application_train.csv')
    test_df = pd.read_csv('application_test.csv')
    df = pd.concat([df, test_df], ignore_index=True)
    del test_df; gc.collect()

    # 业务特征衍生
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df, _ = one_hot_encoder(df)

    # ---------------------------------------------------------
    print(">>> 2. 处理 Bureau & Balance (历史信用记录)...")
    bureau = pd.read_csv('bureau.csv')
    bb = pd.read_csv('bureau_balance.csv')
    
    bb, bb_cat = one_hot_encoder(bb)
    bb_agg = bb.groupby('SK_ID_BUREAU').agg({'MONTHS_BALANCE': ['min', 'max', 'size']})
    bb_agg.columns = ['_'.join(col).upper() for col in bb_agg.columns]
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg; gc.collect()
    
    # Bureau 聚合
    num_agg = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum']
    }
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(num_agg)
    bureau_agg.columns = ['BURO_' + '_'.join(col).upper() for col in bureau_agg.columns]
    df = df.join(bureau_agg, how='left', on='SK_ID_CURR')
    del bureau, bureau_agg; gc.collect()

    # ---------------------------------------------------------
    print(">>> 3. 处理 Previous Applications (历史申请)...")
    prev = pd.read_csv('previous_application.csv')
    prev, _ = one_hot_encoder(prev)
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    
    num_agg = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum']
    }
    prev_agg = prev.groupby('SK_ID_CURR').agg(num_agg)
    prev_agg.columns = ['PREV_' + '_'.join(col).upper() for col in prev_agg.columns]
    df = df.join(prev_agg, how='left', on='SK_ID_CURR')
    del prev, prev_agg; gc.collect()

    # ---------------------------------------------------------
    print(">>> 4. 处理 Installments (分期还款 - 关键特征)...")
    ins = pd.read_csv('installments_payments.csv')
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    
    agg = {
        'PAYMENT_PERC': ['mean', 'var'],
        'PAYMENT_DIFF': ['mean', 'var'],
        'DPD': ['max', 'mean', 'sum']
    }
    ins_agg = ins.groupby('SK_ID_CURR').agg(agg)
    ins_agg.columns = ['INSTAL_' + '_'.join(col).upper() for col in ins_agg.columns]
    df = df.join(ins_agg, how='left', on='SK_ID_CURR')
    del ins, ins_agg; gc.collect()

    # ---------------------------------------------------------
    print(">>> 5. 数据保存...")
    # 拆分回训练集和测试集
    df_train = df[df['TARGET'].notnull()]
    df_test = df[df['TARGET'].isnull()]
    
    # 修复列名 (LightGBM 不喜欢列名里有特殊符号)
    import re
    df_train = df_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    df_test = df_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    # 保存为 Pickle 格式 (读取速度比 CSV 快 10 倍，且保留数据类型)
    df_train.to_pickle('train_final.pkl')
    df_test.to_pickle('test_final.pkl')
    print("完成！数据已保存为 train_final.pkl 和 test_final.pkl")

if __name__ == "__main__":
    process_full_data()
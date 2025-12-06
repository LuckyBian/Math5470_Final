import pandas as pd
import numpy as np
import gc
import re

# ==========================================
# 工具函数：one-hot & 内存压缩 & 异常日期处理
# ==========================================

def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def reduce_memory_usage(df, verbose=True):
    """把数值列 downcast 到更小的类型，防止 OOM。"""
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
                # float
                df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory reduced: {start_mem:.2f} MB -> {end_mem:.2f} MB "
              f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    return df

def replace_365243_with_nan(df):
    """根据 Kaggle 讨论，很多 DAYS_* 列里 365243 是 fake date，统一替换为 NaN。"""
    days_cols = [c for c in df.columns if 'DAYS' in c and df[c].dtype != 'object']
    for c in days_cols:
        # 只替换精确等于 365243 的值
        df[c].replace(365243, np.nan, inplace=True)
    return df

# ==========================================
# 主流程
# ==========================================

def process_full_data():
    print(">>> 1. 读取主表 Application Train/Test ...")
    train = pd.read_csv('application_train.csv')
    test = pd.read_csv('application_test.csv')

    # 合并方便统一做特征工程；TARGET 在 test 中自动为 NaN
    df = pd.concat([train, test], ignore_index=True, sort=False)
    del train, test; gc.collect()

    # -------- 主表基础清洗 --------
    print(">>> 主表基础清洗 & 衍生特征 ...")
    # 异常的 DAYS_EMPLOYED 视为缺失（约 365243 天 ≈ 1000 年）【Kaggle 讨论 & 多篇博客】
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # 经典业务特征（在多篇 Home Credit 教程 / kernel 中被证明有效）
    # INCOME_CREDIT_PERC: 收入 / 信用额
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']

    # ANNUITY_INCOME_PERC: 年金 / 收入
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

    # CREDIT_TERM: 信用额 / 年金 （近似贷款期数）
    df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']

    # INCOME_PER_PERSON: 人均收入
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']

    # PAYMENT_RATE: 你原来已有，非常重要，保留
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    # DAYS_EMPLOYED_PERC: 工作年限占年龄的比例（都是负数，用比值表示）
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    # 处理所有包含 DAYS 的列中的 365243 异常值（防御性一点）
    df = replace_365243_with_nan(df)

    # 根据经验，部分 kernel 会把 0 的 DAYS_LAST_PHONE_CHANGE 当成缺失
    if 'DAYS_LAST_PHONE_CHANGE' in df.columns:
        df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)

    # 主表 one-hot
    df, _ = one_hot_encoder(df)

    # ==========================================
    # 2. Bureau & Bureau Balance
    # ==========================================
    print(">>> 2. 处理 Bureau & Balance (历史信用记录) ...")
    bureau = pd.read_csv('bureau.csv')
    bb = pd.read_csv('bureau_balance.csv')

    # 365243 异常日期处理
    bureau = replace_365243_with_nan(bureau)
    bb = replace_365243_with_nan(bb)

    # bureau_balance OHE + 聚合
    bb, _ = one_hot_encoder(bb)
    bb_agg = bb.groupby('SK_ID_BUREAU').agg({
        'MONTHS_BALANCE': ['min', 'max', 'size']
    })
    bb_agg.columns = ['BB_' + '_'.join(col).upper() for col in bb_agg.columns]
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    del bb, bb_agg; gc.collect()

    # bureau 自身也 OHE 一些类别字段（如 CREDIT_ACTIVE/CREDIT_TYPE）
    bureau, _ = one_hot_encoder(bureau)

    # 去掉 bureau 的 SK_ID_BUREAU 主键（聚合后没用）
    if 'SK_ID_BUREAU' in bureau.columns:
        bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)

    # 数值聚合
    buro_num_agg = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_ANNUITY': ['mean']
    }
    # 对所有数值列做一个简单的 count
    buro_num_agg['CREDIT_DAY_OVERDUE'] = ['max', 'mean']
    buro_num_agg['AMT_CREDIT_SUM_DEBT'] = ['max', 'mean', 'sum']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg(buro_num_agg)
    bureau_agg.columns = ['BURO_' + '_'.join(col).upper() for col in bureau_agg.columns]

    df = df.join(bureau_agg, how='left', on='SK_ID_CURR')
    del bureau, bureau_agg; gc.collect()

    # ==========================================
    # 3. Previous Applications
    # ==========================================
    print(">>> 3. 处理 Previous Applications (历史申请) ...")
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

    # ==========================================
    # 4. Installments Payments
    # ==========================================
    print(">>> 4. 处理 Installments (分期还款) ...")
    ins = pd.read_csv('installments_payments.csv')
    ins = replace_365243_with_nan(ins)

    # 经典还款特征（大量 kernel 使用）
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

    # （可选）你也可以在这里继续加 POS_CASH_balance 和 credit_card_balance 的聚合，
    # 但考虑到作业时间和运行速度，这里先不加，保持脚本体量适中。

    # ==========================================
    # 5. 内存压缩 & 拆分 & 列名清洗 & 保存
    # ==========================================
    print(">>> 5. 内存压缩 ...")
    df = reduce_memory_usage(df)

    print(">>> 拆回训练集和测试集 ...")
    df_train = df[df['TARGET'].notnull()].copy()
    df_test = df[df['TARGET'].isnull()].copy()

    # 列名只保留字母数字下划线，避免模型库报错
    def clean_columns(cols):
        return [re.sub('[^A-Za-z0-9_]+', '', c) for c in cols]

    df_train.columns = clean_columns(df_train.columns)
    df_test.columns = clean_columns(df_test.columns)

    print("Train final shape:", df_train.shape)
    print("Test final shape:", df_test.shape)

    print(">>> 保存为 Pickle (train_final.pkl / test_final.pkl) ...")
    df_train.to_pickle('train_final.pkl')
    df_test.to_pickle('test_final.pkl')

    print("完成！数据已保存为 train_final.pkl 和 test_final.pkl")

if __name__ == "__main__":
    process_full_data()

"""
Home Credit Default Risk - Baseline Model
===========================================
这是一个基础的 baseline，使用 LightGBM 模型进行预测
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import gc
import warnings
warnings.filterwarnings('ignore')

# 数据路径
DATA_PATH = '/home/liyiming/home-credit-default-risk/'

def load_data():
    """加载主要数据集"""
    print("正在加载数据...")
    train = pd.read_csv(DATA_PATH + 'application_train.csv')
    test = pd.read_csv(DATA_PATH + 'application_test.csv')
    print(f"训练集大小: {train.shape}")
    print(f"测试集大小: {test.shape}")
    print(f"目标变量分布: \n{train['TARGET'].value_counts(normalize=True)}")
    return train, test


def feature_engineering(df, is_train=True):
    """基础特征工程"""
    print("正在进行特征工程...")
    
    # 1. 处理异常值
    # 将 DAYS_EMPLOYED 中的异常值替换
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    
    # 2. 创建新特征
    # 收入相关特征
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['GOODS_PRICE_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
    
    # 年龄相关特征（DAYS_BIRTH 是负数）
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
    df['EMPLOYED_YEARS'] = -df['DAYS_EMPLOYED'] / 365
    
    # 家庭相关特征
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
    df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
    
    # 文档相关特征
    docs = [col for col in df.columns if 'FLAG_DOCUMENT' in col]
    df['DOCUMENT_COUNT'] = df[docs].sum(axis=1)
    
    # 外部评分相关
    df['EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['EXT_SOURCE_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    
    return df


def process_bureau_data():
    """处理 bureau 和 bureau_balance 数据"""
    print("正在处理 bureau 数据...")
    
    bureau = pd.read_csv(DATA_PATH + 'bureau.csv')
    
    # 聚合统计特征
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
    
    # 添加计数特征
    bureau_counts = bureau.groupby('SK_ID_CURR').size().reset_index(name='BUREAU_COUNT')
    
    # 按信用类型计数
    bureau_active = bureau[bureau['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR').size().reset_index(name='BUREAU_ACTIVE_COUNT')
    bureau_closed = bureau[bureau['CREDIT_ACTIVE'] == 'Closed'].groupby('SK_ID_CURR').size().reset_index(name='BUREAU_CLOSED_COUNT')
    
    # 合并
    bureau_agg = bureau_agg.merge(bureau_counts, on='SK_ID_CURR', how='left')
    bureau_agg = bureau_agg.merge(bureau_active, on='SK_ID_CURR', how='left')
    bureau_agg = bureau_agg.merge(bureau_closed, on='SK_ID_CURR', how='left')
    
    del bureau
    gc.collect()
    
    return bureau_agg


def process_previous_application():
    """处理 previous_application 数据"""
    print("正在处理 previous_application 数据...")
    
    prev = pd.read_csv(DATA_PATH + 'previous_application.csv')
    
    # 聚合统计特征
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
    
    # 添加计数特征
    prev_counts = prev.groupby('SK_ID_CURR').size().reset_index(name='PREV_COUNT')
    prev_approved = prev[prev['NAME_CONTRACT_STATUS'] == 'Approved'].groupby('SK_ID_CURR').size().reset_index(name='PREV_APPROVED_COUNT')
    prev_refused = prev[prev['NAME_CONTRACT_STATUS'] == 'Refused'].groupby('SK_ID_CURR').size().reset_index(name='PREV_REFUSED_COUNT')
    
    # 合并
    prev_agg = prev_agg.merge(prev_counts, on='SK_ID_CURR', how='left')
    prev_agg = prev_agg.merge(prev_approved, on='SK_ID_CURR', how='left')
    prev_agg = prev_agg.merge(prev_refused, on='SK_ID_CURR', how='left')
    
    del prev
    gc.collect()
    
    return prev_agg


def process_installments():
    """处理 installments_payments 数据"""
    print("正在处理 installments_payments 数据...")
    
    ins = pd.read_csv(DATA_PATH + 'installments_payments.csv')
    
    # 创建新特征
    ins['PAYMENT_DIFF'] = ins['AMT_PAYMENT'] - ins['AMT_INSTALMENT']
    ins['PAYMENT_RATIO'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['DAYS_LATE'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DAYS_LATE'] = ins['DAYS_LATE'].apply(lambda x: x if x > 0 else 0)
    
    # 聚合统计特征
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
    
    # 添加计数特征
    ins_counts = ins.groupby('SK_ID_CURR').size().reset_index(name='INSTAL_COUNT')
    ins_agg = ins_agg.merge(ins_counts, on='SK_ID_CURR', how='left')
    
    del ins
    gc.collect()
    
    return ins_agg


def process_credit_card():
    """处理 credit_card_balance 数据"""
    print("正在处理 credit_card_balance 数据...")
    
    cc = pd.read_csv(DATA_PATH + 'credit_card_balance.csv')
    
    # 聚合统计特征
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
    
    # 添加计数特征
    cc_counts = cc.groupby('SK_ID_CURR').size().reset_index(name='CC_COUNT')
    cc_agg = cc_agg.merge(cc_counts, on='SK_ID_CURR', how='left')
    
    del cc
    gc.collect()
    
    return cc_agg


def process_pos_cash():
    """处理 POS_CASH_balance 数据"""
    print("正在处理 POS_CASH_balance 数据...")
    
    pos = pd.read_csv(DATA_PATH + 'POS_CASH_balance.csv')
    
    # 聚合统计特征
    pos_agg = pos.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE': ['min', 'max', 'mean'],
        'CNT_INSTALMENT': ['min', 'max', 'mean'],
        'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    })
    
    pos_agg.columns = ['POS_' + '_'.join(col).upper() for col in pos_agg.columns]
    pos_agg.reset_index(inplace=True)
    
    # 添加计数特征
    pos_counts = pos.groupby('SK_ID_CURR').size().reset_index(name='POS_COUNT')
    pos_agg = pos_agg.merge(pos_counts, on='SK_ID_CURR', how='left')
    
    del pos
    gc.collect()
    
    return pos_agg


def encode_categorical_features(train, test):
    """编码类别特征"""
    print("正在编码类别特征...")
    
    # 找出所有类别特征
    categorical_cols = [col for col in train.columns if train[col].dtype == 'object']
    
    # Label Encoding
    le = LabelEncoder()
    for col in categorical_cols:
        # 合并 train 和 test 来确保编码一致
        train[col].fillna('missing', inplace=True)
        test[col].fillna('missing', inplace=True)
        
        # 拟合并转换
        le.fit(list(train[col].values) + list(test[col].values))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
    
    return train, test


def prepare_data():
    """准备完整的训练和测试数据"""
    # 加载主数据
    train, test = load_data()
    
    # 特征工程
    train = feature_engineering(train, is_train=True)
    test = feature_engineering(test, is_train=False)
    
    # 处理其他数据表
    bureau_agg = process_bureau_data()
    prev_agg = process_previous_application()
    ins_agg = process_installments()
    cc_agg = process_credit_card()
    pos_agg = process_pos_cash()
    
    # 合并所有特征
    print("正在合并所有特征...")
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
    
    # 编码类别特征
    train, test = encode_categorical_features(train, test)
    
    print(f"最终训练集大小: {train.shape}")
    print(f"最终测试集大小: {test.shape}")
    
    return train, test


def train_model(train):
    """训练 LightGBM 模型"""
    print("\n开始训练模型...")
    
    # 准备数据
    target = train['TARGET']
    train_id = train['SK_ID_CURR']
    
    # 删除不需要的列
    features = [col for col in train.columns if col not in ['TARGET', 'SK_ID_CURR']]
    X = train[features]
    y = target
    
    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}")
    
    # LightGBM 参数
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
    
    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # 训练模型
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
    
    # 验证集预测
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    val_auc = roc_auc_score(y_val, val_pred)
    print(f"\n验证集 AUC: {val_auc:.6f}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print("\n前 20 个重要特征:")
    print(feature_importance.head(20))
    
    return model, features


def make_predictions(model, test, features):
    """生成预测结果"""
    print("\n生成预测结果...")
    
    test_id = test['SK_ID_CURR']
    X_test = test[features]
    
    # 预测
    predictions = model.predict(X_test, num_iteration=model.best_iteration)
    
    # 创建提交文件
    submission = pd.DataFrame({
        'SK_ID_CURR': test_id,
        'TARGET': predictions
    })
    
    submission.to_csv('/home/liyiming/submission.csv', index=False)
    print("预测结果已保存到: /home/liyiming/submission.csv")
    print(f"预测结果统计:\n{submission['TARGET'].describe()}")
    
    return submission


def main():
    """主函数"""
    print("=" * 60)
    print("Home Credit Default Risk - Baseline Model")
    print("=" * 60)
    
    # 准备数据
    train, test = prepare_data()
    
    # 训练模型
    model, features = train_model(train)
    
    # 生成预测
    submission = make_predictions(model, test, features)
    
    print("\n" + "=" * 60)
    print("Baseline 完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()


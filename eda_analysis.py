"""
Home Credit Default Risk - 探索性数据分析 (EDA)
================================================
快速查看数据集的基本信息和统计特征
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 数据路径
DATA_PATH = '/home/liyiming/home-credit-default-risk/'

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def basic_info():
    """查看数据基本信息"""
    print("=" * 80)
    print("数据集基本信息")
    print("=" * 80)
    
    train = pd.read_csv(DATA_PATH + 'application_train.csv')
    test = pd.read_csv(DATA_PATH + 'application_test.csv')
    
    print(f"\n训练集形状: {train.shape}")
    print(f"测试集形状: {test.shape}")
    
    print(f"\n目标变量分布:")
    print(train['TARGET'].value_counts())
    print(f"\n违约率: {train['TARGET'].mean():.4%}")
    
    return train, test


def missing_values(train):
    """分析缺失值"""
    print("\n" + "=" * 80)
    print("缺失值分析")
    print("=" * 80)
    
    missing = train.isnull().sum()
    missing_pct = 100 * missing / len(train)
    missing_df = pd.DataFrame({
        '缺失数量': missing,
        '缺失比例(%)': missing_pct
    })
    missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values('缺失比例(%)', ascending=False)
    
    print(f"\n总共 {len(missing_df)} 个特征有缺失值")
    print(f"\n缺失比例前 10 的特征:")
    print(missing_df.head(10))
    
    return missing_df


def numerical_features_stats(train):
    """数值特征统计"""
    print("\n" + "=" * 80)
    print("重要数值特征统计")
    print("=" * 80)
    
    key_features = [
        'AMT_INCOME_TOTAL',      # 收入
        'AMT_CREDIT',            # 贷款金额
        'AMT_ANNUITY',           # 年金
        'AMT_GOODS_PRICE',       # 商品价格
        'DAYS_BIRTH',            # 出生日期（负数）
        'DAYS_EMPLOYED',         # 就业日期（负数）
        'EXT_SOURCE_1',          # 外部评分1
        'EXT_SOURCE_2',          # 外部评分2
        'EXT_SOURCE_3',          # 外部评分3
    ]
    
    stats_df = train[key_features].describe()
    print("\n" + stats_df.to_string())
    
    # 查看异常值
    print(f"\n\nDAYS_EMPLOYED 的异常值:")
    print(f"最大值: {train['DAYS_EMPLOYED'].max()} (365243 是异常值)")
    print(f"异常值数量: {(train['DAYS_EMPLOYED'] == 365243).sum()}")


def categorical_features_analysis(train):
    """类别特征分析"""
    print("\n" + "=" * 80)
    print("重要类别特征分析")
    print("=" * 80)
    
    cat_features = [
        'NAME_CONTRACT_TYPE',    # 合同类型
        'CODE_GENDER',           # 性别
        'FLAG_OWN_CAR',          # 是否拥有汽车
        'FLAG_OWN_REALTY',       # 是否拥有房产
        'NAME_INCOME_TYPE',      # 收入类型
        'NAME_EDUCATION_TYPE',   # 教育程度
        'NAME_FAMILY_STATUS',    # 婚姻状况
        'NAME_HOUSING_TYPE',     # 住房类型
    ]
    
    for feat in cat_features:
        if feat in train.columns:
            print(f"\n{feat}:")
            value_counts = train[feat].value_counts()
            print(value_counts)
            
            # 按目标变量分组的违约率
            default_rate = train.groupby(feat)['TARGET'].mean().sort_values(ascending=False)
            print(f"\n各类别违约率:")
            for cat, rate in default_rate.items():
                print(f"  {cat}: {rate:.4%}")


def target_correlation(train):
    """计算与目标变量的相关性"""
    print("\n" + "=" * 80)
    print("与目标变量相关性最高的特征")
    print("=" * 80)
    
    # 只选择数值特征
    numeric_features = train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features.remove('SK_ID_CURR')  # 移除 ID
    numeric_features.remove('TARGET')      # 移除目标变量
    
    correlations = train[numeric_features + ['TARGET']].corr()['TARGET'].drop('TARGET')
    correlations = correlations.abs().sort_values(ascending=False)
    
    print("\n前 20 个相关性最高的特征:")
    for feat, corr in correlations.head(20).items():
        print(f"{feat:40s}: {corr:.4f}")


def external_sources_analysis(train):
    """外部评分特征分析"""
    print("\n" + "=" * 80)
    print("外部评分特征分析")
    print("=" * 80)
    
    ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    
    print("\n外部评分与违约率:")
    for source in ext_sources:
        # 分成10个区间
        train[f'{source}_RANGE'] = pd.qcut(train[source], q=10, duplicates='drop')
        default_rate = train.groupby(f'{source}_RANGE')['TARGET'].mean()
        
        print(f"\n{source}:")
        print(default_rate.to_string())


def bureau_analysis():
    """Bureau 数据分析"""
    print("\n" + "=" * 80)
    print("Bureau 数据分析")
    print("=" * 80)
    
    bureau = pd.read_csv(DATA_PATH + 'bureau.csv')
    
    print(f"\nBureau 数据形状: {bureau.shape}")
    print(f"涉及的客户数量: {bureau['SK_ID_CURR'].nunique()}")
    print(f"平均每个客户的历史信贷数量: {len(bureau) / bureau['SK_ID_CURR'].nunique():.2f}")
    
    print(f"\n信用状态分布:")
    print(bureau['CREDIT_ACTIVE'].value_counts())
    
    print(f"\n信用类型分布:")
    print(bureau['CREDIT_TYPE'].value_counts())


def previous_app_analysis():
    """Previous Application 数据分析"""
    print("\n" + "=" * 80)
    print("Previous Application 数据分析")
    print("=" * 80)
    
    prev = pd.read_csv(DATA_PATH + 'previous_application.csv')
    
    print(f"\nPrevious Application 数据形状: {prev.shape}")
    print(f"涉及的客户数量: {prev['SK_ID_CURR'].nunique()}")
    print(f"平均每个客户的历史申请数量: {len(prev) / prev['SK_ID_CURR'].nunique():.2f}")
    
    print(f"\n申请状态分布:")
    print(prev['NAME_CONTRACT_STATUS'].value_counts())
    
    print(f"\n合同类型分布:")
    print(prev['NAME_CONTRACT_TYPE'].value_counts())


def generate_summary():
    """生成数据摘要报告"""
    print("\n" + "=" * 80)
    print("数据摘要")
    print("=" * 80)
    
    summary = []
    
    # 检查所有数据文件
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
    
    print("\n数据文件大小统计:")
    for file in files:
        filepath = Path(DATA_PATH) / file
        if filepath.exists():
            df = pd.read_csv(filepath)
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"{file:30s}: {df.shape[0]:>10,} 行 x {df.shape[1]:>4} 列  ({size_mb:>8.2f} MB)")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("Home Credit Default Risk - 探索性数据分析")
    print("=" * 80)
    
    # 基本信息
    train, test = basic_info()
    
    # 缺失值分析
    missing_values(train)
    
    # 数值特征统计
    numerical_features_stats(train)
    
    # 类别特征分析
    categorical_features_analysis(train)
    
    # 相关性分析
    target_correlation(train)
    
    # 外部评分分析
    external_sources_analysis(train)
    
    # Bureau 数据分析
    bureau_analysis()
    
    # Previous Application 分析
    previous_app_analysis()
    
    # 生成摘要
    generate_summary()
    
    print("\n" + "=" * 80)
    print("EDA 完成！")
    print("=" * 80)
    print("\n关键发现:")
    print("1. 数据集严重不平衡，违约率约 8%")
    print("2. 许多特征存在缺失值，需要合理处理")
    print("3. EXT_SOURCE 特征与目标变量高度相关")
    print("4. DAYS_EMPLOYED 存在异常值 (365243)")
    print("5. Bureau 和 Previous Application 数据包含丰富的历史信息")
    print("=" * 80)


if __name__ == '__main__':
    main()


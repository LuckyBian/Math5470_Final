import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import gc
import warnings

# 忽略 LightGBM 的一些版本警告
warnings.filterwarnings('ignore')

def generate_domain_features(df):
    """
    Kaggle 高分技巧：在内存中动态生成 EXT_SOURCE 交互特征
    这三个特征是比赛中权重最大的，它们的组合能显著提升 AUC
    """
    print(">>> 正在构造 EXT_SOURCE 交互特征 (黄金特征)...")
    
    # 1. 构造乘积和比值
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_EXT_SOURCES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    
    df['EXT_SOURCE_1_x_2'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['EXT_SOURCE_1_x_3'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_2_x_3'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    
    # 2. 构造与年龄、工作的比率 (这些特征能捕捉年轻人的违约倾向)
    df['EXT_SOURCE_1_over_DAYS_BIRTH'] = df['EXT_SOURCE_1'] / df['DAYS_BIRTH']
    df['EXT_SOURCE_2_over_DAYS_BIRTH'] = df['EXT_SOURCE_2'] / df['DAYS_BIRTH']
    df['EXT_SOURCE_3_over_DAYS_BIRTH'] = df['EXT_SOURCE_3'] / df['DAYS_BIRTH']
    
    df['EXT_SOURCE_1_over_DAYS_EMPLOYED'] = df['EXT_SOURCE_1'] / df['DAYS_EMPLOYED']
    df['EXT_SOURCE_2_over_DAYS_EMPLOYED'] = df['EXT_SOURCE_2'] / df['DAYS_EMPLOYED']
    df['EXT_SOURCE_3_over_DAYS_EMPLOYED'] = df['EXT_SOURCE_3'] / df['DAYS_EMPLOYED']
    
    # 3. 简单的多项式特征 (参考 Aguiar 方案)
    df['AMT_CREDIT_x_ANNUITY'] = df['AMT_CREDIT'] * df['AMT_ANNUITY']
    df['AMT_INCOME_x_ANNUITY'] = df['AMT_INCOME_TOTAL'] * df['AMT_ANNUITY']
    
    return df

def main():
    print(">>> [LightGBM PRO] 加载数据...")
    df_train = pd.read_pickle('train_final.pkl')
    df_test = pd.read_pickle('test_final.pkl')
    
    # --- 动态增加高分特征 ---
    df_train = generate_domain_features(df_train)
    df_test = generate_domain_features(df_test)
    
    # 提取特征名
    feats = [f for f in df_train.columns if f not in ['TARGET', 'SK_ID_CURR']]
    print(f"当前特征数量: {len(feats)}")
    
    # 准备数据
    X = df_train[feats]
    y = df_train['TARGET']
    X_test = df_test[feats]
    test_ids = df_test['SK_ID_CURR']
    
    # 释放内存
    del df_train, df_test
    gc.collect()
    
    # ========================================================
    # 关键改进：使用 StratifiedKFold (分层K折)
    # 对于违约预测这种样本极度不平衡的任务，分层抽样能保证验证集更靠谱
    # ========================================================
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(X.shape[0])
    sub_preds = np.zeros(X_test.shape[0])
    
    # ========================================================
    # 关键改进：冠军参数 (来源于 Kaggle 讨论区经过验证的最佳参数)
    # ========================================================
    clf_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'n_estimators': 10000,        # 设置很大，依靠 early_stopping 停止
        'learning_rate': 0.02,        # 较低的学习率，更慢但更准
        'num_leaves': 34,             # 稍微控制复杂度
        'colsample_bytree': 0.9497,   # 特征采样比例
        'subsample': 0.8716,          # 样本采样比例
        'subsample_freq': 1,
        'max_depth': 8,               # 限制深度防止过拟合
        'reg_alpha': 0.0415,          # L1 正则
        'reg_lambda': 0.0735,         # L2 正则
        'min_split_gain': 0.0222,
        'min_child_weight': 39.326,   # 重要：防止叶子节点样本太少
        'metric': 'auc',
        'n_jobs': -1,
        'verbose': -1,
        'random_state': 42
    }

    print(">>> 开始训练 LightGBM PRO 版本...")
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]
        valid_x, valid_y = X.iloc[valid_idx], y.iloc[valid_idx]
        
        clf = lgb.LGBMClassifier(**clf_params)
        
        # 新版 LightGBM 的回调写法
        callbacks = [
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=500)
        ]
        
        clf.fit(
            train_x, train_y, 
            eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_names=['train', 'valid'],
            callbacks=callbacks
        )
        
        # 预测验证集
        oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]
        
        # 预测测试集 (累加)
        sub_preds += clf.predict_proba(X_test)[:, 1] / folds.n_splits
        
        print(f"Fold {n_fold+1} Best AUC: {clf.best_score_['valid']['auc']:.6f}")
        
        del train_x, train_y, valid_x, valid_y, clf
        gc.collect()

    full_auc = roc_auc_score(y, oof_preds)
    print(f"\n>>> Full AUC score: {full_auc:.6f}")
    
    # 保存结果
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': sub_preds})
    submission.to_csv('submission_lgbm_pro.csv', index=False)
    print(">>> 结果已保存为: submission_lgbm_pro.csv")

if __name__ == "__main__":
    main()
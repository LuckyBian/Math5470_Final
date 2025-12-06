import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
import gc
import warnings

warnings.filterwarnings('ignore')

def generate_poly_features(df_train, df_test):
    """
    【提分关键】构造多项式特征
    针对最重要的 EXT_SOURCE 特征进行数学扩展（平方、立方、交互乘积）
    """
    print(">>> 正在构造多项式特征 (Polynomial Features)...")
    
    # 挑选出最重要的几个特征进行扩展
    poly_feats = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'AMT_ANNUITY']
    
    # 填充 NaN，因为 PolynomialFeatures 不支持 NaN
    # 注意：我们只在生成多项式时填充，原始特征保留 NaN 给 LGBM 处理
    imputer = SimpleImputer(strategy='median')
    
    poly_train = df_train[poly_feats].copy()
    poly_test = df_test[poly_feats].copy()
    
    poly_train = imputer.fit_transform(poly_train)
    poly_test = imputer.transform(poly_test)
    
    # 构造 3 次多项式 (Degree=3)
    poly_transformer = PolynomialFeatures(degree=3, include_bias=False, interaction_only=False)
    
    poly_train_trans = poly_transformer.fit_transform(poly_train)
    poly_test_trans = poly_transformer.transform(poly_test)
    
    # 生成新列名
    poly_feat_names = poly_transformer.get_feature_names_out(poly_feats)
    
    # 转回 DataFrame
    df_poly_train = pd.DataFrame(poly_train_trans, columns=poly_feat_names)
    df_poly_test = pd.DataFrame(poly_test_trans, columns=poly_feat_names)
    
    # 为了避免列名重复，给新特征加个前缀
    df_poly_train.columns = ['POLY_' + c for c in df_poly_train.columns]
    df_poly_test.columns = ['POLY_' + c for c in df_poly_test.columns]
    
    # 将原来的索引对齐，否则合并会出错
    df_poly_train.index = df_train.index
    df_poly_test.index = df_test.index
    
    # 剔除掉原始列（因为原始列已经在 df_train 里了，不需要重复）
    # PolynomialFeatures 会保留原始列（degree=1），我们需要去掉它们
    cols_to_drop = ['POLY_' + c for c in poly_feats]
    df_poly_train.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    df_poly_test.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    print(f"  新增多项式特征维度: {df_poly_train.shape[1]}")
    
    # 合并
    df_train = pd.concat([df_train, df_poly_train], axis=1)
    df_test = pd.concat([df_test, df_poly_test], axis=1)
    
    del df_poly_train, df_poly_test, poly_transformer
    gc.collect()
    
    return df_train, df_test

def main():
    print(">>> [LightGBM ULTRA] 加载数据...")
    df_train = pd.read_pickle('train_final.pkl')
    df_test = pd.read_pickle('test_final.pkl')
    
    # 简单的 inf 清洗
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.replace([np.inf, -np.inf], np.nan)
    
    # --- 1. 执行多项式特征扩展 ---
    df_train, df_test = generate_poly_features(df_train, df_test)
    
    # --- 2. 简单的特征筛选（剔除全空的列）---
    feats = [f for f in df_train.columns if f not in ['TARGET', 'SK_ID_CURR']]
    
    print(f"最终特征数量: {len(feats)}")
    
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(df_train.shape[0])
    sub_preds = np.zeros(df_test.shape[0])
    
    # --- 3. Ultra 参数设置 ---
    # 更小的学习率，更多的叶子，更强的正则化
    clf_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'n_estimators': 10000,
        'learning_rate': 0.015,       # 极低的学习率，精度更高
        'num_leaves': 40,             # 增加叶子节点以适应多项式特征
        'colsample_bytree': 0.7,      # 每次只选 70% 的特征，增加随机性防止过拟合
        'subsample': 0.8,
        'max_depth': 8,
        'reg_alpha': 0.1,             # 加大 L1 正则
        'reg_lambda': 0.1,            # 加大 L2 正则
        'min_split_gain': 0.02,
        'min_child_weight': 40,
        'metric': 'auc',
        'n_jobs': -1,
        'verbose': -1,
        'random_state': 42
    }

    print(">>> 开始训练 LightGBM ULTRA 版本...")
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train[feats], df_train['TARGET'])):
        train_x, train_y = df_train[feats].iloc[train_idx], df_train['TARGET'].iloc[train_idx]
        valid_x, valid_y = df_train[feats].iloc[valid_idx], df_train['TARGET'].iloc[valid_idx]
        
        clf = lgb.LGBMClassifier(**clf_params)
        
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
        
        oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]
        sub_preds += clf.predict_proba(df_test[feats])[:, 1] / folds.n_splits
        
        print(f"Fold {n_fold+1} Best AUC: {clf.best_score_['valid']['auc']:.6f}")
        
        del train_x, train_y, valid_x, valid_y, clf
        gc.collect()

    full_auc = roc_auc_score(df_train['TARGET'], oof_preds)
    print(f"\n>>> Full CV AUC score: {full_auc:.6f}")
    
    submission = pd.DataFrame({'SK_ID_CURR': df_test['SK_ID_CURR'], 'TARGET': sub_preds})
    submission.to_csv('submission_lgbm_ultra.csv', index=False)
    print(">>> 结果已保存为: submission_lgbm_ultra.csv")

if __name__ == "__main__":
    main()
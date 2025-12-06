import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import gc
import warnings

warnings.filterwarnings('ignore')

# 复用之前定义的特征工程函数（确保和 Pro 版本一致）
def generate_domain_features(df):
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_EXT_SOURCES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_1_x_2'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['EXT_SOURCE_1_x_3'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_2_x_3'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_1_over_DAYS_BIRTH'] = df['EXT_SOURCE_1'] / df['DAYS_BIRTH']
    df['EXT_SOURCE_2_over_DAYS_BIRTH'] = df['EXT_SOURCE_2'] / df['DAYS_BIRTH']
    df['EXT_SOURCE_3_over_DAYS_BIRTH'] = df['EXT_SOURCE_3'] / df['DAYS_BIRTH']
    df['EXT_SOURCE_1_over_DAYS_EMPLOYED'] = df['EXT_SOURCE_1'] / df['DAYS_EMPLOYED']
    df['EXT_SOURCE_2_over_DAYS_EMPLOYED'] = df['EXT_SOURCE_2'] / df['DAYS_EMPLOYED']
    df['EXT_SOURCE_3_over_DAYS_EMPLOYED'] = df['EXT_SOURCE_3'] / df['DAYS_EMPLOYED']
    df['AMT_CREDIT_x_ANNUITY'] = df['AMT_CREDIT'] * df['AMT_ANNUITY']
    df['AMT_INCOME_x_ANNUITY'] = df['AMT_INCOME_TOTAL'] * df['AMT_ANNUITY']
    return df

def main():
    print(">>> [Seed Averaging] 加载数据...")
    df_train = pd.read_pickle('train_final.pkl')
    df_test = pd.read_pickle('test_final.pkl')
    
    # 清洗 inf
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.replace([np.inf, -np.inf], np.nan)
    
    # 特征工程
    df_train = generate_domain_features(df_train)
    df_test = generate_domain_features(df_test)
    
    feats = [f for f in df_train.columns if f not in ['TARGET', 'SK_ID_CURR']]
    X = df_train[feats]
    y = df_train['TARGET']
    X_test = df_test[feats]
    test_ids = df_test['SK_ID_CURR']
    
    del df_train, df_test
    gc.collect()
    
    # 定义随机种子列表 (跑5次)
    SEEDS = [42, 2023, 1024, 555, 999]
    
    # 存储每次的预测结果
    all_test_preds = []
    all_oof_preds = np.zeros(len(X))
    
    # LGBM Pro 参数 (最稳的参数)
    base_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'n_estimators': 10000,
        'learning_rate': 0.02,
        'num_leaves': 34,
        'colsample_bytree': 0.9497,
        'subsample': 0.8716,
        'subsample_freq': 1,
        'max_depth': 8,
        'reg_alpha': 0.0415,
        'reg_lambda': 0.0735,
        'min_split_gain': 0.0222,
        'min_child_weight': 39.326,
        'metric': 'auc',
        'n_jobs': -1,
        'verbose': -1
    }
    
    print(f">>> 开始 Seed Averaging (共 {len(SEEDS)} 个种子)...")

    for i, seed in enumerate(SEEDS):
        print(f"\n>>> 正在训练 Seed {seed} ({i+1}/{len(SEEDS)})...")
        
        # 更新参数里的随机种子
        params = base_params.copy()
        params['random_state'] = seed
        
        # 每一轮种子也使用 StratifiedKFold，但是 shuffle 的种子也要变！
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        
        seed_oof = np.zeros(len(X))
        seed_test = np.zeros(len(X_test))
        
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
            train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]
            valid_x, valid_y = X.iloc[valid_idx], y.iloc[valid_idx]
            
            clf = lgb.LGBMClassifier(**params)
            
            callbacks = [lgb.early_stopping(100), lgb.log_evaluation(0)]
            
            clf.fit(
                train_x, train_y,
                eval_set=[(valid_x, valid_y)],
                eval_metric='auc',
                callbacks=callbacks
            )
            
            # 累加预测
            seed_oof[valid_idx] = clf.predict_proba(valid_x)[:, 1]
            seed_test += clf.predict_proba(X_test)[:, 1] / folds.n_splits
        
        print(f"Seed {seed} AUC: {roc_auc_score(y, seed_oof):.6f}")
        
        # 将这一轮种子的预测结果加入列表
        all_test_preds.append(seed_test)
        # 累加 OOF 用于计算总分
        all_oof_preds += seed_oof / len(SEEDS)

    # 计算最终平均
    final_test_pred = np.mean(all_test_preds, axis=0)
    
    print(f"\n>>> Seed Averaging Full AUC: {roc_auc_score(y, all_oof_preds):.6f}")
    
    # 保存结果
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': final_test_pred})
    submission.to_csv('submission_lgbm_seed_avg.csv', index=False)
    print(">>> 结果已保存为: submission_lgbm_seed_avg.csv")

if __name__ == "__main__":
    main()
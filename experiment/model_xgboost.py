import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import gc

def main():
    print(">>> [XGBoost] 加载数据...")
    df_train = pd.read_pickle('train_final.pkl')
    df_test = pd.read_pickle('test_final.pkl')
    
    # ============================================================
    # 【关键修复】训练前强制清洗无穷大值 (inf)
    # XGBoost 对 inf 非常敏感，必须替换为 NaN
    # ============================================================
    print(">>> 正在清洗数据中的 inf 值...")
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.replace([np.inf, -np.inf], np.nan)
    
    feats = [f for f in df_train.columns if f not in ['TARGET','SK_ID_CURR']]
    print(f"特征数量: {len(feats)}")
    
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(df_train.shape[0])
    sub_preds = np.zeros(df_test.shape[0])
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train[feats], df_train['TARGET'])):
        train_x, train_y = df_train[feats].iloc[train_idx], df_train['TARGET'].iloc[train_idx]
        valid_x, valid_y = df_train[feats].iloc[valid_idx], df_train['TARGET'].iloc[valid_idx]
        
        # XGBoost 参数
        clf = xgb.XGBClassifier(
            n_estimators=5000,
            learning_rate=0.02,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            # tree_method='hist' 是导致对 inf 敏感的主要原因，但它快，所以我们清洗数据保留它
            tree_method='hist', 
            eval_metric='auc',
            n_jobs=-1,
            early_stopping_rounds=100
        )
        
        clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=200)
        
        oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]
        sub_preds += clf.predict_proba(df_test[feats])[:, 1] / folds.n_splits
        
        del train_x, train_y, valid_x, valid_y
        gc.collect()

    print(f"Full AUC score: {roc_auc_score(df_train['TARGET'], oof_preds):.6f}")
    
    submission = pd.DataFrame({'SK_ID_CURR': df_test['SK_ID_CURR'], 'TARGET': sub_preds})
    submission.to_csv('submission_xgboost.csv', index=False)
    print("XGBoost 结果已保存。")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import gc

def main():
    print(">>> [Random Forest] 加载数据...")
    df_train = pd.read_pickle('train_final.pkl')
    df_test = pd.read_pickle('test_final.pkl')
    
    # 1. 清洗 inf
    print(">>> 清洗 inf 值...")
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.replace([np.inf, -np.inf], np.nan)

    feats = [f for f in df_train.columns if f not in ['TARGET','SK_ID_CURR']]
    
    # 2. 填充缺失值 (RF 不支持 NaN)
    print(">>> 正在填充缺失值 (这需要一点时间)...")
    imputer = SimpleImputer(strategy='median')
    
    # 为了节省内存，直接转换并覆盖
    X = imputer.fit_transform(df_train[feats])
    X_test = imputer.transform(df_test[feats])
    y = df_train['TARGET'].values
    
    print(">>> 开始训练 Random Forest...")
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(X.shape[0])
    sub_preds = np.zeros(X_test.shape[0])
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        train_x, train_y = X[train_idx], y[train_idx]
        valid_x, valid_y = X[valid_idx], y[valid_idx]
        
        # 限制深度为 10-12，防止内存溢出；树数量设为 100
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=20,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        
        clf.fit(train_x, train_y)
        
        oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]
        sub_preds += clf.predict_proba(X_test)[:, 1] / folds.n_splits
        print(f"Fold {n_fold+1} Done")
        
        del train_x, train_y, valid_x, valid_y
        gc.collect()

    print(f"Full AUC score: {roc_auc_score(y, oof_preds):.6f}")
    
    submission = pd.DataFrame({'SK_ID_CURR': df_test['SK_ID_CURR'], 'TARGET': sub_preds})
    submission.to_csv('submission_rf.csv', index=False)
    print("Random Forest 结果已保存。")

if __name__ == "__main__":
    main()
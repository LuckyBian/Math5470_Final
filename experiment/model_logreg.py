import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import gc

def main():
    print(">>> [Logistic Regression] 加载数据...")
    df_train = pd.read_pickle('train_final.pkl')
    df_test = pd.read_pickle('test_final.pkl')

    # ============================================================
    # 【关键修复】清洗 inf 值
    # Scikit-Learn 的 Imputer 和 StandardScaler 遇到 inf 会直接报错
    # ============================================================
    print(">>> 正在清洗数据中的 inf 值...")
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.replace([np.inf, -np.inf], np.nan)
    
    feats = [f for f in df_train.columns if f not in ['TARGET','SK_ID_CURR']]
    
    print(">>> 正在处理缺失值和标准化 (LR 必须步骤)...")
    # 1. 填充缺失值 (用中位数)
    imputer = SimpleImputer(strategy='median')
    
    # 2. 标准化 (0-1 缩放)
    scaler = StandardScaler()
    
    # 为了防止内存溢出，我们先合并处理再拆分（或者只用 fit_transform 训练集）
    # 这里为了代码简洁，直接对 train 进行 fit_transform
    # 注意：先把 dataframe 转 numpy array 再由 imputer 处理，可以稍微省点内存
    X = imputer.fit_transform(df_train[feats])
    X = scaler.fit_transform(X)
    
    # 对 test 做 transform (使用 train 的统计量)
    X_test = imputer.transform(df_test[feats])
    X_test = scaler.transform(X_test)
    
    y = df_train['TARGET'].values
    
    print(">>> 开始训练 LR...")
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(X.shape[0])
    sub_preds = np.zeros(X_test.shape[0])
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        train_x, train_y = X[train_idx], y[train_idx]
        valid_x, valid_y = X[valid_idx], y[valid_idx]
        
        # 使用 saga 求解器，它对大规模数据更快
        # C=0.001 是正则化系数，越小正则化越强
        clf = LogisticRegression(C=0.001, solver='saga', n_jobs=-1, max_iter=100)
        clf.fit(train_x, train_y)
        
        oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]
        sub_preds += clf.predict_proba(X_test)[:, 1] / folds.n_splits
        print(f"Fold {n_fold+1} Done")

    print(f"Full AUC score: {roc_auc_score(y, oof_preds):.6f}")
    
    submission = pd.DataFrame({'SK_ID_CURR': df_test['SK_ID_CURR'], 'TARGET': sub_preds})
    submission.to_csv('submission_logreg.csv', index=False)
    print("LR 结果已保存。")

if __name__ == "__main__":
    main()
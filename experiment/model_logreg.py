import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import gc

def main():
    print("Loading data...")
    df_train = pd.read_pickle('train_final.pkl')
    df_test = pd.read_pickle('test_final.pkl')

    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.replace([np.inf, -np.inf], np.nan)
    
    feats = [f for f in df_train.columns if f not in ['TARGET','SK_ID_CURR']]
    
    print("Imputing and Scaling...")
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X = imputer.fit_transform(df_train[feats])
    X = scaler.fit_transform(X)
    
    X_test = imputer.transform(df_test[feats])
    X_test = scaler.transform(X_test)
    
    y = df_train['TARGET'].values
    
    print("Training LR...")
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(X.shape[0])
    sub_preds = np.zeros(X_test.shape[0])
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        train_x, train_y = X[train_idx], y[train_idx]
        valid_x, valid_y = X[valid_idx], y[valid_idx]
        
        clf = LogisticRegression(C=0.001, solver='saga', n_jobs=-1, max_iter=100)
        clf.fit(train_x, train_y)
        
        oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]
        sub_preds += clf.predict_proba(X_test)[:, 1] / folds.n_splits
        print(f"Fold {n_fold+1} Done")

    print(f"Full AUC score: {roc_auc_score(y, oof_preds):.6f}")
    
    submission = pd.DataFrame({'SK_ID_CURR': df_test['SK_ID_CURR'], 'TARGET': sub_preds})
    submission.to_csv('submission_logreg.csv', index=False)
    print("Results saved to submission_logreg.csv")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import gc

def main():
    print("Loading data...")
    df_train = pd.read_pickle('train_final.pkl')
    df_test = pd.read_pickle('test_final.pkl')
    
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.replace([np.inf, -np.inf], np.nan)

    y = df_train['TARGET']
    test_ids = df_test['SK_ID_CURR']
    
    knn_feats = [
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 
        'CREDIT_TERM', 'PAYMENT_RATE', 'DAYS_BIRTH', 'DAYS_EMPLOYED'
    ]
    
    knn_feats = [f for f in knn_feats if f in df_train.columns]
    print(f"Features used for KNN: {knn_feats}")
    
    train_len = len(df_train)
    df_all = pd.concat([df_train[knn_feats], df_test[knn_feats]], axis=0)
    
    print("Imputing and Scaling...")
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_all = imputer.fit_transform(df_all)
    X_all = scaler.fit_transform(X_all)
    
    del df_all, df_train, df_test
    gc.collect()
    
    X_train = X_all[:train_len]
    X_test = X_all[train_len:]
    
    print("Calculating KNN (500 neighbors)...")
    neigh = NearestNeighbors(n_neighbors=500, n_jobs=-1)
    neigh.fit(X_train)
    
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn_oof = np.zeros(train_len)
    knn_test_preds = np.zeros(len(X_test))
    
    print("Calculating test set neighbors...")
    inds = neigh.kneighbors(X_test, return_distance=False)
    
    y_values = y.values
    knn_test_preds = y_values[inds].mean(axis=1)
    
    print("Calculating train set OOF...")
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, y)):
        X_tr_fold = X_train[train_idx]
        X_val_fold = X_train[valid_idx]
        y_tr_fold = y.iloc[train_idx].values
        
        nn_fold = NearestNeighbors(n_neighbors=500, n_jobs=-1)
        nn_fold.fit(X_tr_fold)
        
        inds_fold = nn_fold.kneighbors(X_val_fold, return_distance=False)
        
        knn_oof[valid_idx] = y_tr_fold[inds_fold].mean(axis=1)
        print(f"Fold {n_fold+1} Done")

    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': knn_test_preds})
    submission.to_csv('submission_knn.csv', index=False)
    
    oof_df = pd.DataFrame({'TARGET': knn_oof})
    oof_df.to_csv('oof_knn.csv', index=False)
    
    print("Results saved to submission_knn.csv")

if __name__ == "__main__":
    main()

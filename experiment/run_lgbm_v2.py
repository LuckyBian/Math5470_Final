import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import gc
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Loading data train_v2.pkl ...")
    df_train = pd.read_pickle('train_v2.pkl')
    df_test = pd.read_pickle('test_v2.pkl')
    
    print("Checking and fixing data types...")
    
    obj_cols = [col for col in df_train.columns if df_train[col].dtype == 'object']
    
    if len(obj_cols) > 0:
        print(f"Found {len(obj_cols)} object columns, forcing conversion to float...")
        for col in obj_cols:
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
            df_test[col] = pd.to_numeric(df_test[col], errors='coerce')
    
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.replace([np.inf, -np.inf], np.nan)

    feats = [f for f in df_train.columns if f not in ['TARGET', 'SK_ID_CURR']]
    print(f"Number of features: {len(feats)}")
    
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1001)
    oof_preds = np.zeros(df_train.shape[0])
    sub_preds = np.zeros(df_test.shape[0])
    
    print("Training LGBM V2...")
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train[feats], df_train['TARGET'])):
        train_x, train_y = df_train[feats].iloc[train_idx], df_train['TARGET'].iloc[train_idx]
        valid_x, valid_y = df_train[feats].iloc[valid_idx], df_train['TARGET'].iloc[valid_idx]
        
        clf = lgb.LGBMClassifier(
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9,
            subsample=0.87,
            max_depth=8,
            reg_alpha=0.041,
            reg_lambda=0.073,
            min_split_gain=0.022,
            min_child_weight=39,
            n_jobs=-1,
            verbose=-1
        )
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=500)
        ]
        
        clf.fit(
            train_x, train_y, 
            eval_set=[(valid_x, valid_y)], 
            eval_metric='auc', 
            callbacks=callbacks
        )
        
        oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]
        sub_preds += clf.predict_proba(df_test[feats])[:, 1] / folds.n_splits
        
        best_score = clf.best_score_['valid_0']['auc']
        print(f"Fold {n_fold+1} Best AUC: {best_score:.6f}")
        
        del train_x, train_y, valid_x, valid_y
        gc.collect()

    print(f"LGBM V2 Full AUC: {roc_auc_score(df_train['TARGET'], oof_preds):.6f}")
    
    pd.DataFrame({'SK_ID_CURR': df_test['SK_ID_CURR'], 'TARGET': sub_preds}).to_csv('submission_lgbm_v2.csv', index=False)
    print("Results saved to submission_lgbm_v2.csv")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import gc

def main():
    print("Loading data...")
    df_train = pd.read_pickle('train_final.pkl')
    df_test = pd.read_pickle('test_final.pkl')
    
    feats = [f for f in df_train.columns if f not in ['TARGET','SK_ID_CURR']]
    print(f"Number of features: {len(feats)}")
    
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(df_train.shape[0])
    sub_preds = np.zeros(df_test.shape[0])
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train[feats], df_train['TARGET'])):
        train_x, train_y = df_train[feats].iloc[train_idx], df_train['TARGET'].iloc[train_idx]
        valid_x, valid_y = df_train[feats].iloc[valid_idx], df_train['TARGET'].iloc[valid_idx]
        
        clf = lgb.LGBMClassifier(
            n_estimators=5000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9,
            subsample=0.8,
            max_depth=8,
            reg_alpha=0.04,
            reg_lambda=0.07,
            min_child_weight=40,
            n_jobs=-1,
            verbose=-1
        )
        
        clf.fit(train_x, train_y, 
                eval_set=[(valid_x, valid_y)], 
                eval_metric='auc',
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
        
        oof_preds[valid_idx] = clf.predict_proba(valid_x)[:, 1]
        sub_preds += clf.predict_proba(df_test[feats])[:, 1] / folds.n_splits
        
        del train_x, train_y, valid_x, valid_y
        gc.collect()
    
    print(f"Full AUC score: {roc_auc_score(df_train['TARGET'], oof_preds):.6f}")
    
    submission = pd.DataFrame({'SK_ID_CURR': df_test['SK_ID_CURR'], 'TARGET': sub_preds})
    submission.to_csv('submission_lgbm.csv', index=False)
    print("Results saved to submission_lgbm.csv")

if __name__ == "__main__":
    main()

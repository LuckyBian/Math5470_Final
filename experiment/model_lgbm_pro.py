import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import gc
import warnings

warnings.filterwarnings('ignore')

def generate_domain_features(df):
    print("Generating domain features...")
    
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
    print("Loading data...")
    df_train = pd.read_pickle('train_final.pkl')
    df_test = pd.read_pickle('test_final.pkl')
    
    df_train = generate_domain_features(df_train)
    df_test = generate_domain_features(df_test)
    
    feats = [f for f in df_train.columns if f not in ['TARGET', 'SK_ID_CURR']]
    print(f"Number of features: {len(feats)}")
    
    X = df_train[feats]
    y = df_train['TARGET']
    X_test = df_test[feats]
    test_ids = df_test['SK_ID_CURR']
    
    del df_train, df_test
    gc.collect()
    
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(X.shape[0])
    sub_preds = np.zeros(X_test.shape[0])
    
    clf_params = {
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
        'verbose': -1,
        'random_state': 42
    }

    print("Training LightGBM PRO...")
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]
        valid_x, valid_y = X.iloc[valid_idx], y.iloc[valid_idx]
        
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
        sub_preds += clf.predict_proba(X_test)[:, 1] / folds.n_splits
        
        print(f"Fold {n_fold+1} Best AUC: {clf.best_score_['valid']['auc']:.6f}")
        
        del train_x, train_y, valid_x, valid_y, clf
        gc.collect()

    full_auc = roc_auc_score(y, oof_preds)
    print(f"\nFull AUC score: {full_auc:.6f}")
    
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': sub_preds})
    submission.to_csv('submission_lgbm_pro.csv', index=False)
    print("Results saved to submission_lgbm_pro.csv")

if __name__ == "__main__":
    main()

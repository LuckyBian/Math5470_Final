import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import gc
import warnings

warnings.filterwarnings('ignore')

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
    print("Loading data...")
    df_train = pd.read_pickle('train_final.pkl')
    df_test = pd.read_pickle('test_final.pkl')
    
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.replace([np.inf, -np.inf], np.nan)
    
    df_train = generate_domain_features(df_train)
    df_test = generate_domain_features(df_test)
    
    feats = [f for f in df_train.columns if f not in ['TARGET', 'SK_ID_CURR']]
    X = df_train[feats]
    y = df_train['TARGET']
    X_test = df_test[feats]
    test_ids = df_test['SK_ID_CURR']
    
    del df_train, df_test
    gc.collect()
    
    SEEDS = [42, 2023, 1024, 555, 999]
    
    all_test_preds = []
    all_oof_preds = np.zeros(len(X))
    
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
    
    print(f"Starting Seed Averaging ({len(SEEDS)} seeds)...")

    for i, seed in enumerate(SEEDS):
        print(f"Training Seed {seed} ({i+1}/{len(SEEDS)})...")
        
        params = base_params.copy()
        params['random_state'] = seed
        
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
            
            seed_oof[valid_idx] = clf.predict_proba(valid_x)[:, 1]
            seed_test += clf.predict_proba(X_test)[:, 1] / folds.n_splits
        
        print(f"Seed {seed} AUC: {roc_auc_score(y, seed_oof):.6f}")
        
        all_test_preds.append(seed_test)
        all_oof_preds += seed_oof / len(SEEDS)

    final_test_pred = np.mean(all_test_preds, axis=0)
    
    print(f"Seed Averaging Full AUC: {roc_auc_score(y, all_oof_preds):.6f}")
    
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': final_test_pred})
    submission.to_csv('submission_lgbm_seed_avg.csv', index=False)
    print("Results saved to submission_lgbm_seed_avg.csv")

if __name__ == "__main__":
    main()

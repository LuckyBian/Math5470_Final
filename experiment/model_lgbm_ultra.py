import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
import gc
import warnings

warnings.filterwarnings('ignore')

def generate_poly_features(df_train, df_test):
    print("Generating Polynomial Features...")
    
    poly_feats = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'AMT_ANNUITY']
    
    imputer = SimpleImputer(strategy='median')
    
    poly_train = df_train[poly_feats].copy()
    poly_test = df_test[poly_feats].copy()
    
    poly_train = imputer.fit_transform(poly_train)
    poly_test = imputer.transform(poly_test)
    
    poly_transformer = PolynomialFeatures(degree=3, include_bias=False, interaction_only=False)
    
    poly_train_trans = poly_transformer.fit_transform(poly_train)
    poly_test_trans = poly_transformer.transform(poly_test)
    
    poly_feat_names = poly_transformer.get_feature_names_out(poly_feats)
    
    df_poly_train = pd.DataFrame(poly_train_trans, columns=poly_feat_names)
    df_poly_test = pd.DataFrame(poly_test_trans, columns=poly_feat_names)
    
    df_poly_train.columns = ['POLY_' + c for c in df_poly_train.columns]
    df_poly_test.columns = ['POLY_' + c for c in df_poly_test.columns]
    
    df_poly_train.index = df_train.index
    df_poly_test.index = df_test.index
    
    cols_to_drop = ['POLY_' + c for c in poly_feats]
    df_poly_train.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    df_poly_test.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    print(f"  New poly features: {df_poly_train.shape[1]}")
    
    df_train = pd.concat([df_train, df_poly_train], axis=1)
    df_test = pd.concat([df_test, df_poly_test], axis=1)
    
    del df_poly_train, df_poly_test, poly_transformer
    gc.collect()
    
    return df_train, df_test

def main():
    print("Loading data...")
    df_train = pd.read_pickle('train_final.pkl')
    df_test = pd.read_pickle('test_final.pkl')
    
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.replace([np.inf, -np.inf], np.nan)
    
    df_train, df_test = generate_poly_features(df_train, df_test)
    
    feats = [f for f in df_train.columns if f not in ['TARGET', 'SK_ID_CURR']]
    
    print(f"Total features: {len(feats)}")
    
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(df_train.shape[0])
    sub_preds = np.zeros(df_test.shape[0])
    
    clf_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'n_estimators': 10000,
        'learning_rate': 0.015,
        'num_leaves': 40,
        'colsample_bytree': 0.7,
        'subsample': 0.8,
        'max_depth': 8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_split_gain': 0.02,
        'min_child_weight': 40,
        'metric': 'auc',
        'n_jobs': -1,
        'verbose': -1,
        'random_state': 42
    }

    print("Training LightGBM ULTRA...")
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train[feats], df_train['TARGET'])):
        train_x, train_y = df_train[feats].iloc[train_idx], df_train['TARGET'].iloc[train_idx]
        valid_x, valid_y = df_train[feats].iloc[valid_idx], df_train['TARGET'].iloc[valid_idx]
        
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
        sub_preds += clf.predict_proba(df_test[feats])[:, 1] / folds.n_splits
        
        print(f"Fold {n_fold+1} Best AUC: {clf.best_score_['valid']['auc']:.6f}")
        
        del train_x, train_y, valid_x, valid_y, clf
        gc.collect()

    full_auc = roc_auc_score(df_train['TARGET'], oof_preds)
    print(f"\nFull CV AUC score: {full_auc:.6f}")
    
    submission = pd.DataFrame({'SK_ID_CURR': df_test['SK_ID_CURR'], 'TARGET': sub_preds})
    submission.to_csv('submission_lgbm_ultra.csv', index=False)
    print("Results saved to submission_lgbm_ultra.csv")

if __name__ == "__main__":
    main()

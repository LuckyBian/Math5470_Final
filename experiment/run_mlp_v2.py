import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import gc

def main():
    print(">>> 加载数据用于 MLP ...")
    df_train = pd.read_pickle('train_v2.pkl')
    df_test = pd.read_pickle('test_v2.pkl')
    
    # 替换 inf
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.replace([np.inf, -np.inf], np.nan)
    
    feats = [f for f in df_train.columns if f not in ['TARGET', 'SK_ID_CURR']]
    
    print(">>> 填充缺失值 (MLP必须) ...")
    imputer = SimpleImputer(strategy='mean')
    scaler = MinMaxScaler()
    
    # 分步处理以省内存
    X = imputer.fit_transform(df_train[feats])
    X = scaler.fit_transform(X)
    X_test = imputer.transform(df_test[feats])
    X_test = scaler.transform(X_test)
    
    y = df_train['TARGET'].values
    test_ids = df_test['SK_ID_CURR']
    
    del df_train, df_test
    gc.collect()
    
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sub_preds = np.zeros(len(X_test))
    
    print(">>> 开始训练 MLP ...")
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        clf = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', 
                            alpha=0.0001, batch_size=256, learning_rate_init=0.001, 
                            max_iter=100, early_stopping=True, verbose=True)
        
        clf.fit(X[train_idx], y[train_idx])
        sub_preds += clf.predict_proba(X_test)[:, 1] / folds.n_splits
    
    pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': sub_preds}).to_csv('submission_mlp_v2.csv', index=False)
    print("MLP 完成！")

if __name__ == "__main__":
    main()
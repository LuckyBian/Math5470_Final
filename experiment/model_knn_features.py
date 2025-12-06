import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import gc

def main():
    print(">>> [KNN Features] 正在加载数据...")
    df_train = pd.read_pickle('train_final.pkl')
    df_test = pd.read_pickle('test_final.pkl')
    
    # 清洗 inf
    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.replace([np.inf, -np.inf], np.nan)

    # 提取 TARGET 和 ID
    y = df_train['TARGET']
    test_ids = df_test['SK_ID_CURR']
    
    # -----------------------------------------------------------
    # 关键点：只选择最重要的特征来计算距离
    # -----------------------------------------------------------
    knn_feats = [
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 
        'CREDIT_TERM', 'PAYMENT_RATE', 'DAYS_BIRTH', 'DAYS_EMPLOYED'
    ]
    
    # 确保特征存在
    knn_feats = [f for f in knn_feats if f in df_train.columns]
    print(f">>> 用于 KNN 距离计算的特征: {knn_feats}")
    
    # 合并处理归一化
    train_len = len(df_train)
    df_all = pd.concat([df_train[knn_feats], df_test[knn_feats]], axis=0)
    
    # 填充缺失值 + 标准化 (KNN 对数值幅度非常敏感，必须标准化)
    print(">>> 正在填充缺失值和标准化...")
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_all = imputer.fit_transform(df_all)
    X_all = scaler.fit_transform(X_all)
    
    del df_all, df_train, df_test
    gc.collect()
    
    # 拆分回训练和测试
    X_train = X_all[:train_len]
    X_test = X_all[train_len:]
    
    # -----------------------------------------------------------
    # 计算 KNN 特征
    # -----------------------------------------------------------
    
    print(">>> 开始计算 KNN (寻找 500 个最近邻)... 这可能需要几分钟...")
    neigh = NearestNeighbors(n_neighbors=500, n_jobs=-1)
    neigh.fit(X_train)
    
    # 为了避免过拟合，训练集必须用 CV 方式计算
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    knn_oof = np.zeros(train_len)
    knn_test_preds = np.zeros(len(X_test))
    
    # ========================================================
    # 【修复 1】测试集部分：因为 return_distance=False，只接收一个返回值
    # ========================================================
    print("  正在计算测试集的邻居...")
    # 错误写法: dists, inds = ...
    # 正确写法:
    inds = neigh.kneighbors(X_test, return_distance=False)
    
    # 对每个测试样本，找到它在训练集里的 500 个邻居，计算这些邻居的违约率
    y_values = y.values
    knn_test_preds = y_values[inds].mean(axis=1)
    
    # ========================================================
    # 【修复 2】循环部分：同样只接收一个返回值
    # ========================================================
    print("  正在计算训练集 OOF (防止数据泄露)...")
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X_train, y)):
        X_tr_fold = X_train[train_idx]
        X_val_fold = X_train[valid_idx]
        y_tr_fold = y.iloc[train_idx].values
        
        # 用这一折的训练数据构建 NN
        nn_fold = NearestNeighbors(n_neighbors=500, n_jobs=-1)
        nn_fold.fit(X_tr_fold)
        
        # 找到验证集在训练集里的邻居
        # 错误写法: _, inds_fold = ...
        # 正确写法:
        inds_fold = nn_fold.kneighbors(X_val_fold, return_distance=False)
        
        # 计算均值
        knn_oof[valid_idx] = y_tr_fold[inds_fold].mean(axis=1)
        print(f"  Fold {n_fold+1} Done")

    # 保存结果
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': knn_test_preds})
    submission.to_csv('submission_knn.csv', index=False)
    
    # 同时保存 OOF 结果，用于 Stacking
    oof_df = pd.DataFrame({'TARGET': knn_oof})
    oof_df.to_csv('oof_knn.csv', index=False)
    
    print(">>> KNN 结果已保存为 submission_knn.csv")
    print(">>> 注意：KNN 单模型分数不会很高，但它是融合神器！")

if __name__ == "__main__":
    main()
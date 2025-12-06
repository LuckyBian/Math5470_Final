import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def main():
    print(">>> 正在读取各个模型的预测结果...")
    
    # 定义模型列表和期望的权重
    # 如果你某个模型没跑，脚本会自动跳过，并重新归一化权重
    model_configs = [
        # 文件名,                                   权重,   模型简称
        ('submission_lgbm_ultra.csv',             0.65,   'LGBM_Ultra'),
        ('submission_lgbm_pro.csv',               0.00,   'LGBM_Pro'), # Pro和Ultra相关性太高，通常只取最强的Ultra即可，设为0
        ('submission_xgboost.csv',                0.30,   'XGBoost'),
        ('submission_logreg.csv',                 0.05,   'LogReg'),   # 线性模型，增加多样性
        ('submission_rf.csv',                     0.05,   'RandomForest'),
        ('submission_catboost.csv',               0.00,   'CatBoost')  # 你说不想用Cat，这里设为0或保留备用
    ]

    dfs = []
    valid_weights = []
    model_names = []
    
    # 读取并检查文件是否存在
    base_df = None
    
    for filename, weight, name in model_configs:
        if os.path.exists(filename):
            print(f"  [√] 发现模型: {name} (权重: {weight})")
            df = pd.read_csv(filename)
            
            # 排序以防万一
            df = df.sort_values('SK_ID_CURR')
            
            if base_df is None:
                base_df = df[['SK_ID_CURR']].copy()
            
            # 检查 ID 是否对齐
            if not df['SK_ID_CURR'].equals(base_df['SK_ID_CURR']):
                print(f"  [!] 警告: {name} 的 SK_ID_CURR 顺序不一致，正在尝试合并对齐...")
                df = pd.merge(base_df, df, on='SK_ID_CURR', how='left')
            
            dfs.append(df['TARGET'])
            # 如果权重 > 0 才计入有效权重，否则只用来做相关性分析
            if weight > 0:
                valid_weights.append(weight)
                model_names.append(name)
            else:
                # 即使权重为0，为了后面代码不报错，我们暂不加入 names 列表用于加权
                pass
        else:
            print(f"  [x] 未找到模型: {name} (跳过)")

    if not dfs:
        print("错误：没有找到任何 submission 文件！")
        return

    # 构造预测矩阵
    # 注意：这里我们需要把所有读进来的 df (包括权重为0的) 都放进来看看相关性
    # 但为了加权，我们需要筛选
    
    all_preds = pd.concat(dfs, axis=1)
    # 临时给列名，方便看相关性
    # 注意：dfs 列表包含了所有存在的文件的列，我们需要根据存在的 files 来命名
    existing_models = [m[2] for m in model_configs if os.path.exists(m[0])]
    all_preds.columns = existing_models
    
    # --- 1. 计算并打印相关性矩阵 ---
    print("\n>>> 模型预测结果的相关性矩阵 (越低越好):")
    corr = all_preds.corr()
    print(corr)
    
    # --- 2. 归一化权重 ---
    # 过滤出权重 > 0 的模型列
    active_models = [name for name, weight in zip([m[2] for m in model_configs if os.path.exists(m[0])], 
                                                  [m[1] for m in model_configs if os.path.exists(m[0])]) 
                     if weight > 0]
    
    active_weights = [weight for name, weight in zip([m[2] for m in model_configs if os.path.exists(m[0])], 
                                                     [m[1] for m in model_configs if os.path.exists(m[0])]) 
                      if weight > 0]
    
    if not active_weights:
        print("没有权重 > 0 的模型，无法融合！")
        return

    # 归一化：确保权重之和为 1
    total_weight = sum(active_weights)
    norm_weights = [w / total_weight for w in active_weights]
    
    print("\n>>> 最终使用的融合权重:")
    for name, w in zip(active_models, norm_weights):
        print(f"  {name}: {w:.4f}")

    # --- 3. 执行加权平均 ---
    final_pred = np.zeros(len(base_df))
    
    for name, w in zip(active_models, norm_weights):
        final_pred += all_preds[name].values * w
        
    # --- 4. 保存结果 ---
    submission = pd.DataFrame({
        'SK_ID_CURR': base_df['SK_ID_CURR'],
        'TARGET': final_pred
    })
    
    output_filename = 'submission_ensemble_final.csv'
    submission.to_csv(output_filename, index=False)
    print(f"\n>>> 融合完成！文件已保存为: {output_filename}")
    print(">>> 祝你在 Leaderboard 上取得好成绩！")

if __name__ == "__main__":
    main()
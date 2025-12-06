import pandas as pd
import numpy as np
import os

def main():
    print(">>> [Rank Blending] 终极融合程序...")
    
    # 读取所有模型结果
    # 记得把你的 LGBM_Ultra 结果也放进来
    files = {
        'submission_lgbm_ultra.csv': 0.60, # 依然是主力
        'submission_xgboost.csv': 0.20,
        'submission_mlp.csv': 0.10,       # 神经网络
        'submission_knn.csv': 0.05,       # KNN
        'submission_logreg.csv': 0.05     # 线性基准
    }
    
    dfs = []
    weights = []
    
    # 读取基准 ID
    base_df = pd.read_csv('submission_lgbm_ultra.csv')
    
    for filename, weight in files.items():
        if os.path.exists(filename):
            print(f"  加载模型: {filename} (权重: {weight})")
            df = pd.read_csv(filename)
            
            # Rank 归一化核心代码
            # 将概率转换为 0-1 之间的排名值 (Rank Probability)
            # 例如：概率最高的样本，Rank值就是 1.0；概率最低的是 0.0
            rank_pred = df['TARGET'].rank() / len(df)
            
            dfs.append(rank_pred)
            weights.append(weight)
        else:
            print(f"  [Warning] 文件 {filename} 不存在，跳过。")
            
    if not dfs:
        print("没有文件，退出。")
        return

    # 重新归一化权重
    weights = [w / sum(weights) for w in weights]
    
    print(">>> 开始计算加权 Rank 平均...")
    final_rank_pred = np.zeros(len(base_df))
    
    for rank_pred, w in zip(dfs, weights):
        final_rank_pred += rank_pred * w
        
    # 注意：Rank 平均后的结果是均匀分布的，如果直接提交 AUC 没问题，
    # 但如果你想看着像概率，可以做一个简单的 MinMax 缩放，但这不影响 AUC 排名
    
    submission = pd.DataFrame({
        'SK_ID_CURR': base_df['SK_ID_CURR'],
        'TARGET': final_rank_pred
    })
    
    submission.to_csv('submission_rank_ensemble.csv', index=False)
    print(">>> 融合完成！结果已保存为 submission_rank_ensemble.csv")
    print(">>> 这个结果集合了 树模型(LGBM/XGB) + 线性模型(LR) + 神经网络(MLP) + 距离模型(KNN)")
    print(">>> 这就是你冲击排行榜前列的最终武器！")

if __name__ == "__main__":
    main()
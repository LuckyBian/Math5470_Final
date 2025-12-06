import pandas as pd
import numpy as np
import os

def main():
    print("Rank Blending...")
    
    files = {
        'submission_lgbm_ultra.csv': 0.60,
        'submission_xgboost.csv': 0.20,
        'submission_mlp.csv': 0.10,
        'submission_knn.csv': 0.05,
        'submission_logreg.csv': 0.05
    }
    
    dfs = []
    weights = []
    
    if not os.path.exists('submission_lgbm_ultra.csv'):
        print("Base file 'submission_lgbm_ultra.csv' not found.")
        return

    base_df = pd.read_csv('submission_lgbm_ultra.csv')
    
    for filename, weight in files.items():
        if os.path.exists(filename):
            print(f"  Loading: {filename} (weight: {weight})")
            df = pd.read_csv(filename)
            
            rank_pred = df['TARGET'].rank() / len(df)
            
            dfs.append(rank_pred)
            weights.append(weight)
        else:
            print(f"  File {filename} not found, skipping.")
            
    if not dfs:
        print("No files found.")
        return

    weights = [w / sum(weights) for w in weights]
    
    print("Calculating weighted rank average...")
    final_rank_pred = np.zeros(len(base_df))
    
    for rank_pred, w in zip(dfs, weights):
        final_rank_pred += rank_pred * w
        
    submission = pd.DataFrame({
        'SK_ID_CURR': base_df['SK_ID_CURR'],
        'TARGET': final_rank_pred
    })
    
    submission.to_csv('submission_rank_ensemble.csv', index=False)
    print("Done! Saved to submission_rank_ensemble.csv")

if __name__ == "__main__":
    main()

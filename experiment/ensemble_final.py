import pandas as pd
import numpy as np
import os

def main():
    print("Reading model predictions...")
    
    model_configs = [
        ('submission_lgbm_ultra.csv',             0.65,   'LGBM_Ultra'),
        ('submission_lgbm_pro.csv',               0.00,   'LGBM_Pro'),
        ('submission_xgboost.csv',                0.30,   'XGBoost'),
        ('submission_logreg.csv',                 0.05,   'LogReg'),
        ('submission_rf.csv',                     0.05,   'RandomForest'),
        ('submission_catboost.csv',               0.00,   'CatBoost')
    ]

    dfs = []
    valid_weights = []
    model_names = []
    
    base_df = None
    
    for filename, weight, name in model_configs:
        if os.path.exists(filename):
            print(f"  Found model: {name} (weight: {weight})")
            df = pd.read_csv(filename)
            df = df.sort_values('SK_ID_CURR')
            
            if base_df is None:
                base_df = df[['SK_ID_CURR']].copy()
            
            if not df['SK_ID_CURR'].equals(base_df['SK_ID_CURR']):
                print(f"  Warning: {name} SK_ID_CURR mismatch, merging...")
                df = pd.merge(base_df, df, on='SK_ID_CURR', how='left')
            
            dfs.append(df['TARGET'])
            if weight > 0:
                valid_weights.append(weight)
                model_names.append(name)
        else:
            print(f"  Model not found: {name} (skipping)")

    if not dfs:
        print("Error: No submission files found!")
        return

    all_preds = pd.concat(dfs, axis=1)
    existing_models = [m[2] for m in model_configs if os.path.exists(m[0])]
    all_preds.columns = existing_models
    
    print("\nCorrelation Matrix:")
    corr = all_preds.corr()
    print(corr)
    
    active_models = [name for name, weight in zip([m[2] for m in model_configs if os.path.exists(m[0])], 
                                                  [m[1] for m in model_configs if os.path.exists(m[0])]) 
                     if weight > 0]
    
    active_weights = [weight for name, weight in zip([m[2] for m in model_configs if os.path.exists(m[0])], 
                                                     [m[1] for m in model_configs if os.path.exists(m[0])]) 
                      if weight > 0]
    
    if not active_weights:
        print("No active models with weight > 0!")
        return

    total_weight = sum(active_weights)
    norm_weights = [w / total_weight for w in active_weights]
    
    print("\nFinal Weights:")
    for name, w in zip(active_models, norm_weights):
        print(f"  {name}: {w:.4f}")

    final_pred = np.zeros(len(base_df))
    
    for name, w in zip(active_models, norm_weights):
        final_pred += all_preds[name].values * w
        
    submission = pd.DataFrame({
        'SK_ID_CURR': base_df['SK_ID_CURR'],
        'TARGET': final_pred
    })
    
    output_filename = 'submission_ensemble_final.csv'
    submission.to_csv(output_filename, index=False)
    print(f"\nEnsemble finished! Saved to: {output_filename}")

if __name__ == "__main__":
    main()

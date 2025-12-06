import pandas as pd

lgbm = pd.read_csv('submission_lgbm_v2.csv')
mlp = pd.read_csv('submission_mlp_v2.csv')

blend_pred = 0.9 * lgbm['TARGET'] + 0.1 * mlp['TARGET']

submission = pd.DataFrame({'SK_ID_CURR': lgbm['SK_ID_CURR'], 'TARGET': blend_pred})
submission.to_csv('submission_final_0.80.csv', index=False)

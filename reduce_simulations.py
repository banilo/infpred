import numpy as np
import pandas as pd
df = pd.read_pickle('./simulations_may.gzip')

df_out = pd.DataFrame()

df.model_violation[df.model_violation.isnull()] = 'None'
pvals = np.array([x for x in df['lr_pvalues'].values])
pvals[pvals == np.finfo(pvals.dtype).eps] = 2.2250738585072014e-308
pvals[pvals < 1e-300] = np.nan

scores = np.array([x for x in df['scores_debiased'].values])

scores_ = scores.max(-1)
scores_[scores_ < 0] = 0

df_out['scores'] = scores_

df_out['pvals_min'] = pvals.min(1)

df_out['pvals_mean'] = pvals.mean(1)

copy_keys = [
    'corr_strength',
    'model_violation', 'n_corr_feat', 'n_feat', 'n_feat_relevant',
    'n_samples', 'noise', 'sim_id']

df_out[copy_keys] = df[copy_keys]

df_out.to_pickle('simulations_summary.gzip')

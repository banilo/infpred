import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_pickle('./simulations.gzip')

pvals = [row['lr_pvalues'].min() for ii, row in df.iterrows()]

scores = [row['scores_debiased'].max() for ii, row in df.iterrows()]
scores = np.array(scores)
scores[scores < 0] = 0


all_scores = np.concatenate(
    [row['scores_debiased'] for ii, row in df.iterrows()])

sns.set_style('ticks')
plt.close('all')
fig = plt.figure(figsize=(8, 6))
hb = plt.hexbin(-np.log10(pvals), scores, gridsize=200,
                norm=plt.matplotlib.colors.Normalize(),
                bins='log', cmap='viridis')
ax = plt.gca()
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log10(n)', fontsize=14, fontweight=100)
sns.despine(trim=True)

plt.xlabel(r'significance [$-log_{10}(p)$]', fontsize=20, fontweight=150)
plt.ylabel(r'prediction [$R^2$]', fontsize=20, fontweight=150)

sns.set_style('ticks')
plt.close('all')
fig = plt.figure(figsize=(8, 6))
hb = plt.hexbin(pvals, scores, gridsize=40,
                bins='log', cmap='viridis')
ax = plt.gca()
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log10(n)', fontsize=14, fontweight=100)
sns.despine(trim=True)

plt.xlabel(r'significance [$p$]', fontsize=20, fontweight=150)
plt.ylabel(r'prediction [$R^2$]', fontsize=20, fontweight=150)


plt.close('all')
plt.figure(figsize=(8, 6))
plt.scatter(pvals, scores)
plt.ylim(0, 1)
plt.xlim(0.05, 0.2)
plt.xlabel(r'significance [$p$]', fontsize=20, fontweight=150)
plt.ylabel(r'prediction [$R^2$]', fontsize=20, fontweight=150)
sns.despine(trim=True)

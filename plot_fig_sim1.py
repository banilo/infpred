# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Danilo Bzdok <danilobzdok@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable


df = pd.read_pickle('./simulations.gzip')

df.model_violation[df.model_violation.isnull()] = 'None'

pvals = np.array([x for x in df['lr_pvalues'].values])

scores = np.array([x for x in df['scores_debiased'].values])

scores_ = scores.max(-1)
scores_[scores_ < 0] = 0

pvals_ = pvals.min(1)

sns.set_style('ticks')

plt.close('all')

fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
ax = axes[0]
x, y = -np.log10(pvals_), scores_
color = df.n_feat_relevant.values[:]

# arbitrary size mapping
size = (np.log10(df.n_samples.values) ** 3) * 2

hb = ax.hexbin(x, y, gridsize=50,
               norm=plt.matplotlib.colors.Normalize(0, 100),
               # norm=plt.matplotlib.colors.LogNorm(),
               # bins='log',
               vmin=10,
               mincnt=1,
               cmap='viridis')

ax.grid(True)
ax.axvline(-np.log10(0.05), color='red', linestyle='--')

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='3%', pad=0.05)

cb = fig.colorbar(hb, cax=cax, orientation='vertical')
cb.set_alpha(1)
cb.draw_all()

cb.set_label('# simulations ', fontsize=14, fontweight=100)

sns.despine(trim=True, ax=ax)
ax.set_xlabel(r'significance [$-log_{10}(p)$]', fontsize=20, fontweight=150)
ax.set_ylabel(r'prediction [$R^2$]', fontsize=20, fontweight=150)

ax_inset = axes[1]
ax_inset.grid(True)

scat = ax_inset.scatter(
    x, y, c='black', s=size,
    edgecolors='face',
    cmap=plt.get_cmap('viridis', len(np.unique(color))),
    vmin=0.2, vmax=color.max(),
    alpha=0.2)

ax_inset.set_xlim(*-np.log10([0.4, 0.02]))
ax_inset.set_xticks(
    -np.log10([0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.01, 0.001]))
ax_inset.set_ylim(0, 1)
ax_inset.axvline(-np.log10(0.05), color='red', linestyle='--')

pvals_orig = (10 ** (-1 * ax_inset.get_xticks())).round(3)
ax_inset.set_xticklabels(pvals_orig)
ax_inset.set_xlabel(r'significance [$p$-value]', fontsize=20, fontweight=150)
ax_inset.set_ylabel(r'prediction [$R^2$]', fontsize=20, fontweight=150)

ax.annotate(
    'A', xy=(-0.15, 0.95), fontweight=200, fontsize=30,
    xycoords='axes fraction')
ax_inset.annotate(
    'B', xy=(-0.15, 0.95), fontweight=200, fontsize=30,
    xycoords='axes fraction')

sns.despine(trim=True, ax=ax_inset)
plt.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.14, wspace=0.37)

fig.savefig('./figures/simulations_overview_fig1.png', bbox_inches='tight',
            dpi=300)

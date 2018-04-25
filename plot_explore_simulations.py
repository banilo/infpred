import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid.inset_locator import inset_axes


df = pd.read_pickle('./simulations.gzip')

df.model_violation[df.model_violation.isnull()] = 'None'

pvals = np.array([x for x in df['lr_pvalues'].values])

scores = np.array([x for x in df['scores_debiased'].values])

scores_ = scores.max(-1)
scores_[scores_ < 0] = 0

pvals_ = pvals.min(1)

sns.set_style('ticks')

plt.close('all')

fig = plt.figure(figsize=(8, 6))
ax = plt.gca()
x, y = -np.log10(pvals_), scores_
color = df.n_feat_relevant.values[:]
size = np.sqrt(df.n_samples.values)

# scat = ax.scatter(x, y, c=color, s=size,
#                   edgecolors='face',
#                   cmap=plt.get_cmap('viridis', len(np.unique(color))),
#                   vmin=0.2, vmax=color.max(),
#                   alpha=0.2)

hb = ax.hexbin(x, y, gridsize=0,
               # norm=plt.matplotlib.colors.Normalize(0, 500),
               # norm=plt.matplotlib.colors.LogNorm(),
               # bins='log',
               cmap='viridis')

ax.grid(True)
ax.axvline(-np.log10(0.05), color='red', linestyle='--')
cb = fig.colorbar(hb)
cb.set_alpha(1)
cb.draw_all()

cb.set_label('# simulations ', fontsize=14, fontweight=100)

sns.despine(trim=True, ax=ax)
plt.xlabel(r'significance [$-log_{10}(p)$]', fontsize=20, fontweight=150)
plt.ylabel(r'prediction [$R^2$]', fontsize=20, fontweight=150)

plt.subplots_adjust()

ax_inset = plt.axes([.5, .2, .2, .2])
scat = ax_inset.scatter(
    x, y, c=color, s=size,
    edgecolors='face',
    cmap=plt.get_cmap('viridis', len(np.unique(color))),
    vmin=0.2, vmax=color.max(),
    alpha=0.2)
ax_inset.set_xlim(*-np.log10([0.5, 0.01]))
ax_inset.set_ylim(0, 1)
ax_inset.axvline(-np.log10(0.05), color='red', linestyle='--')
ax_inset.set_xlabel(r'$-log_{10}(p)$', fontsize=10, fontweight=150)
ax_inset.set_ylabel(r'$R^2$', fontsize=10, fontweight=150)

fig.savefig('./figures/simulations_overview.tiff', bbox_inches='tight', dpi=300)


pathologies = [
    'None', 'abs', 'log', 'exp', 'sqrt', '1/x', 'x^2', 'x^3', 'x^4',
    'x^5']

plt.close('all')
fig, axes = plt.subplots(2, 5, figsize=(10, 5), sharey=True)

n_samp = df.n_samples
axes = axes.ravel()
for ii, path in enumerate(pathologies):
    ax = axes[ii]
    inds = np.where(df.pathology == path)
    x, y = -np.log10(pvals_[inds]), scores_[inds]

    color = df.n_feat_relevant.values[inds]
    size = df.n_samples.values * 0.05
    scat = ax.scatter(x, y, c=color, s=size,
                      edgecolors='face',
                      cmap=plt.get_cmap('viridis', len(np.unique(color))),
                      vmin=0.2, vmax=color.max(),
                      alpha=0.2)

    # ax.set_facecolor('')
    ax.grid(True)
    # hb = ax.hexbin(x[inds], y[inds], gridsize=30,
    #                norm=plt.matplotlib.colors.Normalize(0, 50),
    #                cmap='viridis')
    # cb = fig.colorbar(hb, ax=ax)
    # cb.set_label('n simulations', fontsize=14, fontweight=100)
    ax.set_xlim(-1, 305)
    ax.set_xticks([0, 100, 200, 300])
    # ax.set_facecolor(plt.matplotlib.cm.viridis(0))
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.8, 1.0])
    ax.set_title(path)
    ax.axvline(-np.log10(0.05), color='red', linestyle='--')

    ax_inset = inset_axes(
        ax, width="30%",  # width = 30% of parent_bbox
        height="40%",  # height : 1 inch
        # bbox_to_anchor=(0.5, .1),
        # bbox_transform=ax.transAxes,
        borderpad=2,
        loc=4)
    scat = ax_inset.scatter(x, y, c=color, s=size,
                            edgecolors='face',
                            cmap=plt.get_cmap('viridis', len(np.unique(color))),
                            vmin=0.2, vmax=color.max(),
                            alpha=0.2)
    ax_inset.set_xlim(*-np.log10([0.5, 0.01]))
    ax_inset.set_ylim(-0.05, 1.05)
    ax_inset.axvline(-np.log10(0.05), color='red', linestyle='--')
    # ax_inset.set_xlabel(r'$-log_{10}(p)$', fontsize=10, fontweight=150)
    # ax_inset.set_ylabel(r'$R^2$', fontsize=10, fontweight=150)
    sns.despine(trim=True, ax=ax)

    if ii > 4:
        ax.set_xlabel(r'$-log_{10}(p)$', fontsize=10, fontweight=150)
    if ii in (0, 5):
        ax.set_ylabel(r'prediction [$R^2$]',
                      fontsize=15, fontweight=150)

plt.subplots_adjust(hspace=0.33, left=.1, right=.97, top=.94, bottom=.10)
fig.savefig('./figures/simulations_by_violation.tiff', bbox_inches='tight', dpi=600)

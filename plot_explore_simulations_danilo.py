import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid.inset_locator import inset_axes


# df = pd.read_hdf('./simulations.h5')
df = pd.read_pickle('./simulations100k.gzip')
# df = pd.read_pickle('./simulations.gzip')

if 'pathology' in df.columns:
    df.pathology[df.pathology.isnull()] = 'None'
else:
    df.model_violation[df.model_violation.isnull()] = 'None'
df.corr_strength.loc[np.isnan(df.corr_strength)] = 'None'



pvals = np.array([x for x in df['lr_pvalues'].values])

scores = np.array([x for x in df['scores_debiased'].values])


for pval_opt in ['mean', 'median', 'min']:
	scores_ = scores.max(-1)
	scores_[scores_ < 0] = 0

	if pval_opt == 'mean':
		pvals_ = pvals.mean(axis=1)
	elif pval_opt == 'median':
		pvals_ = np.median(pvals, axis=1)
	elif pval_opt == 'min':
		pvals_ = pvals.min(axis=1)
	else:
		raise('Ups')

	sns.set_style('ticks')

	plt.close('all')

	fig = plt.figure(figsize=(8, 6))
	ax = plt.gca()
	x, y = -np.log10(pvals_), scores_
	color = df.n_feat_relevant.values[:]
	size = df.n_samples.values * 0.05
	scat = ax.scatter(x, y, c=color, s=size,
	                  edgecolors='face',
	                  cmap=plt.get_cmap('viridis', len(np.unique(color))),
	                  vmin=0.2, vmax=color.max(),
	                  alpha=0.2)
    ax.set_xlim(0, 50)
	# ax.set_xlim(0, 200)
	ax.grid(True)
	ax.axvline(-np.log10(0.05), color='red', linestyle='--')
	cb = fig.colorbar(scat)
	cb.set_alpha(1)
	cb.draw_all()

	cb.set_label('# relevant features ', fontsize=14, fontweight=100)

	sns.despine(trim=True, ax=ax)
	# plt.xlabel(r'significance [$-log_{10}(p)$]', fontsize=20, fontweight=150)
	plt.xlabel(r'significance [-log_10(p)]', fontsize=20, fontweight=150)
	plt.ylabel(r'prediction [R^2]', fontsize=20, fontweight=150)

	plt.subplots_adjust()

	ax_inset = plt.axes([.5, .2, .2, .2])
	scat = ax_inset.scatter(
	    x, y, c=color, s=size,
	    edgecolors='face',
	    cmap=plt.get_cmap('viridis', len(np.unique(color))),
	    vmin=0.2, vmax=color.max(),
	    alpha=0.2)
	ax_inset.set_xlim(-np.log10([0.5, 0.01]))
	ax_inset.set_ylim(0, 1)
	ax_inset.axvline(-np.log10(0.05), color='red', linestyle='--')
	# ax_inset.set_xlabel(r'$-log_{10}(p)$', fontsize=10, fontweight=150)
	# ax_inset.set_ylabel(r'$R^2$', fontsize=10, fontweight=150)
	fig.savefig('./figures/simulations_overview_%s.tiff' % pval_opt,
			bbox_inches='tight', dpi=600)


STOP


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

################################################################################


# pathological versus non-pathological
plt.close('all')
scores_ = scores.max(-1)
pvals_ = pvals.min(axis=1)

fig, axes = plt.subplots(1, 1, figsize=(8, 6), sharey=True)

n_samp = df.n_samples

ax = axes
if "pathology" in df.columns:
    inds_path = np.where(df.pathology != 'None')
    inds_nopath = np.where(df.pathology == 'None')
else:
    inds_path = np.where(df.model_violation != 'None')
    inds_nopath = np.where(df.model_violation == 'None')

for color, cur_inds in zip(['red', 'blue'], [inds_path, inds_nopath]):
    x, y = -np.log10(pvals_[cur_inds]), scores_[cur_inds]
    size = df.n_samples.values[cur_inds] * 0.05
    # scat = ax.plot(x, y, c=color, s=size,
    #                   # edgecolors='face',
    #                   # cmap=plt.get_cmap('viridis', len(np.unique(color))),
    #                   # vmin=0.2, vmax=color.max(),
    #                   alpha=0.2)
    scat = ax.scatter(x, y, c=color, s=size,
                      # edgecolors='face',
                      # cmap=plt.get_cmap('viridis', len(np.unique(color))),
                      # vmin=0.2, vmax=color.max(),
                      alpha=0.2)

# ax.set_facecolor('')
ax.grid(True)
# hb = ax.hexbin(x[inds], y[inds], gridsize=30,
#                norm=plt.matplotlib.colors.Normalize(0, 50),
#                cmap='viridis')
# cb = fig.colorbar(hb, ax=ax)
# cb.set_label('n simulations', fontsize=14, fontweight=100)
# ax.set_xlim(-1, 305)
# ax.set_xlim(-1, 50)
ax.set_xlim(0, 50)
ax.set_xticks(np.arange(0, 51, 10))
# ax.set_xticks([0, 100, 200, 300])
# ax.set_facecolor(plt.matplotlib.cm.viridis(0))
ax.set_ylim(-0.05, 1.05)
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.8, 1.0])
ax.set_title('Data generated by true model or not')
# ax.axvline(-np.log10(0.05), color='red', linestyle='--')

# ax_inset = inset_axes(
#     ax, width="30%",  # width = 30% of parent_bbox
#     height="40%",  # height : 1 inch
#     # bbox_to_anchor=(0.5, .1),
#     # bbox_transform=ax.transAxes,
#     borderpad=2,
#     loc=4)
# scat = ax_inset.scatter(x, y, c=color, s=size,
#                         edgecolors='face',
#                         cmap=plt.get_cmap('viridis', len(np.unique(color))),
#                         vmin=0.2, vmax=color.max(),
#                         alpha=0.2)
# ax_inset.set_xlim(*-np.log10([0.5, 0.01]))
# ax_inset.set_ylim(-0.05, 1.05)
# ax_inset.axvline(-np.log10(0.05), color='red', linestyle='--')
# ax_inset.set_xlabel(r'$-log_{10}(p)$', fontsize=10, fontweight=150)
# # ax_inset.set_ylabel(r'$R^2$', fontsize=10, fontweight=150)
# sns.despine(trim=True, ax=ax)

# if ii > 4:
#     ax.set_xlabel(r'$-log_{10}(p)$', fontsize=10, fontweight=150)
# if ii in (0, 5):
#     ax.set_ylabel(r'prediction [$R^2$]',
#                   fontsize=15, fontweight=150)

plt.subplots_adjust(hspace=0.33, left=.1, right=.97, top=.94, bottom=.10)
fig.savefig('./figures/simulations100k_path.tiff', bbox_inches='tight', dpi=600)



# polynomial transformation degrees
plt.close('all')
scores_ = scores.max(-1)
pvals_ = pvals.min(axis=1)

fig, axes = plt.subplots(1, 1, figsize=(8, 6), sharey=True)

n_samp = df.n_samples

ax = axes
poly_tr_types = ['None', 'x^2', 'x^3', 'x^4', 'x^5']
inds = np.where(df.model_violation.isin(poly_tr_types))[0]

color = df.model_violation.replace(
    poly_tr_types, np.arange(len(poly_tr_types))).values[inds]
# for cur_noise in noise_types:
# cur_inds = np.where(df.noise == cur_noise)[0]
x, y = -np.log10(pvals_[inds]), scores_[inds]
size = df.n_samples.values * 0.05
# scat = ax.plot(x, y, c=color, s=size,
#                   # edgecolors='face',
#                   # cmap=plt.get_cmap('viridis', len(np.unique(color))),
#                   # vmin=0.2, vmax=color.max(),
#                   alpha=0.2)
scat = ax.scatter(x, y, c=color, s=size,
                  # edgecolors='face',
                  cmap=plt.get_cmap('viridis', len(poly_tr_types)),
                  # vmin=0.2, vmax=color.max(),
                  alpha=0.2)

# ax.set_facecolor('')
ax.grid(True)
cb = fig.colorbar(scat)
cb.set_alpha(1)
cb.draw_all()

cb.set_label('order of polynomial transformation', fontsize=14, fontweight=100)
ax.set_xlim(0, 50)
ax.set_xticks(np.arange(0, 51, 10))
# ax.set_xticks([0, 100, 200, 300])
# ax.set_facecolor(plt.matplotlib.cm.viridis(0))
ax.set_ylim(-0.05, 1.05)
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.8, 1.0])
ax.set_title('Data polynomially transformed or not')
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

plt.subplots_adjust(hspace=0.33, left=.1, right=.97, top=.94, bottom=.10)
fig.savefig('./figures/simulations100k_poly.tiff', bbox_inches='tight', dpi=600)



# multicollinearity degrees
plt.close('all')
scores_ = scores.max(-1)
pvals_ = pvals.min(axis=1)

df.corr_strength.replace('None', 0, inplace=True)
fig, axes = plt.subplots(1, 1, figsize=(8, 6), sharey=True)

n_samp = df.n_samples

ax = axes
collin_types = np.unique(df.corr_strength)

color = df.corr_strength
# for cur_noise in noise_types:
# cur_inds = np.where(df.noise == cur_noise)[0]
x, y = -np.log10(pvals_), scores_
size = df.n_samples.values * 0.05
# scat = ax.plot(x, y, c=color, s=size,
#                   # edgecolors='face',
#                   # cmap=plt.get_cmap('viridis', len(np.unique(color))),
#                   # vmin=0.2, vmax=color.max(),
#                   alpha=0.2)
scat = ax.scatter(x, y, c=color, s=size,
                  # edgecolors='face',
                  cmap=plt.get_cmap('viridis', len(collin_types)),
                  # vmin=0.2, vmax=color.max(),
                  alpha=0.2)

# ax.set_facecolor('')
ax.grid(True)
cb = fig.colorbar(scat, ticks=collin_types)
cb.set_alpha(1)
cb.draw_all()

cb.set_label('order of collinearity between relevant fatures',
             fontsize=14, fontweight=100)
ax.set_xlim(0, 50)
ax.set_xticks(np.arange(0, 51, 10))
# ax.set_xticks([0, 100, 200, 300])
# ax.set_facecolor(plt.matplotlib.cm.viridis(0))
ax.set_ylim(-0.05, 1.05)
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.8, 1.0])
ax.set_title('Data are correlated with each other or not')
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

plt.subplots_adjust(hspace=0.33, left=.1, right=.97, top=.94, bottom=.10)
fig.savefig('./figures/simulations100k_collin.tiff', bbox_inches='tight', dpi=600)



# noise degrees
plt.close('all')
scores_ = scores.max(-1)
pvals_ = pvals.min(axis=1)

fig, axes = plt.subplots(1, 1, figsize=(8, 6), sharey=True)

n_samp = df.n_samples

ax = axes
noise_types = np.unique(df.noise)

color = df.noise.values.copy()
# for cur_noise in noise_types:
# cur_inds = np.where(df.noise == cur_noise)[0]
x, y = -np.log10(pvals_), scores_
size = df.n_samples.values * 0.05
# scat = ax.plot(x, y, c=color, s=size,
#                   # edgecolors='face',
#                   # cmap=plt.get_cmap('viridis', len(np.unique(color))),
#                   # vmin=0.2, vmax=color.max(),
#                   alpha=0.2)
scat = ax.scatter(x, y, c=color, s=size,
                  # edgecolors='face',
                  cmap=plt.get_cmap('viridis', len(noise_types)),
                  # vmin=0.2, vmax=color.max(),
                  alpha=0.2)

# ax.set_facecolor('')
ax.grid(True)
cb = fig.colorbar(scat)
cb.set_alpha(1)
cb.draw_all()

cb.set_label('amount of added noise', fontsize=14, fontweight=100)
ax.set_xlim(0, 50)
ax.set_xticks(np.arange(0, 51, 10))
# ax.set_xticks([0, 100, 200, 300])
# ax.set_facecolor(plt.matplotlib.cm.viridis(0))
ax.set_ylim(-0.05, 1.05)
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.8, 1.0])
ax.set_title('Data generated with noise or not')
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

plt.subplots_adjust(hspace=0.33, left=.1, right=.97, top=.94, bottom=.10)
fig.savefig('./figures/simulations100k_noise.tiff', bbox_inches='tight', dpi=600)





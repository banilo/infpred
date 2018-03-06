import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import ShuffleSplit
from sklearn.linear_model import Lasso
import seaborn as sns
from matplotlib import pylab as plt


n_feat = 40
my_rs = np.random.RandomState(42)
X, y, true_coef = make_regression(n_samples=100, n_features=n_feat, n_informative=10,
	                              noise=1.0, shuffle=False, coef=True,
	                              effective_rank=40, random_state=my_rs)

scaler = StandardScaler()
X_ss = scaler.fit_transform(X)


lr = LinearRegression()
lr.fit(X_ss, y)


plt.scatter(np.arange(X.shape[1]), true_coef, color='black', label='true coefs',
	        alpha=0.5)
plt.scatter(np.arange(X.shape[1]), lr.coef_, color='red', label='estimated coefs (LR)',
	        alpha=0.5)
plt.xlabel('input variables')
plt.xticks(np.arange(n_feat), np.arange(n_feat) + 1)
plt.legend(loc='upper right')


# regularization paths
n_verticals = 16
n_verticals = 25
C_grid = np.logspace(-8, 1, n_verticals)
coef_list2 = []
acc_list2 = []
for i_step, my_C in enumerate(C_grid):
    sample_accs = []
    sample_coef = []
    for i_subsample in range(100):
        folder = ShuffleSplit(n=len(y), n_iter=100, test_size=0.1,
                                        random_state=i_subsample)
        train_inds, test_inds = iter(folder).next()

        clf = Lasso(alpha=my_C)

        clf.fit(X_ss[train_inds, :], y[train_inds])
        acc = clf.score(X_ss[test_inds, :], y[test_inds])

        sample_accs.append(acc)        
        sample_coef.append(clf.coef_)

	mean_coefs = np.mean(np.array(sample_coef), axis=0)
    coef_list2.append(mean_coefs)
    acc_list2.append(np.mean(sample_accs))
    print "alpha: %.4f acc: %.2f active_coefs: %i" % (
    	my_C, acc, np.count_nonzero(mean_coefs))

coef_list2 = np.array(coef_list2)
acc_list2 = np.array(acc_list2)

# plot paths
n_cols = 2
n_rows = 1

my_palette = np.array([
    # '#4BBCF6',
    '#F47D7D', '#FBEF69', '#98E466', '#000000',
    '#A7794F', '#CCCCCC', '#85359C', '#FF9300', '#FF0030'
])
my_colors = np.array(['???????'] * coef_list2.shape[-1])
i_col = 0
new_grp_pts_x = []
new_grp_pts_y = []
new_grp_pts_col = []
new_grp_pts_total = []

for i_vertical, (params, acc, C) in enumerate(zip(
    coef_list2, acc_list2, C_grid)):
    b_notset = my_colors == '???????'
    b_nonzeros = params != 0
    b_coefs_of_new_grp = np.logical_and(b_notset, b_nonzeros)
    
    if np.sum(b_coefs_of_new_grp) > 0:
        # import pdb; pdb.set_trace()
        # we found a new subset that became 0
        for new_i in np.where(b_coefs_of_new_grp == True)[0]:
            # color all coefficients of the current group
            cur_col = my_palette[i_col]
            my_colors[new_i] = cur_col
            
        new_grp_pts_x.append(C)
        new_grp_pts_y.append(acc)
        new_grp_pts_col.append(cur_col)
        new_grp_pts_total.append(np.sum(b_nonzeros))
        i_col += 1
    

# plt.figure()
# for i_line in range(X.shape[-1]):
#     plt.plot(np.log10(C_grid),
#         coef_list2[:, i_line], label=X_colnames[i_line],
#             color=my_colors[i_line], linewidth=1.5)

# axarr[0].set_xticks(np.arange(len(C_grid)))
# axarr[0].set_xticklabels(np.log10(C_grid), rotation=75)
# axarr[i_col].set_xlabel('ln(C)')
# axarr[i_col].set_ylabel('Coefficient importance')
# axarr[i_col].legend(loc='lower left', fontsize=7)
# axarr[0].grid(True)

X_colnames = np.arange(n_feat) + 1
subplot_xlabel = 'lambda choices'

f, axarr = plt.subplots(nrows=n_rows, ncols=n_cols,
    figsize=(15, 10), facecolor='white')
t, i_col = 0, 0

for i_line in range(X.shape[-1]):
    axarr[i_col].plot(np.log10(C_grid),
        coef_list2[:, i_line], label=X_colnames[i_line],
            color=my_colors[i_line], linewidth=1.5)

# axarr[0].set_xticks(np.arange(len(C_grid)))
# axarr[0].set_xticklabels(np.log10(C_grid))  #, rotation=75)
axarr[i_col].set_xlabel(subplot_xlabel, fontsize=10)
axarr[i_col].legend(loc='lower left', fontsize=12.6)
axarr[0].grid(True)
# axarr[i_col].set_ylabel('Item groups', fontsize=16)
axarr[0].set_title('Item groups', fontsize=16)
axarr[0].set_xticks([])

# axarr[1].axis('off')
axarr[1].plot(np.arange(len(acc_list2)), acc_list2, color='#000000',
                 linewidth=1.5)
# axarr[1].set_title('ACCURACY')
axarr[1].set_ylim(0.0, 1.00)
axarr[1].grid(True)
# axarr[1].set_xticklabels(np.log10(C_grid), '')
axarr[1].set_xticks([])
axarr[1].set_xlabel(subplot_xlabel, fontsize=10)
# axarr[1].set_ylabel('Out-of-sample accuracy', fontsize=16)
axarr[1].set_title('Out-of-sample accuracy', fontsize=16)

for i_pt, (x, y, col, n_coefs) in enumerate(zip(
    new_grp_pts_x, new_grp_pts_y, new_grp_pts_col, new_grp_pts_total)):
    axarr[1].plot(np.log10(x), y,
                  marker='o', color=col, markersize=10.0)
    axarr[1].text(
        np.log10(x) - 0.95 + 0.07 * i_pt,
        y + 0.003 * i_pt,
        '%i items' % n_coefs)


# plt.savefig('plots/regpath_extended.png', DPI=400, facecolor='white')



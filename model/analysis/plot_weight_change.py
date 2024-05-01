import matplotlib.pyplot as plt
import os
from utils import units_convert, clda_colors
import seaborn as sns
import numpy as np
from scipy.stats import linregress

plt.style.use("rnn4bci_plot_params.dms")

"""How learning under BCI control changes the weights for the readout units
(i.e. participating in BCI control) and nonreadout units (i.e. not participating in BCI control,
but whose weights are also updated through learning."""

plt.rc('font', size=8)          # controls default text sizes
plt.rc('axes', titlesize=8)     # fontsize of the axes title
plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
plt.rc('ytick', labelsize=7)    # fontsize of the tick labels
plt.rc('legend', fontsize=6)    # legend fontsize
plt.rc('figure', titlesize=8)  # fontsize of the figure title
fig_width = 40  # mm

relative = False  # whether to use the relative weights  (verrrry long computation when = True, avoid)

nb_seeds = 1
seeds = list(range(1, 1+nb_seeds))
CLDAs = [1, 0.9, 0.75, 0.5]

data_dir_bci = f"../data/bci-model"
data_dir_manual = f"../data/arm-model"
result_dir = f"../results/bci-model/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

nb_readouts = 12

weight_change = {'W': {CLDA: [] for CLDA in CLDAs},
                 'U': {CLDA: [] for CLDA in CLDAs}}

submatricesW = {CLDA: {'R/R': [], 'R/NR': [], 'NR/R': [], 'NR/NR': []} for CLDA in CLDAs}
submatricesU = {CLDA: {'R': [], 'NR': []} for CLDA in CLDAs}

for CLDA in CLDAs:
    for seed in seeds:
        # Load data
        W_before = np.loadtxt(f"{data_dir_manual}/seed{seed}/manual_trained_network_W.txt", delimiter=',')
        U_before = np.loadtxt(f"{data_dir_manual}/seed{seed}/manual_trained_network_U.txt", delimiter=',')
        W_after = np.loadtxt(f"{data_dir_bci}-clda{CLDA}/seed{seed}/bci_trained_network_W.txt", delimiter=',')
        U_after = np.loadtxt(f"{data_dir_bci}-clda{CLDA}/seed{seed}/bci_trained_network_U.txt", delimiter=',')

        readout_ids = np.arange(nb_readouts, dtype=int)
        try:
            readout_ids = np.loadtxt(f"{data_dir_bci}-clda{CLDA}/seed{seed}/valid_units_pre_swap.txt", dtype=int)
        except FileNotFoundError:
            print()
        non_readout_ids = [i for i in range(W_before.shape[0]) if i not in readout_ids]

        #n_training_days = np.loadtxt(f"{data_dir_bci}-clda{CLDA}/seed{seed}/number_of_training_days.txt", dtype=int)
        #n_training_days_no_CLDA = np.loadtxt(f"{data_dir_bci}-clda1/seed{seed}/number_of_training_days.txt", dtype=int)

        #if n_training_days < n_training_days_no_CLDA:
        if relative:
            weight_change['W'][CLDA].append((W_after - W_before)/(W_before+1e-8))
            weight_change['U'][CLDA].append((U_after - U_before)/(U_before+1e-8))
        else:
            weight_change['W'][CLDA].append(W_after - W_before)
            weight_change['U'][CLDA].append(U_after - U_before)

        DeltaW_onto_readouts = weight_change['W'][CLDA][-1][readout_ids, :]
        DeltaW_onto_nonreadouts = weight_change['W'][CLDA][-1][non_readout_ids, :]
        submatricesW[CLDA]['R/R'] += list(DeltaW_onto_readouts[:, readout_ids].ravel())
        submatricesW[CLDA]['R/NR'] += list(DeltaW_onto_readouts[:, non_readout_ids].ravel())
        submatricesW[CLDA]['NR/R'] += list(DeltaW_onto_nonreadouts[:, readout_ids].ravel())
        submatricesW[CLDA]['NR/NR'] += list(DeltaW_onto_nonreadouts[:, non_readout_ids].ravel())

        submatricesU[CLDA]['R'] += list(weight_change['U'][CLDA][-1][readout_ids, :].ravel())
        submatricesU[CLDA]['NR'] += list(weight_change['U'][CLDA][-1][non_readout_ids, :].ravel())

# Plot singular values for W
plt.figure(figsize=(45 * units_convert['mm'], 45 / 1.25 * units_convert['mm']))
for CLDA in CLDAs:
    eigval = []
    for dW in weight_change['W'][CLDA]:
        #s = np.linalg.eigvals(dW)
        _, s, _ = np.linalg.svd(dW)
        eigval.append(s)
    m = np.mean(eigval, axis=0)
    sem = np.std(eigval, axis=0, ddof=1) / len(eigval)**0.5
    plt.errorbar(np.arange(1, 1 + len(m)), m, yerr=sem, color=clda_colors[CLDA], label=f'CLDA = {1.-CLDA:.2}')
plt.xlim([1-0.1, 10+0.1])
plt.xticks([1, 10])
plt.yticks([0, 0.05])
plt.gca().set_yticklabels([0, 0.05])
plt.xlabel('Rank')
plt.ylabel('Singular value of $\Delta W$')
plt.legend(fontsize=5)
sns.despine()
plt.tight_layout()
plt.savefig(result_dir + f"SingularValuesWChange.png")
plt.close()

# Plot weight distribution
for matrix in ['W', 'U']:
    fig, axes = plt.subplots(ncols=len(CLDAs[1:]),
                             figsize=(len(CLDAs[1:])*fig_width*units_convert['mm'] / 1.25, fig_width * units_convert['mm']),
                             sharex=True, sharey=True)
    for p, CLDA in enumerate(CLDAs[1:]):
        all_W_changes_with_CLDA = []
        all_W_changes_no_CLDA = []
        for i, seed in enumerate(seeds):
            all_W_changes_no_CLDA.append(weight_change[matrix][1][i].ravel())
            all_W_changes_with_CLDA.append(weight_change[matrix][CLDA][i].ravel())
            axes[p].plot(weight_change[matrix][1][i].ravel(), weight_change[matrix][CLDA][i].ravel(), marker='.',
                         markersize=2, lw=0, alpha=0.3, mec='white', mfc=clda_colors[CLDA], mew=0.1)
        all_W_changes_with_CLDA = np.hstack(all_W_changes_with_CLDA)
        all_W_changes_no_CLDA = np.hstack(all_W_changes_no_CLDA)
        result = linregress(all_W_changes_no_CLDA,  all_W_changes_with_CLDA)
        axes[p].text(1, 0.1, s=f"$R^2$ = {result.rvalue**2:.2}", transform=axes[p].transAxes, fontdict={'fontsize': 7}, ha='right')
        min_ = min(min(axes[p].get_xlim()), min(axes[p].get_ylim()))
        max_ = max(max(axes[p].get_xlim()), max(axes[p].get_ylim()))
        #axes[p].plot([min_, max_], [result.intercept + result.slope*min_, result.intercept + result.slope*max_], '--', color='grey', lw=0.5)
        axes[p].set_title(f'CLDA = {1.-CLDA:.2}', pad=-10)
        axes[p].set_xlim([min_, max_])
    axes[0].set_yticks([-0.005, 0, 0.005])
    for ax in axes:
        ax.set_aspect('equal')
    axes[0].set_ylabel(rf"$\Delta {matrix}$ with CLDA")
    axes[1].set_xlabel(rf"$\Delta {matrix}$ with fixed decoder")
    sns.despine()
    plt.tight_layout()
    plt.savefig(result_dir + f"ScatterChange{matrix}.png")
    plt.close()


"""
# Plot weight distribution for submatricesW
fig_all, ax_all = plt.subplots(figsize=(35*units_convert['mm'], 45/1.25*units_convert['mm']))
fig, axes_tmp = plt.subplots(nrows=2, ncols=2, figsize=(2*45*units_convert['mm'], 2*45/1.25*units_convert['mm']),
                             sharex=True, sharey=True)
axes = {'R/R': axes_tmp[0,0], 'R/NR': axes_tmp[0,1], 'NR/R': axes_tmp[1,0], 'NR/NR': axes_tmp[1,1]}
for decoder_type in ['CLDA', 'Fixed']:
    dW_all = []
    for submatricesW_type in axes.keys():
        axes[submatricesW_type].hist(submatricesW[decoder_type][submatricesW_type], bins='auto', density=True,
                                    color='k' if decoder_type=='Fixed' else 'grey', alpha=0.5,
                                     label=decoder_type if decoder_type=='Fixed' else 'CLDA (pooled)')
        dW_all += submatricesW[decoder_type][submatricesW_type]
    ax_all.hist(dW_all, bins='auto',
                density=True,  color='k' if decoder_type=='Fixed' else 'grey', alpha=0.5,
                label=decoder_type if decoder_type=='Fixed' else 'CLDA (pooled)')
ax_all.set_ylabel("Probability density")
ax_all.set_xlabel(r"Relative recurrent weight change" if relative else "Recurrent weight change")
ax_all.set_xticks([-0.01, 0, 0.01])
ax_all.legend(loc='upper right')
sns.despine(ax=ax_all, trim=True)
fig_all.tight_layout()
fig_all.savefig(f"{result_dir}RelativeChangeW.png" if relative else f"{result_dir}WChange.png")

axes_tmp[0,0].set_ylabel("Probability density")
axes_tmp[1,0].set_ylabel("Probability density")
axes_tmp[1,0].set_xlabel(r"Relative recurrent weight change" if relative else "Recurrent weight change")
axes_tmp[1,1].set_xlabel(r"Relative recurrent weight change" if relative else "Recurrent weight change")
axes_tmp[0,0].legend(loc='upper right')
for k, ax in axes.items():
    if k == 'R/R':
        title = r"readout $\leftarrow$ readout"
    elif k == 'R/NR':
        title = r"readout $\leftarrow$ non-readout"
    elif k == 'NR/R':
        title = r"non-readout $\leftarrow$ readout"
    elif k == 'NR/NR':
        title = r"non-readout $\leftarrow$ non-readout"
    ax.set_title(title, pad=1)
plt.tight_layout()
plt.savefig(f"{result_dir}SubmatricesRelativeChangeW.png" if relative else f"{result_dir}SubmatricesWChange.png")

# Plot weight distribution for submatricesU
fig_all, ax_all = plt.subplots(figsize=(35*units_convert['mm'], 45/1.25*units_convert['mm']))
fig, axes_tmp = plt.subplots(nrows=2, ncols=1, figsize=(45*units_convert['mm'], 2*45/1.25*units_convert['mm']),
                             sharex=True, sharey=True)

axes = {'R': axes_tmp[0], 'NR': axes_tmp[1]}
for decoder_type in ['CLDA', 'Fixed']:
    dU_all = []
    for submatricesU_type in axes.keys():
        axes[submatricesU_type].hist(submatricesU[decoder_type][submatricesU_type], bins='auto', density=True,
                                    color='k' if decoder_type=='Fixed' else 'grey', alpha=0.5,
                                     label=decoder_type if decoder_type=='Fixed' else 'CLDA (pooled)')
        dU_all += submatricesU[decoder_type][submatricesU_type]
    print(f"Proportion of nonzero weight change {decoder_type}", len(np.nonzero(dU_all)[0]) / len(dU_all))
    print(len(dU_all))
    ax_all.hist(dU_all, bins='auto',
                density=True, color='k' if decoder_type == 'Fixed' else 'grey', alpha=0.5,
                label=decoder_type if decoder_type=='Fixed' else 'CLDA (pooled)')
ax_all.set_ylabel("Probability density")
ax_all.set_xlabel(r"Relative input weight change" if relative else "Input weight change")
ax_all.set_xticks([-0.01, 0, 0.01])
ax_all.legend(loc='upper right')
sns.despine(ax=ax_all, trim=True)
fig_all.tight_layout()
fig_all.savefig(f"{result_dir}RelativeChangeU.png" if relative else f"{result_dir}UChange.png")

axes_tmp[0].set_ylabel("Probability density")
axes_tmp[1].set_ylabel("Probability density")
axes_tmp[1].set_xlabel(r"Weight change")
axes_tmp[0].legend(loc='upper right')
for k, ax in axes.items():
    if k == 'R':
        title = r"readout $\leftarrow$"
    elif k == 'NR':
        title = r"non-readout $\leftarrow$"
    ax.set_title(title, pad=1)
plt.tight_layout()
plt.savefig(result_dir + f"SubmatricesUChange.png")
""" """"""
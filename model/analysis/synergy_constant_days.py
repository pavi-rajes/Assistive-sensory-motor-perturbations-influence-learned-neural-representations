import matplotlib.pyplot as plt
import os
from utils import units_convert, clda_colors, pooled_data_color
import seaborn as sns
from scipy.stats import iqr
import numpy as np
plt.style.use("rnn4bci_plot_params.dms")

plt.rc('font', size=8)          # controls default text sizes
plt.rc('axes', titlesize=8)     # fontsize of the axes title
plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
plt.rc('ytick', labelsize=7)    # fontsize of the tick labels
plt.rc('legend', fontsize=6)    # legend fontsize
plt.rc('figure', titlesize=8)  # fontsize of the figure title
fig_width = 40  # mm

n_readout = 12
nb_seeds = 1
seeds = list(range(1, nb_seeds+1))
pool_clda_data = True
use_log = False  # do not set to `True`

data_folder_prefix = f"../data/bci-model-clda"
result_folder_prefix = f"../results/bci-model"
if not os.path.exists(result_folder_prefix):
    os.makedirs(result_folder_prefix)

uacs = {'Median': {'With CLDA': [], 'No CLDA': []},
        'Max': {'With CLDA': [], 'No CLDA': []},
        'Min': {'With CLDA': [], 'No CLDA': []}}

alphas_clda = [1, 0.9, 0.75, 0.5]

synergy = {'Late': {seed: {clda: None for clda in alphas_clda} for seed in seeds}}
synergy_no_clda = {'Late': {'min': [], 'max': []}}

# Load data
for seed in seeds:
    for clda in alphas_clda:
        data_folder = f"{data_folder_prefix}{clda}/seed{seed}"
        s = np.loadtxt(f"{data_folder}/synergy_late_.txt")
        if use_log:
            synergy['Late'][seed][clda] = np.loadtxt(f"{data_folder}/synergy_late_.txt")
            synergy['Late'][seed][clda][:, 1] = np.log10(-synergy['Late'][seed][clda][:, 1])
        else:
            synergy['Late'][seed][clda] = np.loadtxt(f"{data_folder}/synergy_late_.txt")
        if abs(clda - 1) < 1e-6:
            synergy_no_clda['Late']['min'].append(np.min(s[:, 1]))
            synergy_no_clda['Late']['max'].append(np.max(s[:, 1]))


# Plotting distributed UAC for each seed
if pool_clda_data:
    shifts = [-0.25, 0, 0, 0]
else:
    shifts = [-0.25, 0, 0.25, 0.5]
for i, seed in enumerate(seeds):
    fig_seed, ax_seed = plt.subplots(figsize=(fig_width * units_convert['mm'], fig_width * units_convert['mm']))
    for clda_i, clda in enumerate(alphas_clda):
        if use_log:
            synergy['Late'][seed][clda][:, 1] = (synergy['Late'][seed][clda][:, 1] - synergy_no_clda['Late']['min'][i]) / \
                                                (synergy_no_clda['Late']['max'][i] - synergy_no_clda['Late']['min'][i])
            #print(f"CLDA {clda}: {np.min(synergy['Late'][seed][clda][:, 1])}")
        else:
            synergy['Late'][seed][clda][:, 1] = (synergy['Late'][seed][clda][:, 1] - synergy_no_clda['Late']['min'][
                i]) / \
                                                (synergy_no_clda['Late']['max'][i] - synergy_no_clda['Late']['min'][i])
            #print(f"CLDA {clda}: {np.max(synergy['Late'][seed][clda][:, 1])}")

        if pool_clda_data:
            if clda_i > 1:
                ax_seed.plot(synergy['Late'][seed][clda][:, 0] - shifts[clda_i], synergy['Late'][seed][clda][:, 1],
                             color=pooled_data_color if clda < 1 else 'black', lw=0, marker='.', markersize=0.3, alpha=0.2,
                             label=None)
            else:
                ax_seed.plot(synergy['Late'][seed][clda][:, 0] - shifts[clda_i], synergy['Late'][seed][clda][:, 1],
                             color=pooled_data_color if clda < 1 else 'black', lw=0, marker='.', markersize=0.3, alpha=0.2,
                             label=f"Pooled CLDA" if clda < 1 else "Fixed")
        else:
            ax_seed.plot(synergy['Late'][seed][clda][:, 0] - shifts[clda_i], synergy['Late'][seed][clda][:, 1],
                         color=clda_colors[clda], lw=0, marker='.', markersize=0.3, alpha=0.2, label=f"CLDA = {1.-clda:.2}" if clda < 1 else "Fixed")
    ax_seed.set_xlabel("Combination size")
    ax_seed.set_ylabel("Normalized log performance" if use_log else "Performance (norm.)")
    ax_seed.set_xticks(np.arange(2, 1 + n_readout, 2))
    ax_seed.set_yticks([0,0.5,1])
    ax_seed.legend(loc='lower right', labelspacing=0.25)
    leg = ax_seed.get_legend()
    if pool_clda_data:
        for clda_i, clda in enumerate(alphas_clda[:2]):
            leg.legend_handles[clda_i].set_visible(False)
        for text, clda in zip(leg.get_texts(), alphas_clda[:2]):
            text.set_color(pooled_data_color if clda < 1 else 'black')
    else:
        for clda_i, clda in enumerate(alphas_clda):
            leg.legend_handles[clda_i].set_visible(False)
        for text, clda in zip(leg.get_texts(), alphas_clda):
            text.set_color(clda_colors[clda])
    sns.despine()
    plt.tight_layout()
    fig_seed.savefig(os.path.join(f"{result_folder_prefix}", f"DistributedUAC_Seed{seed}.png"))


# Plotting the spread of the distributions (IQR)
iqrs = {clda: np.empty((len(seeds), n_readout)) for clda in alphas_clda}
for clda in alphas_clda:
    for seed_i, seed in enumerate(seeds):
        for readout_i in range(n_readout):
            select_readouts = synergy['Late'][seed][clda][:, 0] == readout_i + 1
            iqrs[clda][seed_i, readout_i] = iqr(synergy['Late'][seed][clda][select_readouts, 1])

fig, ax = plt.subplots(figsize=(fig_width * units_convert['mm'], fig_width * units_convert['mm']))
for clda in alphas_clda:
    m = np.mean(iqrs[clda], axis=0)
    sem = np.std(iqrs[clda], axis=0, ddof=1) / iqrs[clda].shape[0]**0.5
    ax.errorbar(np.arange(1, 1+n_readout), m, yerr=sem, color=clda_colors[clda], label=f"CLDA = {1.-clda:.2}" if clda < 1 else "Fixed")
ax.set_xticks(np.arange(2, 1 + n_readout, 2))
ax.set_yticks([0, 0.05, 0.1])
ax.set_xlabel("Combination size")
ax.set_ylabel("Interquartile range")
sns.despine()
ax.legend(loc='upper right', handlelength=0, handletextpad=0, labelspacing=0.25)
leg = ax.get_legend()
for clda_i, clda in enumerate(alphas_clda):
    leg.legend_handles[clda_i].set_visible(False)
for text, clda in zip(leg.get_texts(), alphas_clda):
    text.set_color(clda_colors[clda])
#ax.legend(loc='upper right', handlelength=1, handleheight=0.4)
plt.tight_layout()
fig.savefig(os.path.join(f"{result_folder_prefix}", f"IQR.pdf"))

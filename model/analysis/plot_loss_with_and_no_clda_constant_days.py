import matplotlib.pyplot as plt
import os
import numpy as np
from utils import units_convert, clda_colors
plt.style.use('rnn4bci_plot_params.dms')
import seaborn as sns

"""
Description:
-----------
Plot the loss for learning with and without CLDA, when there is no stopping criterion in the simulation.
I.e., networks learned the task for the same number of days.
Contrast this to sister script `plot_loss_with_and_no_clda.py` which compares loss when the number of training days
are different.
"""
plt.rc('font', size=8)          # controls default text sizes
plt.rc('axes', titlesize=8)     # fontsize of the axes title
plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
plt.rc('ytick', labelsize=7)    # fontsize of the tick labels
plt.rc('legend', fontsize=6)    # legend fontsize
plt.rc('figure', titlesize=8)  # fontsize of the figure title

# Define data (load) and result (save) directories
data_dir = f"../data/bci-model"
result_dir = f"../results/bci-model/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# Plot for each seed and statistics if len(seeds) > 1
subsampling = 100  # number of epochs / day
nb_seeds = 1
normalize = True  # whether to normalize the loss to the loss without CLDA
seeds = np.arange(1, nb_seeds+1)
alphas = [0.5, 0.75, 0.9, 1]   # values of CLDA parameter

losses = {alpha: [] for alpha in alphas}


for seed_id, seed in enumerate(seeds):
    fig, ax = plt.subplots(figsize=(45 * units_convert['mm'], 45 * units_convert['mm']/1.15))

    for alpha in alphas:
        data_folder = f"../data/bci-model-clda{alpha}/seed{seed}"
        loss = np.log10(np.loadtxt(f"{data_folder}/bmi_loss.txt")[::subsampling, 1])

        if normalize:
            data_folder_no_clda = f"../data/bci-model-clda1/seed{seed}"
            loss_no_clda = np.log10(np.loadtxt(f"{data_folder_no_clda}/bmi_loss.txt")[::subsampling, 1])
            #normalized_loss = (-loss + np.max(loss_no_clda)) / (-np.min(loss_no_clda) + np.max(loss_no_clda))
            normalized_loss = (-loss + loss_no_clda[0]) / (-loss_no_clda[-1] + loss_no_clda[0])
            losses[alpha].append(normalized_loss)
            ax.plot(np.arange(1, len(loss)+1), normalized_loss, alpha=1 if subsampling > 1 else 0.5,
                        lw=1, color=clda_colors[alpha], label=f'CLDA = {1 - alpha:0.2}' if alpha < 1 else 'Fixed')
        else:
            losses[alpha].append(loss)
            ax.semilogy(np.arange(1, len(loss)+1), loss, alpha=1 if subsampling > 1 else 0.5,
                        lw=1, color=clda_colors[alpha], label=f'CLDA = {1 - alpha:0.2}' if alpha < 1 else 'Fixed')
        if subsampling > 1:
            ax.set_xlabel("Days")
        else:
            ax.set_xlabel("Epoch")
        if normalize:
            ylabel_ = "Normalized log performance"
        else:
            ylabel_ = "Log$_10$ loss"
    ax.set_ylabel(ylabel_)
    ax.legend(loc='best')
    sns.despine()
    fig.tight_layout()
    plt.savefig(f'{result_dir}/Loss_Seed{seed}.png')
    plt.close(fig)

# Plot average performance across seeds
fig, ax = plt.subplots(figsize=(45 * units_convert['mm'], 45 * units_convert['mm']))
for alpha in alphas[::-1]:
    losses[alpha] = np.vstack(losses[alpha])
    m = np.mean(losses[alpha], axis=0)
    sem = np.std(losses[alpha], ddof=1, axis=0) / losses[alpha].shape[0]**0.5
    ax.plot(np.arange(1, 1 + m.shape[0]), m,
            lw=1, color=clda_colors[alpha], label=f'CLDA = {1 - alpha:0.2}' if alpha < 1 else 'Fixed')
    ax.fill_between(np.arange(1, m.shape[0] + 1), m - sem, m + sem, lw=0,
                    color=clda_colors[alpha], alpha=0.5)
ax.set_yticks([0, 0.5, 1])
if subsampling > 1:
    ax.set_xlabel("Days")
else:
    ax.set_xlabel("Epoch")
if normalize:
    ylabel_ = "Log performance (norm.)"
else:
    ylabel_ = "Loss"

ax.set_ylabel(ylabel_)
ax.legend(loc='best', labelspacing=0.25)
leg = ax.get_legend()
for clda_i, clda in enumerate(alphas):
    leg.legend_handles[clda_i].set_visible(False)
for text, clda in zip(leg.get_texts(), alphas[::-1]):
    text.set_color(clda_colors[clda])
sns.despine()
fig.tight_layout()
plt.savefig(f'{result_dir}/Loss_using_start_and_end.png')
plt.close(fig)


# Plot distributions of average loss on the last n days
n_days_for_average = 5
average_losses = {alpha: None for alpha in alphas}
for alpha in alphas:
    average_losses[alpha] = np.mean(losses[alpha][:, -n_days_for_average*100//subsampling:], axis=1)

# find how many seeds have with-CLDA performance greater than without-CLDA
n_seeds_lower = {alpha: [] for alpha in alphas if alpha < 1}
for alpha in n_seeds_lower.keys():
    n_seeds_lower[alpha] = len(np.nonzero(average_losses[alpha] >= average_losses[1])[0])


plt.figure(figsize=(45*units_convert['mm'], 45*units_convert['mm']))
av_loss = np.empty((len(alphas), len(seeds)))
for i, alpha in enumerate(alphas):
    av_loss[i, :] = average_losses[alpha]
plt.plot(1-np.array(alphas), av_loss, color='grey', lw=0.5)
for i, alpha in enumerate(alphas):
    plt.plot((1-alpha) * np.ones(len(seeds)), average_losses[alpha],
             color=clda_colors[alpha], marker='o', markersize=4, markeredgecolor='white', markeredgewidth=0.5, lw=0)
    if alpha != 1:
        plt.text(1-alpha, np.max(average_losses[alpha])+0.1, f"{n_seeds_lower[alpha]}/{len(seeds)}", ha='center', fontsize=6)
plt.xlabel("CLDA")
plt.xticks(1 - np.array(alphas))
xtick_labels = 1 - np.array(alphas)
plt.gca().set_xticklabels([f'{x:.2}' for x in xtick_labels])
sns.despine()
#plt.ylabel(f"Normalized log perf.\n averaged over last {n_days_for_average} days")
plt.ylabel(f"Avg. end performance")
plt.tight_layout()
plt.savefig(f'{result_dir}/AverageEndPerformance.png')

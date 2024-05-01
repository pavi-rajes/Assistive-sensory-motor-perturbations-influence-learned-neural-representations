import matplotlib.pyplot as plt
import os
from utils import units_convert, clda_colors, pooled_data_color
import seaborn as sns
import numpy as np
from scipy.stats import linregress

plt.style.use("rnn4bci_plot_params.dms")

"""Compute neural adding curve using the single-unit-ranking method.
In a nutshell:
1. Rank units according to their effect on reach performance.
2. In turn, add unit according to its single-unit rank, until no more unit can be added.
See file `nac_schematic.png` for more details.
"""

plt.rc('font', size=8)          # controls default text sizes
plt.rc('axes', titlesize=8)     # fontsize of the axes title
plt.rc('axes', labelsize=8)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
plt.rc('ytick', labelsize=7)    # fontsize of the tick labels
plt.rc('legend', fontsize=6)    # legend fontsize
plt.rc('figure', titlesize=8)  # fontsize of the figure title

# Functions
def rank_units(data):
    nb_readouts = data.shape[1] - 1
    performance_single_units = data[:nb_readouts, -1]
    return np.argsort(performance_single_units), np.sort(performance_single_units)


pool_CLDA_data = False  # whether to pool all CLDA data irrespective of the value of alpha
normalize = True  # whether to normalize the performances before computing statistics
fixed_nb_days = True  # whether the training occurred on a fixed number of days, as opposed to being stopped when a performance threshold was reached
nb_days_for_average_loss = 5  # only used when `fixed_nb_days = True`
subsampling = 100  # nb of epochs per day; only used when `fixed_nb_days = True`
plot_only_with_clda = False
select_seed_from_criterion = False  # whether to select on seeds for which CLDA performance is better than without CLDA
fig_width = 40  # mm

nb_seeds = 1
seeds = list(range(1, 1 + nb_seeds))
alphas = [1, 0.9, 0.75, 0.5]
ranked_perf = {'late': {alpha:[] for alpha in alphas},
               'early': {alpha:[] for alpha in alphas}}
ranked_perf_single_unit = {'late': {alpha:[] for alpha in alphas},
                           'early': {alpha:[] for alpha in alphas}}

data_dir = f"../data/bci-model"
result_dir = f"../results/bci-model/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

diff_n_days = []
starting_loss_no_CLDA = []

n_seeds_with_better_clda = 0
n_seeds_clda = 0

for alpha in alphas:
    for seed in seeds:
        for stage in ['early', 'late']:
            # Load data
            syn_comb = np.loadtxt(f"{data_dir}-clda{alpha}/seed{seed}/synergy_combinations_{stage}_.txt")

            nb_readouts = syn_comb.shape[1] - 1
            size_of_subsets = np.sum(syn_comb[:, :nb_readouts], axis=1)
            size_of_subsets = size_of_subsets.astype(int)

            if fixed_nb_days:
                loss = np.loadtxt(f"{data_dir}-clda{alpha}/seed{seed}/bmi_loss.txt")[::subsampling, 1]
                average_end_loss = np.mean(loss[-nb_days_for_average_loss:])
                loss = np.loadtxt(f"{data_dir}-clda1/seed{seed}/bmi_loss.txt")[::subsampling, 1]
                average_end_loss_no_CLDA = np.mean(loss[-nb_days_for_average_loss:])
                criterion = average_end_loss - average_end_loss_no_CLDA if select_seed_from_criterion else -1
            else:
                n_training_days = np.loadtxt(f"{data_dir}-clda{alpha}/seed{seed}/number_of_training_days.txt", dtype=int)
                n_training_days_no_CLDA = np.loadtxt(f"{data_dir}-clda1/seed{seed}/number_of_training_days.txt", dtype=int)
                criterion = n_training_days - n_training_days_no_CLDA if select_seed_from_criterion else -1

            if alpha != 1:
                if not fixed_nb_days:
                    diff_n_days.append(criterion)
                starting_loss_no_CLDA.append(np.loadtxt(f"{data_dir}-clda1/seed{seed}/bmi_loss.txt")[0, 1])
                n_seeds_clda += 1

            if alpha == 1 or (alpha != 1 and criterion < 0):
                if alpha != 1 and select_seed_from_criterion:
                    n_seeds_with_better_clda += 1
                # Construct NAC
                indices_single_unit, performances_single_unit = rank_units(syn_comb)
                if normalize:
                    norm_perf = (performances_single_unit - performances_single_unit[0]) / \
                                (performances_single_unit[-1] - performances_single_unit[0])
                    ranked_perf_single_unit[stage][alpha].append(norm_perf)
                else:
                    ranked_perf_single_unit[stage][alpha].append(performances_single_unit)
                indices = np.zeros(nb_readouts, dtype=int)
                indices[indices_single_unit[-1]] = 1
                performances = [performances_single_unit[-1]]
                for k in range(2, nb_readouts+1):
                    indices[indices_single_unit[-k]] = 1
                    combinations_with_k = np.where(size_of_subsets == k)[0]
                    data_select = syn_comb[combinations_with_k]
                    #print(np.sum(data_select[:,:-1] - indices, axis=1))
                    #print(np.sum(data_select[:,:-1] - indices, axis=1) == 0)
                    select_subset = np.all(np.isclose(data_select[:,:-1], indices), axis=1)
                    performances.append(data_select[select_subset, -1][0])
                if normalize:
                    norm_perf = (performances - performances[0]) / \
                                (performances[-1] - performances[0])
                    ranked_perf[stage][alpha].append(norm_perf)
                else:
                    ranked_perf[stage][alpha].append(performances)
                # Plot single-unit performances
                # plt.figure(figsize=(30 * units_convert['mm'], 30 / 1.25 * units_convert['mm']))
                # plt.plot(np.arange(1, nb_readouts + 1), performances_single_unit[::-1], marker='o', markersize=1)
                # plt.xticks([1, 6, 12])
                # plt.xlabel('Ranked unit', fontsize=5)
                # plt.ylabel('Single-unit\nreward', fontsize=5)
                # sns.despine(trim=True)
                # plt.tight_layout()
                # plt.close()

                # Plot performances
                # plt.figure(figsize=(45 * units_convert['mm'], 45/1.25 * units_convert['mm']))
                # plt.plot(np.arange(1, nb_readouts+1),  performances, marker='o', markersize=1)
                # plt.xticks([1, 6, 12])
                # plt.xlabel('Ranked unit')
                # plt.ylabel('Reward')
                # sns.despine(trim=True)
                # plt.tight_layout()
                # plt.close()

print(f"Proportion of seeds with improvement using CLDA: {n_seeds_with_better_clda}/{n_seeds_clda}",
      n_seeds_with_better_clda/n_seeds_clda)

if not fixed_nb_days:
    # Plot histogram diff days
    plt.figure(figsize=(45 / 1.25 * units_convert['mm'], 45 / 1.25 * units_convert['mm']))
    bins = np.arange((min(diff_n_days) - 3) / 3, (max(diff_n_days)+5) / 3)
    bins = 3 * (bins - 0.5)
    plt.hist(diff_n_days, bins=bins, color='grey')
    plt.xlabel('Days(CLDA) - Days(fixed)')
    plt.ylabel('Count')
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(result_dir + "DiffDays.png")

    # Plot starting loss w/o CLDA and difference in number of days
    plt.figure(figsize=(45 / 1.25 * units_convert['mm'], 45 / 1.25 * units_convert['mm']))
    slope, intercept, r_value, p_value, std_err = linregress(np.array(diff_n_days), np.array(starting_loss_no_CLDA))
    plt.scatter(np.array(diff_n_days), np.array(starting_loss_no_CLDA), s=2, color='black')
    reg_line = lambda x: intercept + slope * x
    #plt.plot([min(diff_n_days), max(diff_n_days)], [reg_line(min(diff_n_days)), reg_line(max(diff_n_days))], color='grey')
    plt.text(0.5, 1, f"$R^2 = {r_value**2:.1}$, p = {p_value:.1}", fontsize=4, transform=plt.gca().transAxes)
    plt.xlabel('Days(CLDA) - Days(fixed)')
    plt.ylabel('Initial loss')
    sns.despine()
    plt.tight_layout()
    plt.savefig(result_dir + "StartingLossNoCLDA_vs_DiffDaysAll.png")

    plt.figure(figsize=(45 / 1.25 * units_convert['mm'], 45 / 1.25 * units_convert['mm']))
    indices = np.array(diff_n_days) < 0
    slope, intercept, r_value, p_value, std_err = linregress(np.array(diff_n_days)[indices], np.array(starting_loss_no_CLDA)[indices])
    plt.scatter(np.array(diff_n_days)[indices], np.array(starting_loss_no_CLDA)[indices], s=2, color='black')
    reg_line = lambda x: intercept + slope * x
    plt.plot([min(np.array(diff_n_days)[indices]), max(np.array(diff_n_days)[indices])],
             [reg_line(min(np.array(diff_n_days)[indices])), reg_line(max(np.array(diff_n_days)[indices]))], color='grey')
    plt.text(0.5, 1, f"$R^2 = {r_value**2:.1}$, p = {p_value:.1}", fontsize=4, transform=plt.gca().transAxes)
    plt.xlabel('Days(CLDA) - Days(fixed)')
    plt.ylabel('Initial loss')
    sns.despine()
    plt.tight_layout()
    plt.savefig(result_dir + "StartingLossNoCLDA_vs_DiffDays.png")


# Save ranked curves
np.save(f"{result_dir}/ranked_performance", ranked_perf)
np.save(f"{result_dir}/ranked_performance_single_unit", ranked_perf_single_unit)

# Plot average single-unit performances
if pool_CLDA_data:
    plt.figure(figsize=(fig_width * units_convert['mm'], fig_width * units_convert['mm']))
    for stage in ['early', 'late']:
        rpsu = []
        for alpha in alphas[1:]:
            rpsu += ranked_perf_single_unit[stage][alpha]
        if not plot_only_with_clda:
            m = np.mean(ranked_perf_single_unit[stage][1], axis=0)
            sem = np.std(ranked_perf_single_unit[stage][1], ddof=1, axis=0) / len(ranked_perf_single_unit[stage][1]) ** 0.5
            plt.errorbar(np.arange(1, nb_readouts + 1), m[::-1], yerr=sem[::-1], color='k', ls='-' if stage=='late' else (0, (2, 1)),
                         marker='' if stage=='late' else '', markersize=1.5, label=f"Fixed, {stage[0].capitalize()}{stage[1:]}")
        #($n = {len(ranked_perf_single_unit[stage][1])}$)
        m = np.mean(rpsu, axis=0)
        sem = np.std(rpsu, ddof=1, axis=0) / len(rpsu) ** 0.5
        plt.errorbar(np.arange(1, nb_readouts + 1), m[::-1], yerr=sem[::-1], color=pooled_data_color, ls='-' if stage=='late' else (0, (2, 1)),
                     marker='' if stage=='late' else '', markersize=1.5, label=f"Pooled CLDA, {stage[0].capitalize()}{stage[1:]}")
        #($n = {len(rpsu)}$)
    plt.xticks([1, 6, 12])
    if not normalize:
        plt.ylim([-150, -30])
    if normalize:
        plt.yticks([0, 0.5, 1])
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 2, 1, 3]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
               loc='lower left', handlelength=2, labelspacing=0.25)
    plt.xlabel('Ranked readouts')
    plt.ylabel('Performance (norm.)' if normalize else 'Single-unit reward')
    sns.despine()
    plt.tight_layout()
    plt.savefig(result_dir + f"SingleUnitRewardRanked_Normalized{normalize}.pdf")
else:
    for alpha in alphas:
        plt.figure(figsize=(45 * units_convert['mm'], 45 / 1.25 * units_convert['mm']))
        for stage in ['early', 'late']:
            m = np.mean(ranked_perf_single_unit[stage][alpha], axis=0)
            sem = np.std(ranked_perf_single_unit[stage][alpha], ddof=1, axis=0) / len(ranked_perf_single_unit[stage][alpha])**0.5
            plt.errorbar(np.arange(1, nb_readouts + 1), m[::-1], yerr=sem[::-1], color=clda_colors[alpha], ls='-' if stage=='late' else (0, (2, 1)),
                         marker='.', markersize=1,
                         label=f"{stage[0].capitalize()}{stage[1:]}")
        plt.title(f"CLDA = {1-alpha:.2}" if alpha != 1 else "Fixed", pad=0)
        plt.xticks([1, 6, 12])
        if not normalize:
            plt.ylim([-150, -30])
        if normalize:
            plt.yticks([0, 0.5, 1])
        plt.legend(loc='lower left', handlelength=2)
        plt.xlabel('Ranked unit')
        plt.ylabel('Single-unit\nnormalized performance' if normalize else 'Single-unit reward')
        sns.despine()
        plt.tight_layout()
        plt.savefig(result_dir + f"SingleUnitRewardRanked_Normalized{normalize}_CLDA{1.-alpha:.2}.png")


# Plot average performances
if pool_CLDA_data:
    plt.figure(figsize=(45 * units_convert['mm'], 45 / 1.25 * units_convert['mm']))
    for stage in ['early', 'late']:
        rp = []
        for alpha in alphas[1:]:
            rp += ranked_perf[stage][alpha]
        if not plot_only_with_clda:
            m = np.mean(ranked_perf[stage][1], axis=0)
            sem = np.std(ranked_perf[stage][1], ddof=1, axis=0) / len(ranked_perf[stage][1]) ** 0.5
            plt.errorbar(np.arange(1, nb_readouts + 1), m, yerr=sem, color='k', ls='-' if stage=='late' else (0, (2, 1)),
                         marker='.', markersize=1, label=f"{stage[0].capitalize()}{stage[1:]}, Fixed")
        m = np.mean(rp, axis=0)
        sem = np.std(rp, ddof=1, axis=0) / len(rp) ** 0.5
        plt.errorbar(np.arange(1, nb_readouts + 1), m, yerr=sem, color='grey', ls='-' if stage=='late' else (0, (2, 1)),
                     marker='.', markersize=1, label=f"{stage[0].capitalize()}{stage[1:]}, pooled CLDA")
        plt.xticks([1, 6, 12])
        if not normalize:
            plt.ylim([-50, 0])
        if normalize:
            plt.ylim([-0.05, 1.05])
            plt.yticks([0, 0.5, 1])
        plt.legend(loc='lower right', fontsize=5,  handlelength=2)
        plt.xlabel('No. of ranked readout units')
        plt.ylabel('Normalized performance' if normalize else "Reward")
        sns.despine()
        plt.tight_layout()
        plt.savefig(result_dir + f"SingleUnitBasedNAC_Normalized{normalize}.png")
else:
    for alpha in alphas:
        plt.figure(figsize=(fig_width * units_convert['mm'], fig_width * units_convert['mm']))

        for stage in ['early', 'late']:
            m = np.mean(ranked_perf[stage][alpha], axis=0)
            sem = np.std(ranked_perf[stage][alpha], ddof=1, axis=0) / len(ranked_perf[stage][alpha]) ** 0.5
            plt.errorbar(np.arange(1, nb_readouts + 1), m, yerr=sem, color=clda_colors[alpha], ls='-' if stage=='late' else (0, (2, 1)),
                         marker='' if stage=='late' else '', markersize=1.5, label=f"{stage[0].capitalize()}{stage[1:]}")
        #plt.plot(np.arange(1, nb_readouts + 1), 0.8*np.ones(nb_readouts), ':', lw=0.5, color=(0.6, 0.6, 0.6))

        plt.title(f"CLDA = {1-alpha:.2}" if alpha != 1 else "Fixed", pad=1)
        plt.xticks([1, 6, 12])
        if not normalize:
            plt.ylim([-50, 0])
        if normalize:
            plt.ylim([-0.05, 1.05])
            plt.yticks([0, 0.5, 1])
        plt.legend(loc='lower right', handlelength=2, labelspacing=0.25)
        plt.xlabel('# of ranked readouts')
        plt.ylabel('Performance (norm.)' if normalize else "Reward")
        sns.despine()
        plt.tight_layout()
        plt.savefig(result_dir + f"SingleUnitBasedNAC_Normalized{normalize}_CLDA{1.-alpha:.2}.pdf")

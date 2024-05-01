import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import argparse
from utils import units_convert, task_colors
plt.style.use('rnn4bci_plot_params.dms')

# Parsing input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-resultdir", type=str, help="parent result directory", default=".")
parser.add_argument("-datadir", type=str, help="parent data directory", default=".")
parser.add_argument("-seed", type=str, help="seed", default=1)
parser.add_argument("-type", type=str,
                    help="type of trajectories to plot: manual, bci or manual_before_learning", default="bci")
parser.add_argument("-subsampling", type=int, help="plot loss every `subsampling` epoch", default=1)
args = parser.parse_args()

seed = args.seed
subsampling = args.subsampling
type_of_loss = args.type

if type_of_loss == "manual_before_learning":
    file_suffix = "ManualBeforeLearning"
else:
    file_suffix = f"{type_of_loss[0].capitalize()}{type_of_loss[1:]}"

result_folder = f"{args.resultdir}/seed{seed}"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

data_folder = f"{args.datadir}/seed{seed}"
if type_of_loss == "bci":
    loss = np.loadtxt(f"{data_folder}/bmi_loss.txt")[::subsampling,1]
elif type_of_loss == "manual_before_learning":
    loss = np.loadtxt(f"{data_folder}/manual_trained_network_loss.txt")[::subsampling, 1]
else:
    loss = np.loadtxt(f"{data_folder}/{type_of_loss}_loss.txt")[::subsampling,1]

# Plot
fig, ax = plt.subplots(figsize=(45 * units_convert['mm'], 45 * units_convert['mm']/1.25))
ax.semilogy(np.arange(1, len(loss)+1), loss, lw=1,
            color=task_colors['manual'] if "manual" in type_of_loss else task_colors['bci'])
if subsampling > 1:
    ax.set_xlabel("Day")
else:
    ax.set_xlabel("Epoch")
ylabel_ = "Manual loss" if "manual" in type_of_loss else "BCI loss"
ax.set_ylabel(ylabel_)
fig.tight_layout()
plt.savefig(f'{result_folder}/Loss_{file_suffix}.png')
plt.close(fig)

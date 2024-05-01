import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import argparse
from utils import units_convert, target_colors
plt.style.use('rnn4bci_plot_params.dms')

# Parsing input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-n_targets", type=int, help="number of targets", default=8)
parser.add_argument("-n_reals", type=int, help="number of realizations", default=5)
parser.add_argument("-hold_duration", type=int, help="duration of the preparatory hold", default=25)
parser.add_argument("-resultdir", type=str, help="parent result directory", default=".")
parser.add_argument("-datadir", type=str, help="parent data directory", default=".")
parser.add_argument("-seed", type=str, help="seed", default=1)
parser.add_argument("-type", type=str,
                    help="type of trajectories to plot: naive, manual, bci, manual_before_learning, manual_with_bci_context or gen_bci",
                    default="bci")
args = parser.parse_args()


n_targets, n_reals, hold_duration_int, seed = args.n_targets, args.n_reals, args.hold_duration, args.seed
type_of_trajectory = args.type

if type_of_trajectory == "manual_before_learning":
    file_suffix = "ManualBeforeLearning"
elif type_of_trajectory == "bci_before_learning":
    file_suffix = "BCIBeforeLearning"
elif type_of_trajectory == "manual_with_bci_context":
    file_suffix = "ManualWithBCIcontext"
elif type_of_trajectory == "gen_bci":
    file_suffix = "GeneralizationBCI"
else:
    file_suffix = f"{type_of_trajectory[0].capitalize()}{type_of_trajectory[1:]}"


result_folder = f"{args.resultdir}/seed{seed}"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

data_folder = f"{args.datadir}/seed{seed}"

# Trajectories
fig, ax = plt.subplots(figsize=(45 * units_convert['mm'], 45 * units_convert['mm']))
targets = [[7*np.cos(2*np.pi*i/n_targets), 7*np.sin(2*np.pi*i/n_targets)] for i in range(n_targets)]
targets = np.array(targets)
target_radius = 1.2 if "manual" in type_of_trajectory else 1.4
for target_id in range(n_targets):
    circle = plt.Circle((targets[target_id,0], targets[target_id,1]), target_radius, color='k', clip_on=False)
    ax.add_patch(circle)
    trajectories = np.loadtxt(f"{data_folder}/trajectory_{type_of_trajectory}_{target_id}.txt")
    total_duration = trajectories.shape[0] // n_reals
    for r in range(n_reals):
        hand_motion = trajectories[r*total_duration:(r+1)*total_duration,:2]
        ax.plot(100 * hand_motion[:, 0], 100 * hand_motion[:, 1], color=target_colors[target_id%len(target_colors)], lw=0.5)
ax.vlines(5, -2, -1, colors=['k'], lw=1)
ax.hlines(-2, 5, 6, colors=['k'], lw=1)
ax.set_axis_off()
ax.set_aspect('equal')
fig.tight_layout()
plt.savefig(f'{result_folder}/Trajectory_{file_suffix}.png')
plt.close(fig)

# Speed and acceleration
fig, axes = plt.subplots(figsize=(45 * units_convert['mm'], 45 * units_convert['mm']), nrows=2, sharex=True)
for target_id in range(n_targets):
    trajectories = np.loadtxt(f"{data_folder}/trajectory_{type_of_trajectory}_{target_id}.txt")
    total_duration = trajectories.shape[0] // n_reals
    for r in range(n_reals):
        speed = np.sqrt(trajectories[r * total_duration:(r + 1) * total_duration, 2]**2 +
                        trajectories[r * total_duration:(r + 1) * total_duration, 3]**2)
        acc = np.sqrt(trajectories[r * total_duration:(r + 1) * total_duration, 4]**2 +
                        trajectories[r * total_duration:(r + 1) * total_duration, 5]**2)
        axes[0].plot(0.01*np.arange(-hold_duration_int, speed.shape[0]-hold_duration_int), 100*speed,
                     color=target_colors[target_id%len(target_colors)], lw=0.5)
        axes[1].plot(0.01*np.arange(-hold_duration_int, speed.shape[0]-hold_duration_int), 100*acc,
                     color=target_colors[target_id%len(target_colors)], lw=0.5)
axes[0].set_ylabel("Speed [cm/s]")
axes[1].set_ylabel("Acceleration [cm/s$^2$]")
axes[1].set_xlabel("Time [s]")
axes[0].set_xticks([0,0.5,1])
sns.despine()
fig.tight_layout()
plt.savefig(f'{result_folder}/SpeedAndAcceleration_{file_suffix}.png')
plt.close(fig)
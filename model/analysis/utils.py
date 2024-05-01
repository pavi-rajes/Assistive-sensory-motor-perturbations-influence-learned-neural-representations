import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from numba import njit
import copy
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score
from seaborn import color_palette

# units
units_convert = {'cm': 1 / 2.54, 'mm': 1 / 2.54 / 10}

# Define colorblind-friendly colors
target_colors = [(200. / 255, 0, 0),  # red
                 (0.9, 0.6, 0),  # orange
                 (0.95, 0.9, 0.25),  # yellow
                 (0, 158. / 255, 115. / 255),  # bluish green
                 (86. / 255, 180. / 255, 233. / 255),  # sky_blue
                 (0, 0.45, 0.7),  # blue
                 (75. / 255, 0., 146. / 255),  # purple
                 (0.8, 0.6, 0.7)]  # pink

unit_colors = {'NR': 'black', 'R': (123 / 255, 197 / 255, 181 / 255)}
task_colors = {'manual': (136 / 255, 28 / 255, 125 / 255), 'bci': (123 / 255, 197 / 255, 181 / 255)}

clda_colors = {1: "black",
               0.75: color_palette("colorblind")[1],
               0.5: color_palette("colorblind")[2],
               0.25: color_palette("colorblind")[6],
               0.9: color_palette("colorblind")[0]}

pooled_data_color = color_palette("colorblind")[4]

# Targets representations for labels
labels_for_targets = {1: '0',
                      2: ['0', r'$\pi$'],
                      3: ['0', r'$2\pi/3$', r'$4\pi/3$'],
                      4: ['0', r'$2\pi/4$', r'$2\pi/4$', r'$2\pi/4$'],
                      5: ['0', r'$2\pi/5$', r'$4\pi/5$', r'$6\pi/5$', r'$8\pi/5$'],
                      6: ['0', r'$\pi/3$', r'$2\pi/3$', r'$\pi$', r'$4\pi/3$', r'$5\pi/3$'],
                      7: ['0', r'$2\pi/7$', r'$4\pi/7$', r'$6\pi/7$', r'$8\pi/7$', r'$10\pi/7$', r'$12\pi/7$'],
                      8: ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$',
                          r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$']}

ylabels_for_targets = {1: '0',
                       2: ['0', r'$\pi$'],
                       3: ['0', r'$2\pi/3$', r'$4\pi/3$'],
                       4: ['0', r'$2\pi/4$', r'$2\pi/4$', r'$2\pi/4$'],
                       5: ['0', r'$2\pi/5$', r'$4\pi/5$', r'$6\pi/5$', r'$8\pi/5$'],
                       6: ['0', r'$\pi/3$', r'$2\pi/3$', r'$\pi$', r'$4\pi/3$', r'$5\pi/3$'],
                       7: ['0', r'$2\pi/7$', r'$4\pi/7$', r'$6\pi/7$', r'$8\pi/7$', r'$10\pi/7$', r'$12\pi/7$'],
                       8: ['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$',
                           r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$']}

# User-defined colormaps using the above colors
N = 256
target_color_maps = []
for color in target_colors:
    vals = np.ones((N, 4))
    for i in range(3):
        vals[:, i] = np.flip(np.linspace(color[i], 1, N))
    target_color_maps.append(ListedColormap(vals))


def plot_trajectory(trajectory, cmap, ax=None, s=0.1, marker='o', **kwargs):
    """
    Plot a cursor trajectory in space.

    :param v: velocity
    :param cmap: color map
    :param dt: time increment
    :param x_0: initial position
    :param ax: matplotlib.pyplot.Axes
    :return:
    """
    if ax is None:
        ax = plt.gca()

    times = np.arange(trajectory.shape[0])
    path = ax.scatter(trajectory[:, 0], trajectory[:, 1], marker=marker, s=s, c=times[::1], cmap=cmap, **kwargs)
    return path


def remove_all_spines(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)


def remove_all_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def adjust_subplot_ylim(axes):
    y_lim_min = float("inf")
    y_lim_max = -float("inf")
    for ax in axes.ravel():
        y_lim_min = min(ax.get_ylim()) if min(ax.get_ylim()) < y_lim_min else y_lim_min
        y_lim_max = max(ax.get_ylim()) if max(ax.get_ylim()) > y_lim_max else y_lim_max
    for ax in axes.ravel():
        ax.set_ylim([y_lim_min, y_lim_max])


def adjust_subplot_xlim(axes):
    x_lim_min = float("inf")
    x_lim_max = -float("inf")
    for ax in axes.ravel():
        x_lim_min = min(ax.get_xlim()) if min(ax.get_xlim()) < x_lim_min else x_lim_min
        x_lim_max = max(ax.get_xlim()) if max(ax.get_xlim()) > x_lim_max else x_lim_max
    for ax in axes.ravel():
        ax.set_xlim([x_lim_min, x_lim_max])


def plot_circle(center, radius, ax, **kwargs):
    x = np.linspace(center[0] - radius + 1e-6, center[0] + radius - 1e-6, 1000, endpoint=True)
    top_semi_circle = center[1] + np.sqrt(radius ** 2 - (x - center[0]) ** 2)
    bottom_semi_circle = center[1] - np.sqrt(radius ** 2 - (x - center[0]) ** 2)
    ax.plot(x, top_semi_circle, **kwargs)
    ax.plot(x, bottom_semi_circle, **kwargs)


@njit
def bootstrap_mean(x, select_n, n=10000, circ_data=False):
    s = np.empty(n)
    for k in range(n):
        c = np.random.choice(x, select_n, replace=True)
        if circ_data:
            s[k] = circmean(c)
        else:
            s[k] = np.mean(c)
    return s


@njit
def circmean(x):
    return np.arctan2(np.mean(np.sin(x)), np.mean(np.cos(x)))


def circstd(x):
    """From Mardia & Jupp (2000) and NCSS statistical software."""
    C_1, S_1 = np.sum(np.cos(x)), np.sum(np.sin(x))
    R_1 = np.sqrt(C_1 ** 2 + S_1 ** 2)
    R_1_bar = R_1 / len(x)
    return np.sqrt(-2. * np.log(R_1_bar))


def tuning_curve(rates, angles):
    """Compute tuning curve using linear regression."""
    N = rates.shape[1]
    params = np.empty(shape=(3, N))
    scores = np.empty(shape=(N,))

    X = np.concatenate((np.sin(angles), np.cos(angles)), axis=1)

    lin_reg = LinearRegression()

    for n in range(N):
        reg = lin_reg.fit(X, rates[:, n])
        scores[n] = reg.score(X, rates[:, n])
        B_sin, B_cos = reg.coef_

        PD = np.arctan2(B_sin, B_cos)
        MD = (B_sin * B_sin + B_cos * B_cos) ** 0.5
        PD = PD + 2 * np.pi if PD < 0 else PD

        params[:, n] = [MD, PD, reg.intercept_]

    return params, scores


def cosine_tuning(x, md, pd, offset):
    return md * np.cos(x - pd) + offset


def angle_between_vectors(v1, v2):
    angle = np.arccos(v1 / np.sqrt(v1 @ v1) @ v2 / np.sqrt(v2 @ v2))
    return angle


def angle_between_plane_and_vector(normal, v):
    angle = angle_between_vectors(normal, v)

    tmp = angle_between_vectors(-normal, v)
    angle = tmp if tmp < angle else angle

    tmp = angle_between_vectors(normal, -v)
    angle = tmp if tmp < angle else angle

    tmp = angle_between_vectors(-normal, -v)
    angle = tmp if tmp < angle else angle

    return np.pi / 2 - angle


def angle_between_two_planes(normal1, normal2):
    angle = angle_between_vectors(normal1, normal2)

    tmp = angle_between_vectors(-normal1, normal2)
    angle = tmp if tmp < angle else angle

    tmp = angle_between_vectors(normal1, -normal2)
    angle = tmp if tmp < angle else angle

    tmp = angle_between_vectors(-normal1, -normal2)
    angle = tmp if tmp < angle else angle

    return angle


def mstd(x, axis):
    """Compute mean and standard deviation of x along `axis`"""
    return np.mean(x, axis=axis), np.std(x, axis=axis, ddof=1)


def set_linewidth_in_legend(ax, linewidth=1):
    handles, labels = ax.get_legend_handles_labels()
    handles = [copy.copy(ha) for ha in handles]
    [ha.set_linewidth(linewidth) for ha in handles]
    return handles, labels


def adjust_angular_difference(dangles):
    dangles[dangles > 180] = dangles[dangles > 180] - 360
    dangles[dangles < -180] = dangles[dangles < -180] + 360
    return dangles


# def hist_with_bootstrap_pdf(x, ax=None, bins=50, circ_data=False, **params):
#     if ax is None:
#         ax = plt.gca()
#
#     n, bins, _ = ax.hist(x, cumulative=False, density=False, bins=bins, color='black', zorder=0)
#     bootstrap = bootstrap_mean(np.array(x), len(x), n=50000, circ_data=circ_data)
#     n_bs, bins = np.histogram(bootstrap, density=True, bins=100)
#     ax.plot(bins[:-1], n_bs, color='blue', lw=1, label='bootstrap means PDF')
#     ax.annotate('mean = {:.2}'.format(np.mean(x)), (0.95, 0.9), xycoords='axes fraction', fontsize=4, ha='right')
#
#     #ax.legend()
#     #ax.set_yticks([])
#
#
#     # Handling params
#     for key in params:
#         if key == 'xlim':
#             ax.set_xlim(params[key])
#         elif key == 'xticks':
#             ax.set_xticks(params[key])
#         elif key == 'xticklabels':
#             ax.set_xticklabels(params[key])
#         elif key == 'xlabel':
#             ax.set_xlabel(params[key])
#
#     return n, n_bs

def cross_val(data, nb_factors, nb_folds=5):
    n_samples, n_features = data.shape
    X = copy.deepcopy(data)

    # Shuffle
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X = X[indices, :]

    # FA object
    fa = FactorAnalysis(nb_factors)

    # Construct folds
    n_samples_per_folds = n_samples // nb_folds
    test_data = [X[f * n_samples_per_folds:(f + 1) * n_samples_per_folds, :] for f in range(nb_folds)]
    training_data = [X[n_samples_per_folds:, :]]
    for i in range(1, nb_folds):
        training_data.append(np.vstack((X[:i * n_samples_per_folds, :],
                                        X[(i + 1) * n_samples_per_folds:, :])))

    # Model eval
    average_ll = 0.
    for f in range(nb_folds):
        fa.fit(training_data[f])
        average_ll += fa.score(test_data[f])
    return average_ll / nb_folds


def linear_dynamical_system_test(n_features, n_samples, rank, sigma=1.):
    rng = np.random.default_rng(seed=1723)
    A = np.array([[-0.813361,         0,         0,         0,         0],
  [0.55566, -0.838527,         0 ,        0,         0],
[0.0383651,  0.956975,  -0.66782,         0,         0],
 [0.161779,  0.687755, -0.223855, -0.987973,         0],
[-0.603254, -0.213236, -0.606959,  0.670517,  0.559478]])
    #A = rng.uniform(low=-1, high=1, size=(rank, rank))
    #for c in range(1, rank):
    #    for r in range(c):
    #        A[r, c] = 0.
    sigma = 1.0
    Z = np.empty(shape=(n_samples, rank))
    Z[0, :] = rng.normal(size=(rank,))
    for i in range(1, n_samples):
        Z[i, :] = A@Z[i-1, :] + rng.normal(size=(rank,))
    U, _, _ = np.linalg.svd(rng.normal(size=(n_features, n_features)))

    X = Z @ U[:, :rank].T

    # Adding homoscedastic noise
    X += sigma * rng.normal(size=(n_samples, n_features))
    return X

def compute_scores(X, n_components, subsample=1):
    fa = FactorAnalysis()

    fa_scores = []
    for n in n_components:
        fa.n_components = n
        fa_scores.append(np.mean(cross_val_score(fa, X[::subsample, :])))

    return fa_scores


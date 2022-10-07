import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from .algorithms import GradientDescent, HeavyBall, ConjugateGradient


def compare_methods(dimension, function, nb_steps, path, seed=None):
    np.random.seed(seed=seed)
    distances = list()
    legend_list = list()
    excess_losses = list()

    x0 = np.random.randn(dimension).astype(np.float128)

    for alg in tqdm([GradientDescent(parametrization="Constant"),
                     GradientDescent(parametrization="PS variant"),
                     GradientDescent(parametrization="PS"),
                     HeavyBall(parametrization="Constant"),
                     HeavyBall(parametrization="Adaptive"),
                     ConjugateGradient()
                     ]):
        x_list, f_list = alg.run(function=function, x0=x0, nb_steps=nb_steps)

        distance = [np.sum((x - function.argmin) ** 2) for x in x_list]
        distances.append(distance)

        excess_loss = [fx - function.min_value for fx in f_list]
        excess_losses.append(excess_loss)

        legend_list.append(alg.name)

    path_search = os.sep
    for intermediate_dir in path.split(os.sep)[:-1]:
        path_search = os.path.join(path_search, intermediate_dir)
        if not os.path.isdir(path_search):
            os.mkdir(path_search)

    fontsize = 38
    plt.rcParams.update({'font.size': fontsize})
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = [default_colors[index] for index in [0, 3, 5, 2, 1, 4]]
    linestyles = ["-", "--"] * 3

    plt.figure(figsize=(15, 8))
    plt.rcParams['axes.linewidth'] = 3
    plt.gca().set_prop_cycle(color=colors, linestyle=linestyles)
    lines = plt.plot(np.log10(distances).T, lw=4)
    plt.ylim(ymin=max(-30, -.2 + np.min(np.log10(distances))))
    plt.xlabel("Iterations")
    plt.ylabel("Distance (log)")
    plt.savefig(path + "_distances.png", bbox_inches="tight")

    legend_fig = plt.figure(figsize=(15, 8))
    legend_fig.legend(lines, legend_list, fontsize=26, loc='center', ncol=3, frameon=False)
    legend_fig.gca().axis('off')
    plt.savefig(path + "_legend.png")

    fig = plt.figure(figsize=(15, 8))
    plt.rcParams['axes.linewidth'] = 3
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(color=colors, linestyle=linestyles)
    plt.plot(np.log10(excess_losses).T, lw=4)
    plt.ylim(ymin=max(-30, -.2 + np.min(np.log10(excess_losses))))
    plt.xlabel("Iterations")
    plt.ylabel("Excess loss (log)")
    plt.savefig(path + "_losses.png", bbox_inches="tight")

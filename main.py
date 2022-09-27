import numpy as np

import matplotlib.pyplot as plt

from Optimization.Functions import Quadratic
from Optimization.Algorithms import HeavyBall, ConjugateGradient

kappa = 10**6
d = 10**3

precision = np.float128
seed = 42
np.random.seed(seed=seed)
nb_steps = 2*d

log_L = 1
log_mu = log_L - np.log(kappa) / np.log(10)

eig_list = np.logspace(log_mu, log_L, d).astype(precision)

f = Quadratic(eig_list=eig_list)
x0 = np.random.randn(d).astype(precision)

legend_list = list()
distances = list()
excess_losses = list()

alg = HeavyBall()

for momentum in {True, False}:
    for adaptive in {True, False}:
        x_list, f_list = alg.run(adaptive=adaptive, momentum=momentum, function=f, x0=x0, nb_steps=nb_steps)

        distance = [np.sum(x ** 2) for x in x_list]
        distances.append(distance)

        excess_loss = [fx - f.min_value for fx in f_list]
        excess_losses.append(excess_loss)

        legend_list.append(alg.name)

alg = ConjugateGradient()
x_list, f_list = alg.run(function=f, x0=x0, nb_steps=nb_steps)

distance = [np.sum(x ** 2) for x in x_list]
distances.append(distance)

excess_loss = [fx - f.min_value for fx in f_list]
excess_losses.append(excess_loss)

legend_list.append(alg.name)

fontsize = 28
plt.rcParams.update({'font.size': fontsize})

plt.figure(figsize=(15, 8))
# plt.title("Comparison of distances to optimum over {} steps of first order methods \n"
#           " applied on a {}-dimensional quadratic objective with condition number {}".format(nb_steps, d, kappa),
#           fontsize=24)
plt.yscale("log")
plt.plot(np.array(distances).T)
plt.xlabel("Iterations")
plt.ylabel("Distances to optimal")
plt.legend(legend_list, fontsize=fontsize-2)
plt.savefig("high_precision_figures/kappa_%s_d_%s_distances.png" % (kappa, d))

plt.figure(figsize=(15, 8))
# plt.title("Comparison of excess losses over {} steps of first order methods \n"
#           " applied on a {}-dimensional quadratic objective with condition number {}".format(nb_steps, d, kappa),
#           fontsize=24)
plt.yscale("log")
plt.plot(np.array(excess_losses).T)
plt.xlabel("Iterations")
plt.ylabel("Excess losses")
plt.legend(legend_list, fontsize=fontsize-2)
plt.savefig("high_precision_figures/kappa_%s_d_%s_losses.png" % (kappa, d))

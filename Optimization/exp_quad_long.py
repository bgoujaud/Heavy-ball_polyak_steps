import os
import numpy as np

from Optimization.Functions import Quadratic
from Optimization.compare_methods import compare_methods


d = 1000
kappa = 100000

log_L = 1
log_mu = log_L - np.log(kappa) / np.log(10)
eig_list = np.logspace(log_mu, log_L, d).astype(np.float128)
f = Quadratic(eig_list=eig_list)

nb_steps = 2*d
path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "paper",
                    "figures",
                    "quad_long",
                    "kappa_%s_d_%s" % (kappa, d)
                    )
compare_methods(dimension=d, function=f, nb_steps=nb_steps, path=path, seed=42)

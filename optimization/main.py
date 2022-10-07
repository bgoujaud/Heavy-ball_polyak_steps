import os
import sys
import numpy as np

from optimization.functions import Quadratic
from optimization.compare_methods import compare_methods

assert len(sys.argv) == 3

d = int(sys.argv[1])
if "." in sys.argv[2]:
    kappa = float(sys.argv[2])
else:
    kappa = int(sys.argv[2])

log_L = 1
log_mu = log_L - np.log(kappa) / np.log(10)
eig_list = np.logspace(log_mu, log_L, d).astype(np.float128)
f = Quadratic(eig_list=eig_list)

nb_steps = 2*d
path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "paper",
                    "figures",
                    "kappa_%s_d_%s" % (kappa, d)
                    )
compare_methods(dimension=d, function=f, nb_steps=nb_steps, path=path, seed=42)

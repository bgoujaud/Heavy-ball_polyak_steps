import numpy as np

from Optimization.Algorithms.algorithm import Algorithm


class ConjugateGradient(Algorithm):

    def __init__(self):
        super().__init__()

        self.name = "CG"

    def run(self, function, x0, nb_steps):
        x_list = [x0]
        r = -function.backward(x0)
        p = r

        for k in range(nb_steps):
            Ap = np.matmul(function.H, p)
            alpha = np.dot(r, r) / np.dot(p, Ap)
            x_list.append(x_list[-1] + alpha * p)
            r_new = r - alpha * Ap
            beta = np.dot(r_new, r_new) / np.dot(r, r)
            r = r_new
            p = r + beta * p

        f_list = [function.forward(x) for x in x_list]

        return x_list, f_list

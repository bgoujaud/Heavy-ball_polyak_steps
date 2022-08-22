import numpy as np
from scipy.stats import ortho_group

from Optimization.Functions.function import Function


class Quadratic(Function):

    def __init__(self, eig_list):
        super().__init__()

        self.dimension = len(eig_list)
        self.P = ortho_group.rvs(dim=self.dimension)
        self.H = np.matmul(self.P.T * np.array(eig_list), self.P)

        self.mu = min(eig_list)
        self.L = max(eig_list)
        self.min_value = 0

    def forward(self, x):
        # Compute 1/2 x.T H x
        return np.dot(self.backward(x), x) / 2

    def backward(self, x):
        # Compute H x
        return np.matmul(self.H, x)

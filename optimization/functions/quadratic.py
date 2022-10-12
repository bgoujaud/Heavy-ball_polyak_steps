import numpy as np
from scipy.stats import ortho_group

from .function import Function


class Quadratic(Function):
    """
    Quadratic objective class.
    A quadratic objective spectrum can either be determiniscally generated or randomly generated.
    """

    def __init__(self, eig_list=None, dimension=None, mu=None, L=None):
        """
        Generates the quadratic objective.
        Args:
            eig_list (Iterable): if provided, this list is the spectrum of H, hessian of the objective function.
                                 if None, this spectrum is randomly generated.
            dimension (int): if eig_list is None, dimension of the input space.
            mu (float): if eig_list is None,
                        the objective hessian spectrum is randomly generated enforcing it to be larger than mu.
            L (float): if eig_list is None,
                       the objective hessian spectrum is randomly generated enforcing it to be smaller than L.
        """
        super().__init__()

        # If eig_list if provided, compute mu and L.
        # If mu, L and dimension are provided, eig_list is generated.
        if eig_list is not None:
            self.mu = min(eig_list)
            self.L = max(eig_list)
        else:
            eig_list = mu * (L / mu) ** np.random.rand(dimension)
            self.mu = mu
            self.L = L

        # In both cases, a random rotation is generated.
        self.dimension = len(eig_list)
        self.P = ortho_group.rvs(dim=self.dimension)
        self.H = np.matmul(self.P.T * np.array(eig_list), self.P)
        self.argmin = np.zeros(self.dimension)
        self.min_value = 0

    def forward(self, x):
        # Compute 1/2 x.T H x
        return np.dot(self.backward(x), x) / 2

    def backward(self, x):
        # Compute H x
        return np.matmul(self.H, x)

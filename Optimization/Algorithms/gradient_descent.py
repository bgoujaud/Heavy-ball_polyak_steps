import numpy as np

from .heavy_ball import HeavyBall


class GradientDescent(HeavyBall):

    def __init__(self, parametrization="Constant"):
        super().__init__()

        self.name = "GD"

        if parametrization == "Constant":
            self.name += " with \nconstant step-size"
            self.set_parameters = lambda **params: (self.set_constant_step_size(**params), 0)
        elif parametrization == "PS":
            self.name += " with \nPolyak step-size"
            self.set_parameters = lambda **params: (self.set_polyak_step_size(**params), 0)
        elif parametrization == "PS variant":
            self.name += " with \nvariant of Polyak step-size"
            self.set_parameters = lambda **params: (self.set_polyak_step_size_variant(**params), 0)
        else:
            raise ValueError("\'parametrization\' must be either \'Constant\' or \'PS\' or \'PS variant\'. Got {}".format(
                parametrization
            ))

    @staticmethod
    def set_constant_step_size(function, **kwargs):

        step_size = 2 / (function.mu + function.L)

        return step_size

    @staticmethod
    def set_polyak_step_size(function, x, **kwargs):

        f_k = function.forward(x)
        f_star = function.min_value

        g_k = function.backward(x)

        step_size = (f_k - f_star) / np.sum(g_k ** 2)

        return step_size

    @staticmethod
    def set_polyak_step_size_variant(function, x, **kwargs):

        f_k = function.forward(x)
        f_star = function.min_value

        g_k = function.backward(x)

        step_size = 2 * (f_k - f_star) / np.sum(g_k ** 2)

        return step_size

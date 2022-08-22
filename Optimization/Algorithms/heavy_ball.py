from math import sqrt
import numpy as np

from Optimization.Algorithms.algorithm import Algorithm


class HeavyBall(Algorithm):

    def __init__(self):
        super().__init__()

        self.name = "Heavy-Ball"

    def step(self, function, x, x_pre, step_size, momentum_size):

        g_x = function.backward(x)

        x_new = x - (1 + momentum_size) * step_size * g_x + momentum_size * (x - x_pre)

        return x_new

    @staticmethod
    def set_constant_step_size(function):

        step_size = 2 / (function.mu + function.L)

        return step_size

    @staticmethod
    def set_polyak_step_size(function, x):

        step_size = 2 * (function.forward(x) - function.min_value) / np.sum(function.backward(x) ** 2)

        return step_size

    @staticmethod
    def set_constant_momentum_size(function):

        momentum_size = ((sqrt(function.L) - sqrt(function.mu)) / (sqrt(function.L) + sqrt(function.mu))) ** 2

        return momentum_size

    @staticmethod
    def set_adaptive_momentum_size(function, x, x_pre):

        f_k = function.forward(x)
        f_km1 = function.forward(x_pre)
        f_star = function.min_value

        g_k = function.backward(x)
        g_km1 = function.backward(x_pre)

        momentum_size = - ((f_k - f_star) * np.dot(g_k, g_km1)) / ((f_km1 - f_star) * np.dot(g_k, g_k) + (f_k - f_star) * np.dot(g_k, g_km1))

        return momentum_size

    def run(self, adaptive, momentum, function, x0, nb_steps):

        x_list = [x0]
        f_list = [function.forward(x0)]
        function.explored_min = f_list[-1]

        if adaptive:
            step_size = self.set_polyak_step_size(function=function, x=x_list[-1])
            self.name = "Adaptive"
        else:
            step_size = self.set_constant_step_size(function=function)
            self.name = "Constant"

        momentum_size = 0
        if momentum:
            self.name += " Heavy-Ball"
        else:
            self.name += " GD"

        x_new = self.step(function=function, x=x0, x_pre=x0, step_size=step_size,
                          momentum_size=momentum_size)
        x_list.append(x_new)
        f_list.append(function.forward(x_new))

        for _ in range(1, nb_steps):

            if adaptive:
                step_size = self.set_polyak_step_size(function=function, x=x_list[-1])
            else:
                step_size = self.set_constant_step_size(function=function)
            if momentum:
                if adaptive:
                    momentum_size = self.set_adaptive_momentum_size(function=function, x=x_list[-1], x_pre=x_list[-2])
                else:
                    momentum_size = self.set_constant_momentum_size(function=function)

            x_new = self.step(function=function, x=x_list[-1], x_pre=x_list[-2], step_size=step_size, momentum_size=momentum_size)
            x_list.append(x_new)
            f_list.append(function.forward(x_new))

        return x_list, f_list

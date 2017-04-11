from __future__  import division
from interpolation import *
import numpy as np
import itertools

class lagrange_2d(object):
    x_vals = None
    y_vals = None
    coordinates = None
    lagrange_x = None
    lagrange_y = None
    func_vals = None

    def __init__(self, x_vals, y_vals, func_vals=None):
        self.x_vals = x_vals
        self.y_vals = y_vals

        self.lagrange_x = lagrange_interpolation(x_vals)
        self.lagrange_y = lagrange_interpolation(y_vals)

        if func_vals != None:
            self.func_vals = np.array(func_vals).reshape(len(x_vals), len(y_vals))

    def fit_lagrange(self, func_vals):
        self.func_vals = np.array(func_vals).reshape(len(self.x_vals), len(self.y_vals))

    def evaluate(self, x, y, round_off=15):
        lag_x_eval = [func(x) for func in self.lagrange_x.basis_functions]
        lag_y_eval = [func(y) for func in self.lagrange_y.basis_functions]

        res = 0
        for i in range(len(self.x_vals)):
            for j in range(len(self.y_vals)):
                res += lag_x_eval[i]*lag_y_eval[j]*self.func_vals[i, j]

        return np.around(res, round_off)

    def evaluate_derivative_x(self, x, y, round_off=15):
        lag_derivative_x_eval = [func(x) for func in self.lagrange_x.deriv_basis]
        lag_y_eval = [func(y) for func in self.lagrange_y.basis_functions]
        
        res = 0
        for i in range(len(self.x_vals)):
            for j in range(len(self.y_vals)):
                res += lag_derivative_x_eval[i]*lag_y_eval[j]*self.func_vals[i, j]

        return np.around(res, round_off)

    def evaluate_derivative_y(self, x, y, round_off=15):
        lag_derivative_y_eval = [func(y) for func in self.lagrange_y.deriv_basis]
        lag_x_eval = [func(x) for func in self.lagrange_x.basis_functions]

        res = 0
        for i in range(len(self.x_vals)):
            for j in range(len(self.y_vals)):
                res += lag_x_eval[i]*lag_derivative_y_eval[j]*self.func_vals[i, j]
        
        return np.around(res, round_off)


if __name__ == '__main__':
    print('Running main!!!')
    from nodes_and_weights import *
    nodes_x = get_lobatto_points(64)
    nodes_y = get_lobatto_points(64)

    def func(x, y):
        return 1/(1 + 50*x**2 + 20*y**2)

    func_vals = [func(x, y) for x in nodes_x for y in nodes_y]

    lang_interpolation_2d = lagrange_2d(nodes_x, nodes_y, func_vals)

    import time
    t0 = time.time()
    for i in range(50):
        lang_interpolation_2d.evaluate(0.5, 0.5)
        lang_interpolation_2d.evaluate_derivative_x(0.5, 0.5)
        lang_interpolation_2d.evaluate_derivative_y(0.5, 0.5)
    t1 = time.time()

    print((t1 - t0)/50)
    
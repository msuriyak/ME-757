from __future__ import division
import numpy as np

def polyMult(coeff_a, coeff_b):
    coeff_a = np.array(coeff_a)
    coeff_b = np.array(coeff_b)

    res_coeff = np.zeros(coeff_a.shape[0] + coeff_b.shape[0] - 2  )
    for o1, i1 in enumerate(coeff_a):
        for o2, i2 in enumerate(coeff_b):
            res_coeff[o1 + o2] += i1*i2

    return res_coeff

def polyEval(coeff, x):
    coeff = np.array(coeff)
    x = np.array(x)
    res = 0
    for i in range(coeff.shape[0] - 1, -1, -1):
        res = x*res + coeff[i]
    return res


class lagrange_1d(object):
    coeff_matrix = None
    coeff = None

    def __init__(self, nodes):
        self.nodes = np.array(nodes)
        self.order = self.nodes.shape[0] - 1

        vandermonde = np.concatenate((np.ones((self.order + 1, 1)), self.nodes[:, np.newaxis]), axis=1)
        for i in range(self.order - 1):
            vandermonde = np.concatenate((vandermonde, (vandermonde[:, -1]*self.nodes)[:, np.newaxis]), axis=1)

        self.coeff_matrix = np.linalg.inv(vandermonde).T

    def fit(self, func_vals):
        self.func_vals = np.array(func_vals)[:, np.newaxis]
        self.coeff = np.dot(self.func_vals.T, self.coeff_matrix).reshape(self.order + 1)

    def evaluate(self, x):
        if self.coeff == None:
            msg = 'Use fit method before evaluate'
            raise ValueError(msg)

        x = np.array(x)
        return polyEval(self.coeff, x)

    def derivative(self, x, k=1):
        if self.coeff == None:
            msg = 'Use fit method before evaluate'
            raise ValueError(msg)

        x = np.array(x)

        temp = np.array([np.prod([i-j for j in range(k)]) for i in range(k, self.order + 1)])
        derivative_coeff = temp*self.coeff[k:]

        return polyEval(derivative_coeff, x)

class lagrange_2d(object):
    func_vals = None

    def __init__(self, nodes_x, nodes_y):
        self.nodes_x = np.array(nodes_x)
        self.nodes_y = np.array(nodes_y)
        self.order_x = self.nodes_x.shape[0] - 1
        self.order_y = self.nodes_y.shape[0] - 1

        self.lag_x = lagrange_1d(nodes_x)
        self.lag_y = lagrange_1d(nodes_y)

        self.basis = [(poly_x, poly_y) for poly_x in self.lag_x.coeff_matrix for poly_y in self.lag_y.coeff_matrix]

    def fit(self, func_vals):
        self.func_vals = np.array(func_vals)

    def evaluate(self, x, y):
        if self.func_vals == None:
            msg = 'Use fit method before evaluate'
            raise ValueError(msg)

        x = np.array(x)
        y = np.array(y)
        res = 0
        for i, item in enumerate(self.basis):
            res += polyEval(item[0], x)*polyEval(item[1], y)*self.func_vals[i]
        return res

    def evaluate_basis(self, x, y, i):
        x = np.array(x)
        y = np.array(y)
        return polyEval(self.basis[i][0], x)*polyEval(self.basis[i][1], y)

if __name__ == '__main__':
    print('Running main!!!')
    def func(x):
        return 1/(1 + 50*x**2)

    def func_1(x):
        return -100*x/(1 + 50*x**2)**2

    def func_2(x):
        return (100*(1 + 50*x**2) - 20000*x)/(1 + 50*x**2)**3


    from nodes_and_weights import *
    nodes_x = get_lobatto_points(36)

    x = np.array([0.5, 0.75])

    func_vals = func(nodes_x)
    lag_interpolation_1d = lagrange_1d(nodes_x)
    lag_interpolation_1d.fit(func_vals)
    print(lag_interpolation_1d.evaluate(x), func(x))
    print(lag_interpolation_1d.derivative(x), func_1(x))
    print(lag_interpolation_1d.derivative(x, k=2), func_2(x))

    import time
    t0 = time.time()
    for i in range(1000):
        lag_interpolation_1d.evaluate(0.5)
    t1 = time.time()
    print((t1 - t0)/1000)

    nodes_x = get_lobatto_points(10)
    nodes_y = get_lobatto_points(10)

    def func(x, y):
        return 1/(1 + 50*x**2 + 20*y**2)

    func_vals = [func(x, y) for x in nodes_x for y in nodes_y]

    lag_interpolation_2d = lagrange_2d(nodes_x, nodes_y)
    lag_interpolation_2d.fit(func_vals)
    print(lag_interpolation_2d.evaluate(x, x), func(x, x))

    t0 = time.time()
    for i in range(1000):
        lag_interpolation_2d.evaluate(0.5, 0.5)
    t1 = time.time()
    print((t1 - t0)/1000)
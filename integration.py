from __future__ import division
import numpy as np
import itertools

def integration(func, a, b, nodes, weights, round_off=15):
    nodes = 0.5*(b - a)*np.array(nodes) + 0.5*(b + a)
    weights = np.array(weights)
    return np.around(0.5*(b-a)*np.sum(weights*func(nodes)), round_off)

def integration_2d(func, a_x, b_x, a_y, b_y, nodes_x, nodes_y, weights_x, weights_y, round_off=15):
    nodes_x = 0.5*(b_x-a_x)*np.array(nodes_x).reshape(len(nodes_x), 1) + 0.5*(b_x + a_x)
    nodes_y = 0.5*(b_y-a_y)*np.array(nodes_y).reshape(len(nodes_y), 1) + 0.5*(b_y + a_y)
    weights_x = np.array(weights_x).reshape(len(weights_x), 1)
    weights_y = np.array(weights_y).reshape(len(weights_y), 1)

    cross_pdt = list(itertools.product(nodes_x, nodes_y))
    x_coordinates = np.array([item[0] for item in cross_pdt]).reshape(nodes_x.shape[0]*nodes_y.shape[0])
    y_coordinates = np.array([item[1] for item in cross_pdt]).reshape(nodes_x.shape[0]*nodes_y.shape[0])

    func_evals = func(x_coordinates, y_coordinates).reshape(nodes_x.shape[0], nodes_y.shape[0])

    return np.around(float(0.25*(b_y - a_y)*(b_x - a_x)*np.dot(weights_x.T, np.dot(func_evals, weights_y))[0, 0]), round_off)

if __name__ == '__main__':
    print('Running main!!!')
    def func(x, y):
        return 1/(1 + 50*x**2 + 20*y**2)

    def func2(x):
        return 1/(1 + 50*x**2)

    from nodes_and_weights import *
    nodes_x = get_lobatto_points(64)
    nodes_y = get_lobatto_points(64)
    weights_x = get_lobatto_weights(64)
    weights_y = get_lobatto_weights(64)

    x = get_legendre_points(64)
    y = get_legendre_points(64)

    from interpolation2d import *
    func_vals = [func(xi, yi) for xi in x for yi in y]
    lang_interpolation_2d = lagrange_2d(x, y, func_vals)

    from interpolation import *
    func_vals = np.array(func2(x))
    lang_interpolation = lagrange_1d(x, func_vals)

    print(integration_2d(func, 0, 1, 0, 1, nodes_x, nodes_y, weights_x, weights_y))
    print(integration_2d(lang_interpolation_2d.evaluate, 0, 1, 0, 1, nodes_x, nodes_y, weights_x, weights_y))

    print(integration(func2, 0, 1, nodes_x, weights_x))
    print(integration(lang_interpolation.evaluate, 0, 1, nodes_x, weights_x))

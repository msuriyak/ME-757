from __future__ import division
import numpy as np
from interpolation import *
from integration import *
from nodes_and_weights import *

def mass_matrix(N, quad_nodes='lobatto', quad_order=32):
    points = get_lobatto_points(N + 1)
    
    if quad_nodes =='legendre':
        nodes = get_legendre_points(quad_order + 1)
        weights = get_legendre_weights(quad_order + 1)

    else :
        nodes = get_lobatto_points(quad_order + 1)
        weights = get_lobatto_weights(quad_order + 1)

    lag_interpolation = lagrange_1d(points)

    result = np.zeros((N + 1, N + 1))
    for i in range(N+1) :
        for j in range(N+1) :
            if i > j :
                result[i, j] = result[j, i]
            else :
                def temp(x):
                    return lag_interpolation.basis_functions[i](x)*lag_interpolation.basis_functions[j](x)
                result[i, j] = integration(temp, -1, 1, nodes, weights)
    return result

if __name__ == '__main__':
    print('Running main!!!')
    print(mass_matrix(1, quad_order=2))
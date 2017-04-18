from __future__ import division
import numpy as np
from interpolation import *
from integration import *
from nodes_and_weights import *

##### kth 2d interpolation is pdt of k = j*(N_x + 1) + i
##### xi_k = (phi_i1*psi_i2) * (phi_j1*psi_j2)

def mass_matrix(N, *, quad_nodes='lobatto', quad_order=64):
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

def mass_matrix_2d(N_x, N_y, *, quad_nodes_x='lobatto', quad_nodes_y='lobatto', quad_order_x=64, quad_order_y=64):
    mass_x = mass_matrix(N_x, quad_nodes=quad_nodes_x, quad_order=quad_order_x)
    mass_y = mass_matrix(N_y, quad_nodes=quad_nodes_y, quad_order=quad_order_y)

    result = np.zeros(((N_x + 1)*(N_y + 1), (N_x + 1)*(N_y + 1)))

    for i1 in range(N_x + 1):
        for j1 in range(N_x + 1):
            for i2 in range(N_y + 1):
                for j2 in range(N_y + 1):
                    result[i2*N_x + i2 + i1, j2*N_x + j2 + j1] = mass_x[i1, j1]*mass_y[i2, j2]

    return result

def derivative_matrix(N, *, quad_nodes='lobatto', quad_order=64):
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
            def temp(x):
                return lag_interpolation.deriv_basis[i](x)*lag_interpolation.basis_functions[j](x)
            result[i, j] = integration(temp, -1, 1, nodes, weights)
    return result

def derivative_matrix_2d(N_x, N_y, *, quad_nodes_x='lobatto', quad_nodes_y='lobatto', quad_order_x=64, quad_order_y=64):
    derivative_x = derivative_matrix(N_x, quad_nodes=quad_nodes_x, quad_order=quad_order_x)
    derivative_y = derivative_matrix(N_y, quad_nodes=quad_nodes_y, quad_order=quad_order_y)

    mass_x = mass_matrix(N_x, quad_nodes=quad_nodes_x, quad_order=quad_order_x)
    mass_y = mass_matrix(N_y, quad_nodes=quad_nodes_y, quad_order=quad_order_y)

    deriv_x = np.zeros(((N_x + 1)*(N_y + 1), (N_x + 1)*(N_y + 1)))
    deriv_y = np.zeros(((N_x + 1)*(N_y + 1), (N_x + 1)*(N_y + 1)))

    for i1 in range(N_x + 1):
        for j1 in range(N_x + 1):
            for i2 in range(N_y + 1):
                for j2 in range(N_y + 1):
                    deriv_x[i2*N_x + i2 + i1, j2*N_x + j2 + j1] = mass_y[i2, j2]*derivative_x[i1, j1]

    for i1 in range(N_x + 1):
        for j1 in range(N_x + 1):
            for i2 in range(N_y + 1):
                for j2 in range(N_y + 1):
                    deriv_y[i2*N_x + i2 + i1, j2*N_x + j2 + j1] = mass_x[i1, j1]*derivative_y[i2, j2]

    return deriv_x, deriv_y

def flux_matrix(N):
    points = get_lobatto_points(N + 1)

    lag_interpolation = lagrange_1d(points)

    result = np.zeros((N + 1, N + 1))
    for i in range(N+1) :
        for j in range(N+1) :
            def temp(x):
                return lag_interpolation.basis_functions[i](x)*lag_interpolation.basis_functions[j](x)
            result[i, j] = temp(1) - temp(-1)
    return result 

def flux_matrix_2d(N_x, N_y, *, quad_nodes_x='lobatto', quad_nodes_y='lobatto', quad_order_x=64, quad_order_y=64):
    up    = np.zeros(((N_x + 1)*(N_y + 1), (N_x + 1)*(N_y + 1)))
    down  = np.zeros(((N_x + 1)*(N_y + 1), (N_x + 1)*(N_y + 1)))
    right = np.zeros(((N_x + 1)*(N_y + 1), (N_x + 1)*(N_y + 1)))
    left  = np.zeros(((N_x + 1)*(N_y + 1), (N_x + 1)*(N_y + 1)))

    mass_x = mass_matrix(N_x, quad_nodes=quad_nodes_x, quad_order=quad_order_x)
    mass_y = mass_matrix(N_y, quad_nodes=quad_nodes_y, quad_order=quad_order_y)

    i2=j2=0
    for i1 in range(N_x + 1) :
        for j1 in range(N_x + 1) :
            down[i2*N_x + i2 + i1, j2*N_x + j2 + j1] = mass_x[i1, j1]

    i2=j2=N_y
    for i1 in range(N_x + 1) :
        for j1 in range(N_x + 1) :
            up[i2*N_x + i2 + i1, j2*N_x + j2 + j1] = mass_x[i1, j1]

    i1=j1=0
    for i2 in range(N_y + 1) :
        for j2 in range(N_y + 1) :
            left[i2*N_x + i2 + i1, j2*N_x + j2 + j1] = mass_x[i2, j2]

    i1=j1=N_x
    for i2 in range(N_y + 1) :
        for j2 in range(N_y + 1) :
            right[i2*N_x + i2 + i1, j2*N_x + j2 + j1] = mass_x[i2, j2]
    
    return down, right, up, left


if __name__ == '__main__':
    print('Running main!!!')
    print(3*mass_matrix(1, quad_order=2))
    print(9*mass_matrix_2d(1, 1, quad_order_x=2, quad_order_y=2))

    print(2*derivative_matrix(1, quad_order=2))
    temp = derivative_matrix_2d(1, 1, quad_order_x=2, quad_order_y=2)
    print(6*temp[0])
    print(6*temp[1])

    print(flux_matrix(1))
    print(3*flux_matrix_2d(1, 1)[0])
    print(3*flux_matrix_2d(1, 1)[1])
    print(3*flux_matrix_2d(1, 1)[2])
    print(3*flux_matrix_2d(1, 1)[3])
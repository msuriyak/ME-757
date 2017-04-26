from __future__ import division
import numpy as np
import os
from tqdm import tqdm

from nodes_and_weights import *
from matrices import *

g = 9.81

class shallow_water(object):
    # order of the polynomial interpolation (used in p-refinement)
    N_x = None # order of interpolation in x-direction
    N_y = None # order of interpolation in y-direction
    nodes_x = None
    nodes_y = None

    # Numerical integration specifications
    q_n_x = None
    q_n_y = None
    q_o_x = None
    q_o_y = None

    # Elements in grid (used in h-refinement)
    Nex = None # number of elements in x-direction
    Ney = None # number of elements in y-direction
    
    # Grid co-ordinates
    X   = None # x-co-ordinate
    Y   = None # y-co-ordinate

    # Domain details
    x_start = None # X-coordinate starting
    x_end   = None # X-coordinate ending
    y_start = None # Y-coordinate starting
    y_end   = None # Y-coordinate ending
    delta_x = None # x element size
    delta_y = None # y-element size

    # Water height, velocity and mean height
    eta  = None # total height of water
    H    = None # total height to mean level
    u    = None # x-velocity
    v    = None # y-velocity
    etau = None # product of eta and u, conservation term on second equation
    etav = None # product of eta and v, conservation term on third equation

    H_x = None # partial derivative of H wrt x
    H_y = None # partial derivative of H wrt y

    mat_shape  = None # shape of all the above matrices

    # Simulation specs
    dt = None
    time_steps = None
    end_time = None

    # Matrices used in computation
    mass         = None
    mass_inverse = None
    derivative_x = None
    derivative_y = None
    flux_down    = None
    flux_right   = None
    flux_up      = None
    flux_left    = None

    def __init__(self, Nex, Ney, *, N=None, N_x=8, N_y=8, x_start=-5, x_end=5, y_start=-5, y_end=5, 
                 quad_nodes_x='lobatto', quad_nodes_y='lobatto', quad_order_x=16, quad_order_y=16):
        self.q_n_x = quad_nodes_x
        self.q_n_y = quad_nodes_y
        self.q_o_x = quad_order_x
        self.q_o_y = quad_order_y

        if N != None:
            self.N_x = N
            self.N_y = N
        else :
            self.N_x = N_x
            self.N_y = N_y

        self.Nex = Nex
        self.Ney = Ney

        self.x_start = x_start
        self.x_end   = x_end
        self.y_start = y_start
        self.y_end   = y_end

        self.delta_x = (x_end - x_start)/Nex
        self.delta_y = (y_end - y_start)/Ney

        self.nodes_x = get_lobatto_points(self.N_x + 1)*self.delta_x/2
        self.nodes_y = get_lobatto_points(self.N_y + 1)*self.delta_y/2

        x_mid = np.linspace(x_start, x_end, Nex + 1)[:-1] + self.delta_x/2
        y_mid = np.linspace(y_start, y_end, Ney + 1)[:-1] + self.delta_y/2

        x_temp = x_mid[0] + self.nodes_x
        for i in range(1, Nex):
            x_temp = np.concatenate((x_temp, self.nodes_x + x_mid[i]))

        y_temp = y_mid[0] + self.nodes_y
        for i in range(1, Ney):
            y_temp = np.concatenate((y_temp, self.nodes_y + y_mid[i]))
    
        self.X = np.array(x_temp.tolist()*Ney*(self.N_y + 1)).reshape((Ney*(self.N_y + 1), Nex*(self.N_x + 1)))
        self.Y = np.array(y_temp.tolist()*Nex*(self.N_x + 1)).reshape((Ney*(self.N_y + 1), Nex*(self.N_x + 1))).T

        self.mat_shape = self.X.shape

        del x_mid, x_temp, y_mid, y_temp, i

        self.eta = np.zeros(self.mat_shape)
        self.H = np.zeros(self.mat_shape)
        self.u = np.zeros(self.mat_shape)
        self.v = np.zeros(self.mat_shape)
        self.etau = np.zeros(self.mat_shape)
        self.etav = np.zeros(self.mat_shape)

        self.H_x = np.zeros(self.mat_shape)
        self.H_y = np.zeros(self.mat_shape)

        self.create_matrices()

    def setH(self, funcH, epsilon=1e-8):
        from itertools import product
        for i, j in product(np.arange(self.mat_shape[0]), np.arange(self.mat_shape[1])):
            self.H[i, j] = funcH(self.X[i, j], self.Y[i, j])
            self.H_x[i, j] = (funcH(self.X[i, j] + epsilon, self.Y[i, j]) - funcH(self.X[i, j] - epsilon, self.Y[i, j]))/(2*epsilon)
            self.H_y[i, j] = (funcH(self.X[i, j], self.Y[i, j] + epsilon) - funcH(self.X[i, j], self.Y[i, j] - epsilon))/(2*epsilon)

    def setEta_initial(self, funcEta):
        from itertools import product
        for i, j in product(np.arange(self.mat_shape[0]), np.arange(self.mat_shape[1])):
            self.eta[i, j] = funcEta(self.X[i, j], self.Y[i, j]) + self.H[i, j]

    def setVelocity_initial(self, func_u, func_v):
        from itertools import product
        for i, j in product(np.arange(self.mat_shape[0]), np.arange(self.mat_shape[1])):
            self.u[i, j] = func_u(self.X[i, j], self.Y[i, j])
            self.v[i, j] = func_v(self.X[i, j], self.Y[i, j])
            
            self.etau[i, j] = self.eta[i, j]*self.u[i, j]
            self.etav[i, j] = self.eta[i, j]*self.v[i, j]

    def setSimulationSpecs(self, dt, end_time):
        self.dt = dt
        self.end_time = end_time
        self.time_steps = end_time/dt

    def create_matrices(self):
        self.mass = mass_matrix_2d(self.N_x, self.N_y, quad_nodes_x=self.q_n_x, quad_nodes_y=self.q_n_y, 
                              quad_order_x=self.q_o_x, quad_order_y=self.q_o_y)

        self.mass_inverse = np.linalg.inv(self.mass)

        self.derivative_x, self.derivative_y = derivative_matrix_2d(self.N_x, self.N_y, quad_nodes_x=self.q_n_x, 
                                                                    quad_nodes_y=self.q_n_y, quad_order_x=self.q_o_x,
                                                                    quad_order_y=self.q_o_y)

        self.flux_down, self.flux_right, self.flux_up, self.flux_left = flux_matrix_2d(self.N_x, self.N_y, quad_nodes_x=self.q_n_x, 
                                                                                       quad_nodes_y=self.q_n_y, quad_order_x=self.q_o_x,
                                                                                       quad_order_y=self.q_o_y)


    def find_rates(self):
        etaRHS  = np.zeros(self.mat_shape)
        etauRHS = g*self.eta*self.H_x
        etavRHS = g*self.eta*self.H_y

        # Flux terms
        eta_flux_x  = self.eta*self.u
        eta_flux_y  = self.eta*self.v
        etau_flux_x = self.eta*self.u**2 + 0.5*g*self.eta**2
        etau_flux_y = self.eta*self.u*self.v
        etav_flux_x = self.eta*self.u*self.v
        etav_flux_y = self.eta*self.v**2 + 0.5*g*self.eta**2

        # Numerical flux calculation for internal nodes (Assuring continuity in interfaces)
        # i -> row , j -> column 
        eta_flux_x_num  = np.zeros(self.mat_shape)
        eta_flux_y_num  = np.zeros(self.mat_shape)
        etau_flux_x_num = np.zeros(self.mat_shape)
        etau_flux_y_num = np.zeros(self.mat_shape)
        etav_flux_x_num = np.zeros(self.mat_shape)
        etav_flux_y_num = np.zeros(self.mat_shape)

        for i in range(self.Ney + 1):
            for j in range(1, self.Nex + 1):
                for k in range(self.N_y + 1):
                    alpha = max(abs(self.u[i*(self.N_y + 1) + k, j*(self.N_x + 1)]) + np.sqrt(g*self.eta[i*(self.N_y + 1) + k, j*(self.N_x + 1)]), 
                                abs(self.u[i*(self.N_y + 1) + k, (j - 1)*(self.N_x + 1) + self.N_x]) + np.sqrt(g*self.eta[i*(self.N_y + 1) + k, (j - 1)*(self.N_x + 1) + self.N_x]))

                    temp =  0.5*(eta_flux_x[i*(self.N_y + 1) + k, j*(self.N_x + 1)] + eta_flux_x[i*(self.N_y + 1) + k, (j - 1)*(self.N_x + 1) + self.N_x])
                    temp += alpha*self.eta[i*(self.N_y + 1) + k, j*(self.N_x + 1)] - self.eta[i*(self.N_y + 1) + k, (j - 1)*(self.N_x + 1) + self.N_x] 
                    eta_flux_x_num[i*(self.N_y + 1) + k, j*(self.N_x + 1)] = eta_flux_x_num[i*(self.N_y + 1) + k, (j - 1)*(self.N_x + 1) + self.N_x] = temp

                    temp =  0.5*(etau_flux_x[i*(self.N_y + 1) + k, j*(self.N_x + 1)] + etau_flux_x[i*(self.N_y + 1) + k, (j - 1)*(self.N_x + 1) + self.N_x])
                    temp += alpha*self.etau[i*(self.N_y + 1) + k, j*(self.N_x + 1)] - self.etau[i*(self.N_y + 1) + k, (j - 1)*(self.N_x + 1) + self.N_x] 
                    etau_flux_x_num[i*(self.N_y + 1) + k, j*(self.N_x + 1)] = etau_flux_x_num[i*(self.N_y + 1) + k, (j - 1)*(self.N_x + 1) + self.N_x] = temp

                    temp =  0.5*(etav_flux_x[i*(self.N_y + 1) + k, j*(self.N_x + 1)] + etav_flux_x[i*(self.N_y + 1) + k, (j - 1)*(self.N_x + 1) + self.N_x])
                    temp += alpha*self.etav[i*(self.N_y + 1) + k, j*(self.N_x + 1)] - self.etav[i*(self.N_y + 1) + k, (j - 1)*(self.N_x + 1) + self.N_x] 
                    etav_flux_x_num[i*(self.N_y + 1) + k, j*(self.N_x + 1)] = etav_flux_x_num[i*(self.N_y + 1) + k, (j - 1)*(self.N_x + 1) + self.N_x] = temp

        for i in range(1, self.Ney + 1):
            for j in range(self.Nex + 1):
                for k in range(self.N_x + 1):
                    alpha = max(abs(self.v[i*(self.N_y + 1), j*(self.N_x + 1) + k]) + np.sqrt(g*self.eta[i*(self.N_y + 1), j*(self.N_x + 1) + k]), 
                                abs(self.v[(i - 1)*(self.N_y + 1) + self.N_y, j*(self.N_x + 1) + k]) + np.sqrt(g*self.eta[(i - 1)*(self.N_y + 1) + self.N_y, j*(self.N_x + 1) + k]))

                    temp =  0.5*(eta_flux_x[i*(self.N_y + 1), j*(self.N_x + 1) + k] + eta_flux_x[(i - 1)*(self.N_y + 1) + self.N_y, j*(self.N_x + 1) + k])
                    temp += alpha*eta[i*(self.N_y + 1), j*(self.N_x + 1) + k] - eta[(i - 1)*(self.N_y + 1) + self.N_y, j*(self.N_x + 1) + k] 
                    eta_flux_x_num[i*(self.N_y + 1), j*(self.N_x + 1) + k] = eta_flux_x_num[(i - 1)*(self.N_y + 1) + self.N_y, j*(self.N_x + 1) + k] = temp

                    temp =  0.5*(etau_flux_x[i*(self.N_y + 1), j*(self.N_x + 1) + k] + etau_flux_x[(i - 1)*(self.N_y + 1) + self.N_y, j*(self.N_x + 1) + k])
                    temp += alpha*etau[i*(self.N_y + 1), j*(self.N_x + 1) + k] - etau[(i - 1)*(self.N_y + 1) + self.N_y, j*(self.N_x + 1) + k] 
                    etau_flux_x_num[i*(self.N_y + 1), j*(self.N_x + 1) + k] = etau_flux_x_num[(i - 1)*(self.N_y + 1) + self.N_y, j*(self.N_x + 1) + k] = temp

                    temp =  0.5*(etav_flux_x[i*(self.N_y + 1), j*(self.N_x + 1) + k] + etav_flux_x[(i - 1)*(self.N_y + 1) + self.N_y, j*(self.N_x + 1) + k])
                    temp += alpha*etav[i*(self.N_y + 1), j*(self.N_x + 1) + k] - etav[(i - 1)*(self.N_y + 1) + self.N_y, j*(self.N_x + 1) + k] 
                    etav_flux_x_num[i*(self.N_y + 1), j*(self.N_x + 1) + k] = etav_flux_x_num[(i - 1)*(self.N_y + 1) + self.N_y, j*(self.N_x + 1) + k] = temp

        # Applying boundary conditions - Bath-tub model
        


if __name__ == '__main__':
    print('Running main!!!')
    a = shallow_water(10, 10, N=8)#, x_start=0, x_end=5, y_start=0, y_end=5)
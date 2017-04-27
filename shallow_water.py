from __future__ import division
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

from nodes_and_weights import *
from matrices import *

g = 9.81

'''
def matrix_vec_product_DSS(a, b):
    res = np.zeros(b.shape)
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            res[i, j, :] = np.dot(a, b[i, j, :])
    return res
'''

def RK4(x0, func_x_dot, h, func=None, change_variables=None):
    if func == None :
        def func():
            pass
    m = len(x0)
    k1_x_dot = func_x_dot()
    for i in range(m):
        change_variables[i](x0[i] + 0.5*h*k1_x_dot[i])
    func()

    k2_x_dot = func_x_dot()
    for i in range(m):
        change_variables[i](x0[i] + 0.5*h*k2_x_dot[i])
    func()

    k3_x_dot = func_x_dot()
    for i in range(m):
        change_variables[i](x0[i] + h*k3_x_dot[i])
    func()

    k4_x_dot = func_x_dot()
    for i in range(m):
        change_variables[i](x0[i] + (1/6.0)*h*(k1_x_dot[i] + 2*k2_x_dot[i] + 2*k3_x_dot[i] + k4_x_dot[i]))
    func()

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

    def __init__(self, Nex, Ney, *, N=None, N_x=8, N_y=8, x_start=-1, x_end=1, y_start=-1, y_end=1, 
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
        self.H = funcH(self.X, self.Y)
        self.H_x = (funcH(self.X + epsilon, self.Y) - funcH(self.X - epsilon, self.Y))/(2*epsilon)
        self.H_y = (funcH(self.X, self.Y + epsilon) - funcH(self.X, self.Y - epsilon))/(2*epsilon)
        
    def setEta_initial(self, funcEta):
        self.eta = funcEta(self.X, self.Y) + self.H

    def setVelocity_initial(self, func_u, func_v):
        self.u = func_u(self.X, self.Y)
        self.v = func_v(self.X, self.Y)

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
        Nex = self.Nex
        Ney = self.Ney
        N_x = self.N_x
        N_y = self.N_y
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

        # Numerical flux calculation for internal nodes
        # i -> row , j -> column 
        eta_flux_x_num  = np.zeros(self.mat_shape)
        eta_flux_y_num  = np.zeros(self.mat_shape)
        etau_flux_x_num = np.zeros(self.mat_shape)
        etau_flux_y_num = np.zeros(self.mat_shape)
        etav_flux_x_num = np.zeros(self.mat_shape)
        etav_flux_y_num = np.zeros(self.mat_shape)

        for i in range(Ney):
            for j in range(1, Nex):
                for k in range(N_y + 1):
                    alpha = max(abs(self.u[i*(N_y + 1) + k, j*(N_x + 1)]) + np.sqrt(g*self.eta[i*(N_y + 1) + k, j*(N_x + 1)]), 
                                abs(self.u[i*(N_y + 1) + k, (j - 1)*(N_x + 1) + N_x]) + np.sqrt(g*self.eta[i*(N_y + 1) + k, (j - 1)*(N_x + 1) + N_x]))

                    temp =  0.5*(eta_flux_x[i*(N_y + 1) + k, j*(N_x + 1)] + eta_flux_x[i*(N_y + 1) + k, (j - 1)*(N_x + 1) + N_x])
                    temp += 0.5*alpha*(self.eta[i*(N_y + 1) + k, j*(N_x + 1)] - self.eta[i*(N_y + 1) + k, (j - 1)*(N_x + 1) + N_x]) 
                    eta_flux_x_num[i*(N_y + 1) + k, j*(N_x + 1)] = eta_flux_x_num[i*(N_y + 1) + k, (j - 1)*(N_x + 1) + N_x] = temp

                    temp =  0.5*(etau_flux_x[i*(N_y + 1) + k, j*(N_x + 1)] + etau_flux_x[i*(N_y + 1) + k, (j - 1)*(N_x + 1) + N_x])
                    temp += 0.5*alpha*(self.etau[i*(N_y + 1) + k, j*(N_x + 1)] - self.etau[i*(N_y + 1) + k, (j - 1)*(N_x + 1) + N_x]) 
                    etau_flux_x_num[i*(N_y + 1) + k, j*(N_x + 1)] = etau_flux_x_num[i*(N_y + 1) + k, (j - 1)*(N_x + 1) + N_x] = temp

                    temp =  0.5*(etav_flux_x[i*(N_y + 1) + k, j*(N_x + 1)] + etav_flux_x[i*(N_y + 1) + k, (j - 1)*(N_x + 1) + N_x])
                    temp += 0.5*alpha*(self.etav[i*(N_y + 1) + k, j*(N_x + 1)] - self.etav[i*(N_y + 1) + k, (j - 1)*(N_x + 1) + N_x]) 
                    etav_flux_x_num[i*(N_y + 1) + k, j*(N_x + 1)] = etav_flux_x_num[i*(N_y + 1) + k, (j - 1)*(N_x + 1) + N_x] = temp
        #print('First x flux')
        #print(eta_flux_x_num)
        #print(etau_flux_x_num)
        #print(etav_flux_x_num, '\n\n\n\n')

        for i in range(1, Ney):
            for j in range(Nex):
                for k in range(N_x + 1):
                    alpha = max(abs(self.v[i*(N_y + 1), j*(N_x + 1) + k]) + np.sqrt(g*self.eta[i*(N_y + 1), j*(N_x + 1) + k]), 
                                abs(self.v[(i - 1)*(N_y + 1) + N_y, j*(N_x + 1) + k]) + np.sqrt(g*self.eta[(i - 1)*(N_y + 1) + N_y, j*(N_x + 1) + k]))

                    temp =  0.5*(eta_flux_y[i*(N_y + 1), j*(N_x + 1) + k] + eta_flux_y[(i - 1)*(N_y + 1) + N_y, j*(N_x + 1) + k])
                    temp += 0.5*alpha*(self.eta[i*(N_y + 1), j*(N_x + 1) + k] - self.eta[(i - 1)*(N_y + 1) + N_y, j*(N_x + 1) + k])
                    eta_flux_y_num[i*(N_y + 1), j*(N_x + 1) + k] = eta_flux_y_num[(i - 1)*(N_y + 1) + N_y, j*(N_x + 1) + k] = temp

                    temp =  0.5*(etau_flux_y[i*(N_y + 1), j*(N_x + 1) + k] + etau_flux_y[(i - 1)*(N_y + 1) + N_y, j*(N_x + 1) + k])
                    temp += 0.5*alpha*(self.etau[i*(N_y + 1), j*(N_x + 1) + k] - self.etau[(i - 1)*(N_y + 1) + N_y, j*(N_x + 1) + k])
                    etau_flux_y_num[i*(N_y + 1), j*(N_x + 1) + k] = etau_flux_y_num[(i - 1)*(N_y + 1) + N_y, j*(N_x + 1) + k] = temp

                    temp =  0.5*(etav_flux_y[i*(N_y + 1), j*(N_x + 1) + k] + etav_flux_y[(i - 1)*(N_y + 1) + N_y, j*(N_x + 1) + k])
                    temp += 0.5*alpha*(self.etav[i*(N_y + 1), j*(N_x + 1) + k] - self.etav[(i - 1)*(N_y + 1) + N_y, j*(N_x + 1) + k])
                    etav_flux_y_num[i*(N_y + 1), j*(N_x + 1) + k] = etav_flux_y_num[(i - 1)*(N_y + 1) + N_y, j*(N_x + 1) + k] = temp
        #print('First y flux')
        #print(eta_flux_y_num)
        #print(etau_flux_y_num)
        #print(etav_flux_y_num, '\n\n\n\n')

        # Applying boundary conditions - Bath-tub model
        for i in range(Ney):
            for k in range(N_y + 1):
                eta_flux_x_num[i*(N_y + 1) + k, 0] = 0 
                eta_flux_x_num[i*(N_y + 1) + k, Nex*(N_x + 1) - 1] = 0

                alpha1 = abs(self.u[i*(N_y + 1) + k, 0]) + np.sqrt(g*self.eta[i*(N_y + 1) + k, 0])
                alpha2 = abs(self.u[i*(N_y + 1) + k, Nex*(N_x + 1) - 1]) + np.sqrt(g*self.eta[i*(N_y + 1) + k, Nex*(N_x + 1) - 1])
                etau_flux_x_num[i*(N_y + 1) + k, 0] = etau_flux_x[i*(N_y + 1) + k, 0] - alpha1*(self.etau[i*(N_y + 1) + k, 0])
                etau_flux_x_num[i*(N_y + 1) + k, Nex*(N_x + 1) - 1] = etau_flux_x[i*(N_y + 1) + k, Nex*(N_x + 1) - 1] + alpha2*(self.u[i*(N_y + 1) + k, Nex*(N_x + 1) - 1])

                etav_flux_x_num[i*(N_y + 1) + k, 0] = 0
                etav_flux_x_num[i*(N_y + 1) + k, Nex*(N_x + 1) - 1] = 0
        #print('boundary x flux')
        #print(eta_flux_x_num)
        #print(etau_flux_x_num)
        #print(etav_flux_x_num, '\n\n\n\n')
         
        for j in range(Nex):
            for k in range(N_x + 1):
                eta_flux_y_num[0, j*(N_x + 1) + k] = 0
                eta_flux_y_num[Ney*(N_y + 1) - 1, j*(N_x + 1) + k] = 0

                etau_flux_y_num[0, j*(N_x + 1) + k] = 0
                etau_flux_y_num[Ney*(N_y + 1) - 1, j*(N_x + 1) + k] = 0

                alpha1 = abs(self.v[0, j*(N_x + 1) + k]) + np.sqrt(g*self.eta[0, j*(N_x + 1) + k])
                alpha2 = abs(self.v[Ney*(N_y + 1) - 1, j*(N_x + 1) + k]) + np.sqrt(g*self.eta[Ney*(N_y + 1) - 1, j*(N_x + 1) + k])
                etav_flux_y_num[0, j*(N_x + 1) + k] = etav_flux_y[0, j*(N_x + 1) + k] - alpha1*(self.etav[0, j*(N_x + 1) + k])
                etav_flux_y_num[Ney*(N_y + 1) - 1, j*(N_x + 1) + k] = etav_flux_y[Ney*(N_y + 1) - 1, j*(N_x + 1) + k] + alpha2*(self.v[Ney*(N_y + 1) - 1, j*(N_x + 1) + k])
        #print('boundary y flux')
        #print(eta_flux_y_num)
        #print(etau_flux_y_num)
        #print(etav_flux_y_num, '\n\n\n\n')

        eta_dot = np.zeros(self.mat_shape)
        etau_dot = np.zeros(self.mat_shape)
        etav_dot = np.zeros(self.mat_shape)

        derivative_x = 0.5*self.delta_y*self.derivative_x
        derivative_y = 0.5*self.delta_x*self.derivative_y
        
        flux_down  =  0.5*self.delta_x*self.flux_down
        flux_right = -0.5*self.delta_y*self.flux_right
        flux_up    = -0.5*self.delta_x*self.flux_up
        flux_left  =  0.5*self.delta_y*self.flux_left


        for i in range(Ney):
            for j in range(Nex):
                i_start = i*(N_y + 1)
                i_end   = N_y + 1 + i*(N_y + 1)  
                j_start = j*(N_x + 1)
                j_end   = N_x + 1 + j*(N_x + 1)
                etaRHS[i_start:i_end, j_start:j_end]  += np.dot(derivative_x, eta_flux_x[i_start:i_end, j_start:j_end].reshape((N_x+1)*(N_y+1))).reshape((N_y+1), (N_x+1))
                etaRHS[i_start:i_end, j_start:j_end]  += np.dot(derivative_y, eta_flux_y[i_start:i_end, j_start:j_end].reshape((N_x+1)*(N_y+1))).reshape((N_y+1), (N_x+1))
                etaRHS[i_start:i_end, j_start:j_end]  += np.dot(flux_down,    eta_flux_x_num[i_start:i_end, j_start:j_end].reshape((N_x+1)*(N_y+1))).reshape((N_y+1), (N_x+1))
                etaRHS[i_start:i_end, j_start:j_end]  += np.dot(flux_right,   eta_flux_x_num[i_start:i_end, j_start:j_end].reshape((N_x+1)*(N_y+1))).reshape((N_y+1), (N_x+1))
                etaRHS[i_start:i_end, j_start:j_end]  += np.dot(flux_up,      eta_flux_x_num[i_start:i_end, j_start:j_end].reshape((N_x+1)*(N_y+1))).reshape((N_y+1), (N_x+1))
                etaRHS[i_start:i_end, j_start:j_end]  += np.dot(flux_left,    eta_flux_x_num[i_start:i_end, j_start:j_end].reshape((N_x+1)*(N_y+1))).reshape((N_y+1), (N_x+1))
                
                etauRHS[i_start:i_end, j_start:j_end] += np.dot(derivative_x, etau_flux_x[i_start:i_end, j_start:j_end].reshape((N_x+1)*(N_y+1))).reshape((N_y+1), (N_x+1))
                etauRHS[i_start:i_end, j_start:j_end] += np.dot(derivative_y, etau_flux_y[i_start:i_end, j_start:j_end].reshape((N_x+1)*(N_y+1))).reshape((N_y+1), (N_x+1))
                etauRHS[i_start:i_end, j_start:j_end] += np.dot(flux_down,    etau_flux_y_num[i_start:i_end, j_start:j_end].reshape((N_x+1)*(N_y+1))).reshape((N_y+1), (N_x+1))
                etauRHS[i_start:i_end, j_start:j_end] += np.dot(flux_right,   etau_flux_x_num[i_start:i_end, j_start:j_end].reshape((N_x+1)*(N_y+1))).reshape((N_y+1), (N_x+1))
                etauRHS[i_start:i_end, j_start:j_end] += np.dot(flux_up,      etau_flux_y_num[i_start:i_end, j_start:j_end].reshape((N_x+1)*(N_y+1))).reshape((N_y+1), (N_x+1))
                etauRHS[i_start:i_end, j_start:j_end] += np.dot(flux_left,    etau_flux_x_num[i_start:i_end, j_start:j_end].reshape((N_x+1)*(N_y+1))).reshape((N_y+1), (N_x+1))

                etavRHS[i_start:i_end, j_start:j_end] += np.dot(derivative_x, etav_flux_x[i_start:i_end, j_start:j_end].reshape((N_x+1)*(N_y+1))).reshape((N_y+1), (N_x+1))
                etavRHS[i_start:i_end, j_start:j_end] += np.dot(derivative_y, etav_flux_y[i_start:i_end, j_start:j_end].reshape((N_x+1)*(N_y+1))).reshape((N_y+1), (N_x+1))
                etavRHS[i_start:i_end, j_start:j_end] += np.dot(flux_down,    etav_flux_y_num[i_start:i_end, j_start:j_end].reshape((N_x+1)*(N_y+1))).reshape((N_y+1), (N_x+1))
                etavRHS[i_start:i_end, j_start:j_end] += np.dot(flux_right,   etav_flux_x_num[i_start:i_end, j_start:j_end].reshape((N_x+1)*(N_y+1))).reshape((N_y+1), (N_x+1))
                etavRHS[i_start:i_end, j_start:j_end] += np.dot(flux_up,      etav_flux_y_num[i_start:i_end, j_start:j_end].reshape((N_x+1)*(N_y+1))).reshape((N_y+1), (N_x+1))
                etavRHS[i_start:i_end, j_start:j_end] += np.dot(flux_left,    etav_flux_x_num[i_start:i_end, j_start:j_end].reshape((N_x+1)*(N_y+1))).reshape((N_y+1), (N_x+1))

                eta_dot[i_start:i_end, j_start:j_end]  = (4)/(self.delta_x*self.delta_y)*np.dot(self.mass_inverse, etaRHS[i_start:i_end, j_start:j_end].reshape((N_x+1)*(N_y+1))).reshape((N_y+1), (N_x+1))
                etau_dot[i_start:i_end, j_start:j_end] = (4)/(self.delta_x*self.delta_y)*np.dot(self.mass_inverse, etauRHS[i_start:i_end, j_start:j_end].reshape((N_x+1)*(N_y+1))).reshape((N_y+1), (N_x+1))
                etav_dot[i_start:i_end, j_start:j_end] = (4)/(self.delta_x*self.delta_y)*np.dot(self.mass_inverse, etavRHS[i_start:i_end, j_start:j_end].reshape((N_x+1)*(N_y+1))).reshape((N_y+1), (N_x+1))

        return eta_dot, etau_dot, etav_dot

    def compute_velocities(self):
        self.u = self.etau/self.eta
        self.v = self.etav/self.eta

    def change_variables(self):
        def change_eta(eta):
            self.eta = eta
        def change_etau(etau):
            self.etau = etau
        def change_etav(etav):
            self.etav = etav
        return [change_eta, change_etau, change_etav]

    def make_etau_and_etav(self):
        self.etau = self.eta*self.u
        self.etav = self.eta*self.v

    '''
    def create_fig(self):
        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        self.surf = self.ax.plot_surface(model.X, model.Y, model.eta - model.H, cmap=cm.coolwarm)

    def animate(self, i):
    '''

    def solve(self):
        time = [0]
        for i in range(int(self.time_steps) + 1):
            print('Time t = %2.4f, Courant number = %2.4f' %(time[-1], max(abs(self.u).max(), abs(self.v).max())*self.dt*(1/self.delta_x + 1/self.delta_y)))
            RK4([self.eta, self.etau, self.etav], self.find_rates,self. dt, self.compute_velocities, self.change_variables())
            time.append(time[-1] + self.dt)
            if i%20 == 0:
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                surf = ax.plot_surface(self.X, self.Y, self.eta - self.H, cmap=cm.coolwarm)
                plt.show()

if __name__ == '__main__':
    print('Running main!!!')
    def funcH(x, y):
        return 10*np.ones(x.shape)

    def func_u(x, y):
        return np.zeros(x.shape)

    def func_v(x, y):
        return np.zeros(x.shape)

    def funcEta(x, y):
        return 0.1*np.exp(-(x**2 + y**2)/10)

    model = shallow_water(10, 10, N=2)
    model.setH(funcH)
    model.setEta_initial(funcEta)
    model.setVelocity_initial(func_u, func_v)
    model.setSimulationSpecs(1e-4, 0.025)
    model.make_etau_and_etav()
    
    model.solve()

'''
model = shallow_water.shallow_water(30, 30, N=1)
model.setH(funcH)
model.setEta_initial(funcEta)
model.setVelocity_initial(func_u, func_v)
model.setSimulationSpecs(1e-4, 0.5)
model.make_etau_and_etav()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(model.X, model.Y, model.eta - model.H, cmap=cm.coolwarm)
plt.show()
'''
from __future__ import division
import numpy as np
from numpy.matlib import repmat

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

from nodes_and_weights import *
from matrices import *

g = 9.81

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
            N_x = N
            N_y = N
        
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

        nodes_x = get_lobatto_points(N_x + 1)*self.delta_x/2
        nodes_y = get_lobatto_points(N_y + 1)*self.delta_y/2

        x_mid = np.linspace(x_start, x_end, Nex + 1)[:-1] + self.delta_x/2
        y_mid = np.linspace(y_start, y_end, Ney + 1)[:-1] + self.delta_y/2

        x_temp = repmat(nodes_x, N_y + 1, 1)
        y_temp = repmat(nodes_y[:, np.newaxis], 1, N_x + 1)
    
        X = np.zeros((Ney, Nex, N_y + 1, N_x + 1))
        Y = np.zeros((Ney, Nex, N_y + 1, N_x + 1))

        for i in range(Ney):
            for j in range(Nex):
                X[i, j] = x_temp + x_mid[j]
                Y[i, j] = y_temp + y_mid[i]

        self.X = X
        self.Y = Y

        self.mat_shape = (Ney, Nex, N_y + 1, N_x + 1)
        self.mat_shape_modified = (Ney*(N_y + 1), Nex*(N_x + 1))

        del x_mid, x_temp, X, y_mid, y_temp, Y, i, nodes_x, nodes_y

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
        X = self.X.reshape(self.mat_shape_modified)
        Y = self.Y.reshape(self.mat_shape_modified)
        self.H = funcH(X, Y).reshape(self.mat_shape)
        self.H_x = ((funcH(X + epsilon, Y) - funcH(X - epsilon, Y))/(2*epsilon)).reshape(self.mat_shape)
        self.H_y = ((funcH(X, Y + epsilon) - funcH(X, Y - epsilon))/(2*epsilon)).reshape(self.mat_shape)
        
    def setEta_initial(self, funcEta):
        X = self.X.reshape(self.mat_shape_modified)
        Y = self.Y.reshape(self.mat_shape_modified)
        self.eta = (funcEta(X, Y) + self.H.reshape(self.mat_shape_modified)).reshape(self.mat_shape)

    def setVelocity_initial(self, func_u, func_v):
        X = self.X.reshape(self.mat_shape_modified)
        Y = self.Y.reshape(self.mat_shape_modified)
        self.u = func_u(X, Y).reshape(self.mat_shape)
        self.v = func_v(X, Y).reshape(self.mat_shape)

    def setSimulationSpecs(self, dt, end_time):
        self.dt = dt
        self.end_time = end_time
        self.time_steps = end_time/dt

    def return_XnY(self):
        return self.X.reshape(self.mat_shape_modified), self.Y.reshape(self.mat_shape_modified)

    def return_h(self):
        return (self.eta - self.H).reshape(self.mat_shape_modified)

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

        etaRHS  = np.zeros((Ney, Nex, (N_y+1)*(N_x+1), 1))
        etauRHS = np.zeros((Ney, Nex, (N_y+1)*(N_x+1), 1))
        etavRHS = np.zeros((Ney, Nex, (N_y+1)*(N_x+1), 1))

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
                    alpha = max(abs(self.u[i, j  , k, 0])   + np.sqrt(g*self.eta[i, j  , k, 0]), 
                                abs(self.u[i, j-1, k, N_x]) + np.sqrt(g*self.eta[i, j-1, k, N_x]))

                    temp =  0.5*(eta_flux_x[i, j  , k, 0] + eta_flux_x[i, j-1, k, N_x])
                    temp += 0.5*alpha*(self.eta[i, j  , k, 0] - self.eta[i, j-1, k, N_x]) 
                    eta_flux_x_num[i, j, k, 0] = eta_flux_x_num[i, j-1, k, N_x] = temp

                    temp =  0.5*(etau_flux_x[i, j, k, 0] + etau_flux_x[i, j-1, k, N_x])
                    temp += 0.5*alpha*(self.etau[i, j, k, 0] - self.etau[i, j-1, k, N_x]) 
                    etau_flux_x_num[i, j, k, 0] = etau_flux_x_num[i, j-1, k, N_x] = temp

                    temp =  0.5*(etav_flux_x[i, j, k, 0] + etav_flux_x[i, j-1, k, N_x])
                    temp += 0.5*alpha*(self.etav[i, j, k, 0] - self.etav[i, j-1, k, N_x]) 
                    etav_flux_x_num[i, j, k, 0] = etav_flux_x_num[i, j-1, k, N_x] = temp
    
        for i in range(1, Ney):
            for j in range(Nex):
                for k in range(N_x + 1):
                    alpha = max(abs(self.v[i, j, 0, k]) + np.sqrt(g*self.eta[i, j, 0, k]), 
                                abs(self.v[i-1, j, N_y, k]) + np.sqrt(g*self.eta[i-1, j, N_y, k]))

                    temp =  0.5*(eta_flux_y[i, j, 0, k] + eta_flux_y[i-1, j, N_y, k])
                    temp += 0.5*alpha*(self.eta[i, j, 0, k] - self.eta[i-1, j, N_y, k])
                    eta_flux_y_num[i, j, 0, k] = eta_flux_y_num[i-1, j, N_y, k] = temp

                    temp =  0.5*(etau_flux_y[i, j, 0, k] + etau_flux_y[i-1, j, N_y, k])
                    temp += 0.5*alpha*(self.etau[i, j, 0, k] - self.etau[i-1, j, N_y, k])
                    etau_flux_y_num[i, j, 0, k] = etau_flux_y_num[i-1, j, N_y, k] = temp

                    temp =  0.5*(etav_flux_y[i, j, 0, k] + etav_flux_y[i-1, j, N_y, k])
                    temp += 0.5*alpha*(self.etav[i, j, 0, k] - self.etav[i-1, j, N_y, k])
                    etav_flux_y_num[i, j, 0, k] = etav_flux_y_num[i-1, j, N_y, k] = temp
        
        # Applying boundary conditions - Bath-tub model
        for i in range(Ney):
            for k in range(N_y + 1):
                eta_flux_x_num[i, 0, k, 0] = 0 
                eta_flux_x_num[i, Nex-1, k, N_x] = 0

                alpha1 = abs(self.u[i, 0, k, 0]) + np.sqrt(g*self.eta[i, 0, k, 0])
                alpha2 = abs(self.u[i, Nex-1, k, N_x]) + np.sqrt(g*self.eta[i, Nex-1, k, N_x])
                etau_flux_x_num[i, 0, k, 0] = etau_flux_x[i, 0, k, 0] - alpha1*(self.etau[i, 0, k, 0])
                etau_flux_x_num[i, Nex-1, k, N_x] = etau_flux_x[i, Nex-1, k, N_x] + alpha2*(self.u[i, Nex-1, k, N_x])

                etav_flux_x_num[i, 0, k, 0] = 0
                etav_flux_x_num[i, Nex-1, k, N_x] = 0
          
        for j in range(Nex):
            for k in range(N_x + 1):
                eta_flux_y_num[0, j, 0, k] = 0
                eta_flux_y_num[Ney-1, j, N_y, k] = 0

                etau_flux_y_num[0, j, 0, k] = 0
                etau_flux_y_num[Ney-1, j, N_y, k] = 0

                alpha1 = abs(self.v[0, j, 0, k]) + np.sqrt(g*self.eta[0, j, 0, k])
                alpha2 = abs(self.v[Ney-1, j, N_y, k]) + np.sqrt(g*self.eta[Ney-1, j, N_y, k])
                etav_flux_y_num[0, j, 0, k] = etav_flux_y[0, j, 0, k] - alpha1*(self.etav[0, j, 0, k])
                etav_flux_y_num[Ney-1, j, N_y, k] = etav_flux_y[Ney-1, j, N_y, k] + alpha2*(self.v[Ney-1, j, N_y, k])
        
        eta_dot = np.zeros((Ney, Nex, (N_y+1)*(N_x+1), 1))
        etau_dot = np.zeros((Ney, Nex, (N_y+1)*(N_x+1), 1))
        etav_dot = np.zeros((Ney, Nex, (N_y+1)*(N_x+1), 1))

        derivative_x = 0.5*self.delta_y*self.derivative_x
        derivative_y = 0.5*self.delta_x*self.derivative_y
        
        flux_down  =  0.5*self.delta_x*self.flux_down
        flux_right = -0.5*self.delta_y*self.flux_right
        flux_up    = -0.5*self.delta_x*self.flux_up
        flux_left  =  0.5*self.delta_y*self.flux_left

        mass_inverse = (4)/(self.delta_x*self.delta_y)*self.mass_inverse

        element_shape = ((N_y+1)*(N_x+1), 1)

        for i in range(Ney):
            for j in range(Nex):            
                etaRHS[i, j]  += np.dot(derivative_x, eta_flux_x[i, j]    .reshape(element_shape))
                etaRHS[i, j]  += np.dot(derivative_y, eta_flux_y[i, j]    .reshape(element_shape))
                etaRHS[i, j]  += np.dot(flux_down,    eta_flux_y_num[i, j].reshape(element_shape))
                etaRHS[i, j]  += np.dot(flux_right,   eta_flux_x_num[i, j].reshape(element_shape))
                etaRHS[i, j]  += np.dot(flux_up,      eta_flux_y_num[i, j].reshape(element_shape))
                etaRHS[i, j]  += np.dot(flux_left,    eta_flux_x_num[i, j].reshape(element_shape))

                etauRHS[i, j] += np.dot(derivative_x, etau_flux_x[i, j]    .reshape(element_shape))
                etauRHS[i, j] += np.dot(derivative_y, etau_flux_y[i, j]    .reshape(element_shape))
                etauRHS[i, j] += np.dot(flux_down,    etau_flux_y_num[i, j].reshape(element_shape))
                etauRHS[i, j] += np.dot(flux_right,   etau_flux_x_num[i, j].reshape(element_shape))
                etauRHS[i, j] += np.dot(flux_up,      etau_flux_y_num[i, j].reshape(element_shape))
                etauRHS[i, j] += np.dot(flux_left,    etau_flux_x_num[i, j].reshape(element_shape))

                etavRHS[i, j] += np.dot(derivative_x, etav_flux_x[i, j]    .reshape(element_shape))
                etavRHS[i, j] += np.dot(derivative_y, etav_flux_y[i, j]    .reshape(element_shape))
                etavRHS[i, j] += np.dot(flux_down,    etav_flux_y_num[i, j].reshape(element_shape))
                etavRHS[i, j] += np.dot(flux_right,   etav_flux_x_num[i, j].reshape(element_shape))
                etavRHS[i, j] += np.dot(flux_up,      etav_flux_y_num[i, j].reshape(element_shape))
                etavRHS[i, j] += np.dot(flux_left,    etav_flux_x_num[i, j].reshape(element_shape))

                eta_dot[i, j]  = np.dot(mass_inverse, etaRHS[i, j]) 
                etau_dot[i, j] = np.dot(mass_inverse, etauRHS[i, j])
                etav_dot[i, j] = np.dot(mass_inverse, etavRHS[i, j])
        
        eta_dot = eta_dot.reshape(self.mat_shape)
        etau_dot = etau_dot.reshape(self.mat_shape)
        etav_dot = etav_dot.reshape(self.mat_shape)

        # Cleaning workspace
        del etaRHS, etauRHS, etavRHS 
        del eta_flux_x_num, eta_flux_y_num, etau_flux_x_num, etau_flux_y_num, etav_flux_x_num, etav_flux_y_num
        del eta_flux_x, eta_flux_y, etau_flux_x, etau_flux_y, etav_flux_x, etav_flux_y
        del Nex, Ney, N_y, N_x
        del derivative_x, derivative_y
        del flux_down, flux_right, flux_up, flux_left
        del mass_inverse
        del i, j, k, temp, alpha, alpha1, alpha2

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

    def solve(self):
        time = [0]
        for i in range(int(self.time_steps) + 1):
            print('Time t = %2.4f, Courant number = %2.4f' %(time[-1], max(abs(self.u).max(), abs(self.v).max())*self.dt*(1/self.delta_x + 1/self.delta_y)))
            RK4([self.eta, self.etau, self.etav], self.find_rates,self. dt, self.compute_velocities, self.change_variables())
            time.append(time[-1] + self.dt)
            if i%20 == 0:
                X, Y = self.return_XnY()
                h = self.return_h()

                fig = plt.figure()
                ax = fig.gca(projection='3d')
                scat = ax.scatter(X, Y, h, cmap=cm.coolwarm)
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

    model = shallow_water(10, 10, N=4)
    model.setH(funcH)
    model.setEta_initial(funcEta)
    model.setVelocity_initial(func_u, func_v)
    model.setSimulationSpecs(1e-4, 0.025)
    model.make_etau_and_etav()
    
    model.solve()
from __future__ import division
import numpy as np
import os
from tqdm import tqdm

from interpolation import *
from integration import *
from nodes_and_weights import *
from matrices import *

class shallow_water(object):
    # order of the polynomial interpolation (used in p-refinement)
    N_x = None # order of interpolation in x-direction
    N_y = None # order of interpolation in y-direction
    nodes_x = None
    nodes_y = None

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
    eta = None # total height of water
    H   = None # total height to mean level
    u   = None # x-velocity
    v   = None # y-velocity

    H_x = None
    H_y = None
    h_x = None
    h_y = None

    # Simulation specs
    dt = None
    time_steps = None
    end_time = None

    def __init__(self, Nex, Ney, *, N=None, N_x=8, N_y=8, x_start=-5, x_end=5, y_start=-5, y_end=5):
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

        del x_mid, x_temp, y_mid, y_temp, i

        self.eta = np.zeros(self.X.shape)
        self.H = np.zeros(self.X.shape)
        self.u = np.zeros(self.X.shape)
        self.v = np.zeros(self.X.shape)
        self.H_x = np.zeros(self.X.shape)
        self.H_y = np.zeros(self.X.shape)
        self.h_x = np.zeros(self.X.shape)
        self.h_y = np.zeros(self.X.shape)


if __name__ == '__main__':
    print('Running main!!!')
    a = shallow_water(100, 100, N_x=64, N_y=64)#, x_start=0, x_end=5, y_start=0, y_end=5)
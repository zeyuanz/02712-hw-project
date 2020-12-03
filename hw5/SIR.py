import math
import sys

import matplotlib.pyplot as plt
import numpy as np


class grid:
    def __init__(self, x, y, S, I, R):
        '''
        Class grid, property:
            x: float, position of x
            y: float, position of y
            S: float, number of S
            I: float, number of I
            R: float, number of R
        '''
        self.x = x
        self.y = y
        self.S = S
        self.I = I
        self.R = R
        self.N = S + I + R
    
    def update(self, S, I, R):
        '''
            update method: update current value with input
        '''
        self.S = S
        self.I = I
        self.R = R
        self.N = S + I + R

    def __str__(self):
        return 'x: {self.x}, y: {self.y}'.format(self=self)

def init(m, n, S, I, R):
    '''
    init() initialize the grids, delta_x (step_x) and delta_y (step_y)
    args:
        m, n, S, I, R
    return:
        grids, step_x, step_y
    '''
    grids = [None] * m
    for i in range(len(grids)):
        grids[i] = [None] * n

    step_x = 2.0 / (m-1) 
    step_y = 2.0 / (n-1)
    for i in range(len(grids)):
        for j in range(len(grids[i])):
            grids[i][j] = [grid(-1.0 + step_x * i, -1.0 + step_y * j, S, I, R), None]

    return grids, step_x, step_y

def iterator(grids, x_ind, y_ind, mu, l1, l2, rho, delta_x, delta_y, delta_t):
    '''
    iterator() performs second order accuracy simulation at specified grid
    args:
        grids, x_ind, y_ind, mu, l1, l2, rho, delta_x, delta_y, delta_t
    returns:
        new value of S, R, I

    Each grid cantains two value <class = grid>, representing t and t-1
    i.e. grids[i,j] = [grid(t), grid(t-1)]
    '''
    x_ind_next = x_ind + 1
    x_ind_prev = x_ind - 1
    y_ind_next = y_ind + 1
    y_ind_prev = y_ind - 1

    if x_ind == len(grids) - 1:
        x_ind_next = 0
    if x_ind == 0:
        x_ind_prev = len(grids)-1

    if y_ind == len(grids[0]) - 1:
        y_ind_next = 0
    if y_ind == 0:
        y_ind_prev = len(grids[0])-1

    # S iterator 
    s_diffusion = (grids[x_ind_next][y_ind][0].S + grids[x_ind_prev][y_ind][0].S - 2 * grids[x_ind][y_ind][0].S) / delta_x ** 2 + (grids[x_ind][y_ind_next][0].S + grids[x_ind][y_ind_prev][0].S - 2 * grids[x_ind][y_ind][0].S) / delta_y ** 2
    s_convection = rho * (grids[x_ind][y_ind][0].x * ((grids[x_ind_next][y_ind][0].S - grids[x_ind_prev][y_ind][0].S) / 2 / delta_x) + grids[x_ind][y_ind][0].y * ((grids[x_ind][y_ind_next][0].S - grids[x_ind][y_ind_prev][0].S) / 2 / delta_y))
    if grids[x_ind][y_ind][1] is None:
        s_new = grids[x_ind][y_ind][0].S + (mu * s_diffusion - l1 * grids[x_ind][y_ind][0].S * grids[x_ind][y_ind][0].I / grids[x_ind][y_ind][0].N - s_convection) * delta_t
    else:
        s_new = grids[x_ind][y_ind][1].S + (mu * s_diffusion - l1 * grids[x_ind][y_ind][0].S * grids[x_ind][y_ind][0].I / grids[x_ind][y_ind][0].N - s_convection) * 2 * delta_t

    # I iterator
    i_diffusion = (grids[x_ind_next][y_ind][0].I + grids[x_ind_prev][y_ind][0].I - 2 * grids[x_ind][y_ind][0].I) / delta_x ** 2 + (grids[x_ind][y_ind_next][0].I + grids[x_ind][y_ind_prev][0].I - 2 * grids[x_ind][y_ind][0].I) / delta_y ** 2
    i_convection = rho * (grids[x_ind][y_ind][0].x * ((grids[x_ind_next][y_ind][0].I - grids[x_ind_prev][y_ind][0].I) / 2 / delta_x) + grids[x_ind][y_ind][0].y * ((grids[x_ind][y_ind_next][0].I - grids[x_ind][y_ind_prev][0].I) / 2 /delta_y))
    if grids[x_ind][y_ind][1] is None:
        i_new = grids[x_ind][y_ind][0].I + (mu * i_diffusion + l1 * grids[x_ind][y_ind][0].S * grids[x_ind][y_ind][0].I / grids[x_ind][y_ind][0].N - l2 * grids[x_ind][y_ind][0].I - i_convection) * delta_t
    else:
        i_new = grids[x_ind][y_ind][1].I + (mu * i_diffusion + l1 * grids[x_ind][y_ind][0].S * grids[x_ind][y_ind][0].I / grids[x_ind][y_ind][0].N - l2 * grids[x_ind][y_ind][0].I - i_convection) * 2 * delta_t
    
    # R iterator
    r_diffusion = (grids[x_ind_next][y_ind][0].R + grids[x_ind_prev][y_ind][0].R - 2 * grids[x_ind][y_ind][0].R) / delta_x ** 2 + (grids[x_ind][y_ind_next][0].R + grids[x_ind][y_ind_prev][0].R - 2 * grids[x_ind][y_ind][0].R) / delta_y ** 2
    r_convection = rho * (grids[x_ind][y_ind][0].x * ((grids[x_ind_next][y_ind][0].R - grids[x_ind_prev][y_ind][0].R) / 2 / delta_x) + grids[x_ind][y_ind][0].y * ((grids[x_ind][y_ind_next][0].R - grids[x_ind][y_ind_prev][0].R) / 2 / delta_y))
    if grids[x_ind][y_ind][1] is None:
        r_new = grids[x_ind][y_ind][0].R + (mu * r_diffusion + l2 * grids[x_ind][y_ind][0].I - r_convection) * delta_t
    else:
        r_new = grids[x_ind][y_ind][1].R + (mu * r_diffusion + l2 * grids[x_ind][y_ind][0].I - r_convection) * 2 * delta_t

    return s_new, i_new, r_new

def simulator(grids, mu, l1, l2, rho, T, delta_x, delta_y, delta_t):
    '''
    simulator() performs simulation for all grids at a given length T
    '''
    steps = T // delta_t
    S_his = []
    I_his = []
    R_his = []
    early_stop = 10
    for step in range(int(steps)):
        S = 0
        I = 0
        R = 0
        for i in range(len(grids)):
            for j in range(len(grids[i])):
                s_new, i_new, r_new = iterator(grids, i, j, mu, l1, l2, rho, delta_x, delta_y, delta_t)
                grids[i][j][1] = grids[i][j][0]
                grids[i][j][0].update(s_new, i_new, r_new)
                S += s_new
                I += i_new
                R += r_new
        S_his.append(S)
        I_his.append(I)
        R_his.append(R)
        if I < 1e-4:
            early_stop -= 1
        if early_stop == 0:
            break
    return S_his, I_his, R_his

def plot_wrapper(time_steps, time_steps_int, S_his, I_his, R_his, rho):
    '''
    wrapper function for plotting
    '''
    plt.figure(dpi = 300)
    plt.plot(time_steps, S_his, label='S')
    plt.plot(time_steps, I_his, label='I')
    plt.plot(time_steps, R_his, label='R')
    plt.xlabel("T")
    plt.ylabel("total number of people")
    plt.legend()
    plt.title("Rho = %.1f" %(rho))
    plt.xticks(time_steps_int)
    plt.savefig("rho_%.1f.png"%(rho))

def simulate(argv):
    '''
    argparse function
    '''
    lambda1 = float(argv[1])
    lambda2 = float(argv[2])
    mu = float(argv[3])
    rho = float(argv[4])
    S = int(argv[5])
    I = int(argv[6])
    R = int(argv[7])
    T = int(argv[8])
    m = int(argv[9])
    n = int(argv[10])

    delta_t = 1e-3
    grids, delta_x, delta_y = init(m, n, S, I, R)
    S_his, I_his, R_his = simulator(grids, mu, lambda1, lambda2, rho, T, delta_x, delta_y, delta_t)
    time_steps = np.linspace(1,len(S_his), len(S_his)) * delta_t
    time_steps_int = range(math.floor(np.min(time_steps)), math.ceil(np.max(time_steps)))
    plot_wrapper(time_steps, time_steps_int, S_his, I_his, R_his, rho)


if __name__ == "__main__":
    simulate(sys.argv)



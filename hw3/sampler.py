import copy
import math

import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """
    x.shape = (2,n)
    x[0,:] contains x position of all
    x[1,:] contains y position of all
    f(x) return float of the probability density function
    """
    if x.shape[1] == 1:
        x = x ** 2
        return np.exp(-1.0 / np.sqrt(x.sum() + 1e-8))

    x_diff_mat = (x[0,:].reshape(1,-1) - x[0,:].reshape(-1,1)) ** 2 + 1e-8 * np.eye((x.shape[1]))
    y_diff_mat = (x[1,:].reshape(1,-1) - x[1,:].reshape(-1,1)) ** 2 + 1e-8 * np.eye((x.shape[1]))

    distance_mat = np.sqrt(x_diff_mat+y_diff_mat)
    sum_mat = 1.0 / distance_mat
    sum = sum_mat[np.triu_indices(sum_mat.shape[0], k = 1)].sum()
    return np.exp(-sum)

def distance(x):
    """
    compute all pairwise distance if person >= 2
    """
    x_diff_mat = (x[0,:].reshape(1,-1) - x[0,:].reshape(-1,1)) ** 2 + 1e-8 * np.eye((x.shape[1]))
    y_diff_mat = (x[1,:].reshape(1,-1) - x[1,:].reshape(-1,1)) ** 2 + 1e-8 * np.eye((x.shape[1]))
    sum_mat = np.sqrt(x_diff_mat+y_diff_mat)
    sum = sum_mat[np.triu_indices(sum_mat.shape[0], k = 1)].sum()
    return sum

def MH(x0, j, k, s):
    """
    Metroplish Hastings alogrithm at a single dimension
    If the energy comparsion is greater than or equal 1, then we guarantee take the new sample
    If the energy comparsion is smaller than 1, then we have 'prob' from [0,1] to take the new sample
    """
    x = x0
    for i in range(s):
        new_x = copy.deepcopy(x)
        new_val = np.random.rand()
        new_x[j, k] = new_val
        prob = np.random.rand()
        if prob < (f(new_x) / f(x)):
            x = new_x
    return x

def Gibbs(x0, r, s):
    """
    Gibbs sampling alogrithm within r step
    At each step, it randomly chooses one dimension and perform MH sampler in s step
    It returns the r step history and averaga position / distance
    """
    x = x0
    sum_x = 0.0
    sum_y = 0.0
    his_x = []
    his_y = []
    dist = []
    sum_dist = 0.0

    x_r = np.zeros((r,x.shape[0] * x.shape[1]))
    for i in range(r):
        if i % 1000 == 0:
            print(i)
        j = np.random.randint(x.shape[0])
        k = np.random.randint(x.shape[1])
        x = MH(x, j, k, s)
        for j in range(x.shape[0]):
            for k in range(x.shape[1]):
                x_r[i, 2*k+j] = x[j,k]
        if x.shape[1] == 1:
            sum_x += x[0,0]
            sum_y += x[1,0]
            his_x.append(sum_x / (i+1))
            his_y.append(sum_y / (i+1))
        else:
            sum_dist += distance(x)
            if x.shape[1] == 3:
                dist.append(sum_dist / (i+1) / 3)
            else:
                dist.append(sum_dist / (i+1))
    return x_r, his_x, his_y, dist

def wrapper(num_person, r, s):
    """
    wrapper function that initialize positions
    and perform gibbs sampling
    """
    x = np.random.rand(2, num_person)
    x_r, his_x, his_y, dist = Gibbs(x, r, s)
    return x_r, his_x, his_y, dist

def plot(filename, his_x, his_y, dist):
    """
    plot wrapper function 
    """
    plt.figure(dpi = 300)
    if his_x is None:
        plt.title("average distance of "+filename+" person")
        plt.plot(np.linspace(1,r,r), dist, linewidth=0.5, label='distance')
        plt.ylabel("distance")
    else:
        plt.title("average position of one person")
        plt.plot(np.linspace(1,r,r), his_x, linewidth=0.5, label='x')
        plt.plot(np.linspace(1,r,r), his_y, linewidth=0.5, label='y')
        plt.ylabel("position")
    plt.legend()
    plt.xlabel("k")
    plt.savefig(filename+".png")

# constant setup
r = 10000
s = 100

x1, his_x, his_y,_ = wrapper(1, r, s)
x2, _,_, dist_2 = wrapper(2, r, s)
x3, _,_, dist_3 = wrapper(3, r, s)

# save results with given format
np.savetxt("one_person.txt",x1)
np.savetxt("two_person.txt",x2)
np.savetxt("three_person.txt",x3)

# plot results
plot('one', his_x, his_y, None)
plot('two', None, None, dist_2)
plot('three', None, None, dist_3)


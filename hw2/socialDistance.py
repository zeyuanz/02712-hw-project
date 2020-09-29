import copy
import sys

import numpy as np


# compute f(x) according to given x and position
def f(x, position):
    sum_val = 0.0
    for i in range(len(position)):
        sum_val += 1.0 / ((x[0,0] - position[i,0]) ** 2 + (x[0,1] - position[i,1]) ** 2)
    return np.sum(x ** 2) + sum_val

# compute the gradient of f(x) according to x and position
# this is done by closed form solution
def gradient(x, position):
    grad = np.array([[0.0,0.0]])
    sum_1 = 0.0
    sum_2 = 0.0

    for i in range(position.shape[0]):
        denominator = ((x[0,0] - position[i,0]) ** 2 + (x[0,1] - position[i,1]) ** 2) ** 2
        sum_1 += 2.0 * (x[0,0] - position[i,0]) / denominator 
        sum_2 += 2.0 * (x[0,1] - position[i,1]) / denominator

    grad[0,0] = 2.0 * x[0,0] - sum_1
    grad[0,1] = 2.0 * x[0,1] - sum_2

    return grad

# compute hessian matrix of f(x) given x, position and step size z
# this is done by numerical estimation where precision = z
def hessian(x, position, z):
    hes = np.array([[0.0,0.0], [0.0,0.0]])
    for i in range(hes.shape[0]):
        for j in range(hes.shape[1]):
            hes[i,j] = numerical_estimate(x, i, j, z, position)
    return hes

# subroutine to perform numerical estimation
def numerical_estimate(x, i ,j ,z, position):
    temp1 = copy.deepcopy(x)
    temp2 = copy.deepcopy(x)
    temp1[0,j] += z
    temp2[0,j] -= z

    grad1 = gradient(temp1, position)
    grad2 = gradient(temp2 ,position)
    return (grad1[0,i] - grad2[0,i]) / (2.0 * z) 

# compute a 2x2 hessian matrix inverse, using closed form formula
def hessian_inverse(hes):
    denominator = hes[0,0] * hes[1,1] - hes[0,1] * hes[1,0]
    hes_inv = np.array([[0,0],[0,0]])

    hes_inv[0,0] = hes[1,1]
    hes_inv[0,1] = -hes[0,1]
    hes_inv[1,0] = -hes[1,0]
    hes_inv[1,1] = hes[0,0]

    return hes_inv / denominator

# perform newton-raphson method
def newton_raphson(x0, position, z, n_iter):
    x = x0
    while n_iter > 0:
        x = x - hessian_inverse(hessian(x, position, z)).dot(gradient(x, position).T).T
        n_iter -= 1
    return x

# main function dealing with input and use newton raphson to get solution
def main(argv):
    # ----------parsing files----------------
    filename = argv[1]
    f_fd = open(filename, "r")
    x_init_str = f_fd.readline().split(" ")
    x0 = np.array([[0.0,0.0]])
    x0[0,0] = float(x_init_str[0])
    x0[0,1] = float(x_init_str[1])

    s = int(f_fd.readline())
    position = np.zeros((s,2))
    lines = f_fd.readlines()
    count = 0
    for line in lines:
        if lines == "\n":
            break
        line = line.split(" ")
        position[count, 0] = float(line[0])
        position[count, 1] = float(line[1])
        count += 1 
    # ----------parsing files----------------

    z = 0.001
    n_iter = 20
    x = newton_raphson(x0, position, z, n_iter)
    fx = f(x, position)
    print("%f %f" %(x[0,0], x[0,1]))
    f_fd.close()


if __name__ == "__main__":
    main(sys.argv)

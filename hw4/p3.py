import numpy as np


def p(x, y):
    return np.array([[-3*y, y,y,y],[y,-3*y,y,y],[x,x,-y-2*x,y],[x,x,y,-y-2*x]]).T

x = 1
y = 2

v, m = np.linalg.eig(p(x,y))
print(v)
print(m)

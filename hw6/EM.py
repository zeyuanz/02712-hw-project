import sys


def M_step(e, f, g, a, b, c, d):
    n = a + b + c + d
    x = (e + g) / n
    y = (f + g) / n
    return x, y

def E_step(x, y, a, b, c, d):
    n = a + b + c + d
    E_e = x * (1-y) * n
    E_f = (1-x) * y * n
    E_g = x * y * n
    return E_e, E_f, E_g

def EM(argv):
    a = float(argv[1])
    b = float(argv[2])
    c = float(argv[3])
    d = float(argv[4])
    n_iter = int(argv[5])
    x = c / (c+d) * (b+c+d) / (a+b+c+d)
    y = d / (c+d) * (b+c+d) / (a+b+c+d)
    for i in range(n_iter):
        e, f, g = E_step(x, y, a, b, c, d)
        x, y = M_step(e, f, g, a, b ,c, d)

    print("[no disease]: %.4f" %((1-x) * (1-y)))    
    print("[just COVID]: %.4f" %(x * (1-y)))    
    print("[just flu]: %.4f" %((1-x) * y))    
    print("[both COVID and flu]: %.4f" %(x * y))    
    print("[f]: %.4f" %(f))


if __name__ == "__main__":
    EM(sys.argv)

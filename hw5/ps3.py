def forward_euler(A, D, I, k1, k2, delta_t):
    A_ori = A
    D_ori = D
    I_ori = I

    A = A_ori + (-k1 * A_ori * D_ori * D_ori + k2 * I_ori) * delta_t
    D = D_ori + (-2 * k1 * A_ori * D_ori * D_ori + 2 * k2 * I_ori) * delta_t
    I = I_ori + (k1 * A_ori * D_ori * D_ori - k2 * I_ori) * delta_t
    return A, D, I

def backward_fun_grad_A(delta_t, k1, k2, A_new, A_ori):
    functional_val = delta_t * (4 * k1 * A_new ** 3 - 36 * k1 * A_new ** 2 + (81 * k1 + k2) * A_new - 5 * k2) + A_new - A_ori
    grad = delta_t * (12 * k1 * A_new ** 2 - 72 * k1 * A_new + 81 * k1 + k2) + 1
    return functional_val, grad

def backward_fun_grad_D(delta_t, k1, k2, D_new, D_ori):
    functional_val = delta_t * (-k1 * D_new ** 3 + 8 * k1 * D_new ** 2 + k2 * D_new - k2) + D_new - D_ori
    grad = delta_t * (-3 * k1 * D_new ** 2 + 16 * k1 * D_new + k2) + 1
    return functional_val, grad

def backward_fun_grad_I(delta_t, k1, k2, I_new, I_ori):
    functional_val = delta_t * (4 * k1 * I_new ** 3 - 24 * k1 * I_new ** 2 + (21 * k1 + k2) * I_new - 5 * k1) + I_new - I_ori
    grad = delta_t * (12 * k1 * I_new ** 2 - 48 * k1 * I_new + 21 * k1 + k2) + 1
    return functional_val, grad

def newton_raphson(n_iter, k1, k2, A, D, I):
    A_ori = A
    D_ori = D
    I_ori = I
    for i in range(n_iter):
        #func_A, grad_A = backward_fun_grad_A(delta_t, k1, k2, A, A_ori)
        #func_D, grad_D = backward_fun_grad_D(delta_t, k1, k2, D, D_ori)
        func_I, grad_I = backward_fun_grad_I(delta_t, k1, k2, I, I_ori)
        #A = A - func_A / grad_A
        #D = D - func_D / grad_D
        I = I - func_I / grad_I

    return 5 - I, 1 - 2 * I, I

def backward_euler(A, D, I, k1, k2, delta_t):
    n_iter = 1000
    A_ori = A
    D_ori = D
    I_ori = I
    A, D, I = newton_raphson(n_iter, k1, k2, A_ori, D_ori, I_ori)
    return A, D, I

def mid_point_method(A, D, I, k1, k2, delta_t):
    A_ori = A
    D_ori = D
    I_ori = I

    A_mid = A_ori + (-k1 * A_ori * D_ori * D_ori + k2 * I_ori) * delta_t / 2.0
    D_mid = D_ori + (-2 * k1 * A_ori * D_ori * D_ori + 2 * k2 * I_ori) * delta_t / 2.0
    I_mid = I_ori + (k1 * A_ori * D_ori * D_ori - k2 * I_ori) * delta_t / 2.0

    A = A_ori + (-k1 * A_mid * D_mid * D_mid+ k2 * I_mid) * delta_t
    D = D_ori + (-2 * k1 * A_mid* D_mid * D_mid + 2 * k2 * I_mid) * delta_t
    I = I_ori + (k1 * A_mid * D_mid * D_mid- k2 * I_mid) * delta_t
    return A, D, I

def mass_conservation(A, D, I, eps = 0.01):
    if abs(A + I - 5.0) > eps or abs(D + 2.0 * I - 1.0) > eps:
        print('Mass unconserved !')
        print('A + I =', A+I)
        print('D + 2I =', D+2.0*I)

A = 5.0
D = 1.0
I = 0.0
k1 = 3.0
k2 = 1.0
delta_t = 0.01
#--------------3a-----------#
#---------------------------#
A1, D1, I1 = forward_euler(A, D, I, k1, k2, delta_t)
A2, D2, I2 = forward_euler(A1, D1, I1, k1, k2, delta_t)
mass_conservation(A, D, I)
print("[3a]\tstep1\tA: %.4f D: %.4f I: %.4f" %(A1, D1, I1))
print("[3a]\tstep2\tA: %.4f D: %.4f I: %.4f\n" %(A2, D2, I2))

#--------------3b-----------#
#---------------------------#
A1, D1, I1 = backward_euler(A, D, I, k1, k2, delta_t)
A2, D2, I2 = backward_euler(A1, D1, I1, k1, k2, delta_t)
mass_conservation(A, D, I)
print("[3b]\tstep1\tA: %.4f D: %.4f I: %.4f" %(A1, D1, I1))
print("[3b]\tstep2\tA: %.4f D: %.4f I: %.4f\n" %(A2, D2, I2))


#--------------3c-----------#
#---------------------------#
A1, D1, I1 = mid_point_method(A, D, I, k1, k2, delta_t)
A2, D2, I2 = mid_point_method(A1, D1, I1, k1, k2, delta_t)
mass_conservation(A, D, I)
print("[3c]\tstpe1\tA: %.4f D: %.4f I: %.4f" %(A1, D1, I1))
print("[3c]\tstep2\tA: %.4f D: %.4f I: %.4f" %(A2, D2, I2))


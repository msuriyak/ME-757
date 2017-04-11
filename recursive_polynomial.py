import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre

# *********** Evaluation ***********
def legendre_value(n, x):
    value = np.zeros(n + 1)
    value[0] = 1
    value[1] = x
    for i in range(2, n + 1):
        value[i] = ((2 * i - 1) * value[i - 1] *
                    x - (i - 1) * value[i - 2]) / i
    return value[n]

def lobatto_value(n, x):
    coeffs = legendre(n - 1).coeffs[::-1]
    print(coeffs)
    coeff_der = derivative_coeffs(coeffs)
    print(coeff_der)
    value = (1 - x**2) * polynomial_value(coeff_der, x)
    return value

# *********** Roots ***********
def legendre_roots(n):
    x = np.zeros(n)
    for i in range(n):
        a = ((4. * i + 3) / (4 * n + 2)) * np.pi
        b = (n - 1) / (8. * (n**3))
        x[i] = (1 - b) * np.cos(a)
        iters = 0
        while True:
            if(iters > 500 or abs(0 - legendre_value(n, x[i])) < 1e-16):
                break
            y = legendre_value(n, x[i])
            y1 = (n * (legendre_value(n - 1,
                                      x[i]) - x[i] * legendre_value(n, x[i]))) / (1 - x[i]**2)
            y2 = (2 * x[i] * y1 - n * (n + 1) *
                  legendre_value(n, x[i])) / (1 - x[i]**2)
            num = 2 * y * y1
            den = 2 * (y1**2) - y * y2
            x[i] = x[i] - (num / den)
            iters = iters + 1
    return np.sort(x)

def lobatto_roots(n):
    x = np.zeros(n)
    x[0] = 1.0
    x[n - 1] = -1.0
    for i in range(1, n - 1):
        a = ((4. * i + 1) / (4 * n - 3)) * np.pi
        b = 3 * (n - 2) / (8. * ((n - 1)**3))
        x[i] = (1 - b) * np.cos(a)
        iters = 0
        while True:
            y = 1.0
            if(iters > 500 or abs(x[i] - y) < 1e-16):
                break
            d = (1 - x[i]**2)
            y = (n - 1) * (legendre_value(n - 2,
                                          x[i]) - x[i] * legendre_value(n - 1, x[i])) / d
            y1 = (2 * x[i] * y - n * (n - 1) * legendre_value(n - 1, x[i])) / d
            y2 = (2 * x[i] * y1 - (n * (n - 1) - 2) * y) / d
            num = 2 * y * y1
            den = 2 * (y1**2) - y * y2
            x[i] = x[i] - (num / den)
            iters = iters + 1
    return np.sort(x)

# *********** Weights ***********
def legendre_weights(n):
    x = legendre_roots(n)
    w = np.zeros(n)
    for i in range(n):
        val = legendre_value(n - 1, x[i])
        w[i] = 2 * (1 - x[i]**2) / (n * val)**2
    return w

def lobatto_weights(n):
    x = lobatto_roots(n)
    w = np.zeros(n)
    for i in range(n):
        val = legendre_value(n - 1, x[i])
        w[i] = 2.0 / (n * (n - 1) * (val**2))
    return w

# *********** Interpolation ***********
def lagrange(i, n, x, x_k):
    v = 1.0
    for j in range(n):
        if i != j:
            v = v * ((x_k - x[j]) / (x[i] - x[j]))
    return v

def func_interpolation(x, n, x_k, func):
    f = func(x)
    vv = []
    for j in range(len(x_k)):
        v = 0.0
        for i in range(n):
            v = v + f[i] * lagrange(i, n, x, x_k[j])
        vv.append(v)
    return np.array(vv)

NODES = {'lobatto' : lobatto_roots,
         'legendre' : legendre_roots}

WEIGHTS = {'lobatto' : lobatto_weights,
           'legendre' : legendre_weights}

def func_integrate(n, func, type_interpolation='legendre', type_integration='lobatto'):
    x = NODES[type_interpolation](n)
    nodes = NODES[type_integration](n)
    func = func_interpolation(x, n, nodes, func)
    weights = WEIGHTS[type_integration](n)
    
    integral = 0.0
    for i in range(n):
        integral = integral + func[i] * weights[i]
    return integral

if __name__ == '__main__':
    print('Running main!!')
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import warnings 
warnings.filterwarnings("ignore", category=RuntimeWarning) 

"""
points = np.array([[0, 0], [0, 1],[2,0], [1, 0],[2,1], [1, 1]])
tri = Delaunay(points)

print(points[tri.simplices])
plt.figure()
plt.triplot(points[:,0], points[:,1], tri.simplices)
plt.plot(points[:,0], points[:,1], 'o')
plt.show()
"""

def F(x):
    x1,x2=x
    return np.array([x1**(2)+x2**(2)-10,
                    x1-3*x2+10])
def Fprime(x):
    x1, x2 = x
    Fprime = np.zeros((2, len(x)))
    Fprime[0][0] = 2*x1
    Fprime[0][1] =  2*x2
    Fprime[1][0] =  1
    Fprime[1][1] = -3
    return Fprime

"""
def F(x):
    x1,x2 = x
    return np.array([3*x1-x2+2,
                    2*x1**2-x2])
def Fprime(x):
    x1, x2 = x
    Fprime = np.zeros((2, len(x)))
    Fprime[0][0] = 3
    Fprime[0][1] =  -1
    Fprime[1][0] =  4*x1
    Fprime[1][1] = -1
    return Fprime
"""
"""
def F(x):
    x1, x2 = x
    return np.array([1/2 * x2**(1/3) / x1**(1/2) - 1/2,
                     1/3 * x1**(1/2) / x2**(2/3) - 1/3])
def Fprime(x):
    x1, x2 = x
    Fprime = np.zeros((2, len(x)))
    Fprime[0][0] = -1/4 * x2**( 1/3) * x1**(-3/2)
    Fprime[0][1] =  1/6 * x1**(-1/2) * x2**(-2/3)
    Fprime[1][0] =  1/6 * x2**(-2/3) * x1**(-1/2)
    Fprime[1][1] = -2/9 * x1**( 1/2) * x2**(-5/3)
    return Fprime
"""
def H_N(x, lambd, x0):
    return F(x) - (1 - lambd) * F(x0)

def derH(x):
    return Fprime(x)

def predict(lambd, lambd_prev, x0, x_init):
    delta_lambd = lambd - lambd_prev
    J = derH(x0)
    H1 = H_N(x0, lambd, x_init)
    H0 = H_N(x0, lambd_prev, x_init)
    xdot = -np.dot(np.linalg.inv(J), (H1 - H0) / delta_lambd)
    return delta_lambd, xdot

def correct(x1, lambd, x_init, tol, max_iter):
    H1 = H_N(x1, lambd, x_init)
    i = 0
    while np.linalg.norm(H1) > tol and i < max_iter:
        H1 = H_N(x1, lambd, x_init)
        J1 = derH(x1)
        d = -np.dot(np.linalg.inv(J1), H1)
        x1 = x1 + d
        i += 1
    return i, x1

def HomotopyMethod(x_init, lambd_step=0.05, tol=1e-5, max_iter=1000):
    # initialize values
    delta_lambd = lambd_step
    x0 = x_init
    lambd_prev = 0
    lambd = lambd_prev
    result = {'lambda':[lambd], 'x':[x0]}
    
    # print initial system
    print('Initial condition:  \u03BB = {:.2f}, x = {} \n'.format(lambd, x0))
    
    # iterate lambda
    k = 0
    while abs(lambd - 1) > np.finfo(float).eps:
        # calculate predictor using gradient
        lambd = min(lambd_prev + delta_lambd, 1)
        delta_lambd, xdot = predict(lambd, lambd_prev, x0, x_init)
        x1 = x0 + delta_lambd * xdot
        # calculate corrector using Newton-Raphson
        i, x1 = correct(x1, lambd, x_init, tol, max_iter)
        # Newton-Raphson no converge
        if i == max_iter:
            delta_lambd /= 2
        else:
            delta_lambd = lambd_step
            x0 = x1
            lambd_prev = lambd
            k += 1
            result['x'].append(x1)
            result['lambda'].append(lambd)
            # print intermediary system
            print('Iteration: {} \t \u03BB	 = {:.2f} \t x = {}'.format(k, lambd, x1))
        # if the Jacobian is singular, break the loop
        if np.isnan(np.sum(x1)):
            print('Solucion no encontrada. Ingrese otro valor inicial  x.')
            return None
    # print target system
    if not np.isnan(np.sum(x1)):
        print('Solucion: x =', x1)
    
    return result


def Newton_extendido(x,max_iterations):
    tol = 1*10**(-5)
    cont=0
    while 1:
        cont+=1
        F_v= F(x)
        DF_v= Fprime(x)
        new_x = x - np.dot(np.linalg.inv(DF_v),F_v)
        if F(new_x)[0] <= tol and F(new_x)[1] <= tol or cont >= max_iterations:
            break
        x = new_x
    return new_x
x_init = np.array([-8,20])
w = Newton_extendido(x_init, 10000)
print(w)
e = HomotopyMethod(x_init)


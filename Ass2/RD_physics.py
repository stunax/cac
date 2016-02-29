#From http://ipython-books.github.io/featured-05/

import numpy as np
import matplotlib.pyplot as plt


a = 2.8e-4
b = 5e-3
tau = .1
k = -.005


def laplacian(Z, dx):
    Ztop = Z[0:-2,1:-1]
    Zleft = Z[1:-1,0:-2]
    Zbottom = Z[2:,1:-1]
    Zright = Z[1:-1,2:]
    Zcenter = Z[1:-1,1:-1]
    return (Ztop + Zleft + Zbottom + Zright - 4 * Zcenter) / dx**2

def RD(U, V, dx, dt):
    # We compute the Laplacian of u and v.
    deltaU = laplacian(U, dx)
    deltaV = laplacian(V, dx)
    # We take the values of u and v inside the grid.
    Uc = U[1:-1,1:-1]
    Vc = V[1:-1,1:-1]
    # We update the variables.
    U[1:-1,1:-1], V[1:-1,1:-1] = \
        Uc + dt * (a * deltaU + Uc - Uc**3 - Vc + k), \
        Vc + dt * (b * deltaV + Uc - Vc) / tau
    # Neumann conditions: derivatives at the edges
    # are null.
    for Z in (U, V):
        Z[0,:] = Z[1,:]
        Z[-1,:] = Z[-2,:]
        Z[:,0] = Z[:,1]
        Z[:,-1] = Z[:,-2]

def gen_system(size):
    U = np.zeros((size, size), dtype = np.float)
    V = np.zeros((size, size), dtype = np.float)

    center = size/2
    blob = np.ones((10,10), dtype=np.float)
    blob[1:-1,1:] = 0.0
    bsize, _ = blob.shape

    for i in xrange(1, size, 2*bsize):
        U[ i:i+bsize, i:i+bsize ] = blob

    return (U, V)
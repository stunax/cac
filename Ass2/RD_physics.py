#From http://ipython-books.github.io/featured-05/

import numpy as np
import shmarray as sh
from skimage.util.shape import view_as_windows
import matplotlib.pyplot as plt
from multiprocessing import Pool

a = 2.8e-4
b = 5e-3
tau = .1
k = -.005

def laplacian(Z, dx):
    dx = dx**2
    Ztop = Z[0:-2,1:-1]
    Zleft = Z[1:-1,0:-2]
    Zbottom = Z[2:,1:-1]
    Zright = Z[1:-1,2:]
    Zcenter = Z[1:-1,1:-1]
    return (Ztop + Zleft + Zbottom + Zright - 4 * Zcenter) / dx**2

def worker(i,U,V,dx,dt,blocksize,workers,barrier):
    return 2+2

def RD_worker(U, V, dx, dt,Ures,Vres):
    # We compute the Laplacian of u and v.
    deltaU = laplacian(U, dx)
    deltaV = laplacian(V, dx)
    # We take the values of u and v inside the grid.
    Uc = U[1:-1,1:-1]
    Vc = V[1:-1,1:-1]
    # We update the variables.
    Ures[:] = Uc + dt * (a * deltaU + Uc - Uc**3 - Vc + k)
    Vres[:] = Vc + dt * (b * deltaV + Uc - Vc) / tau
    return 0

def mcop(U,Ures,V,Vres):
    U[:] = Ures[:]
    V[:] = Vres[:]
    return 0

def RD(U, V, dx, dt, pool,workers,Ures,Vres):
    blocks = U.shape[0]
    rowsPerWorker = blocks / workers


    args = ((U[i*rowsPerWorker:(i+1)*rowsPerWorker+1],V[i*rowsPerWorker:(i+1)*rowsPerWorker+1],dx,dt,Ures,Vres) for i in xrange(workers))
    results = (pool.apply_async(RD_worker,arg) for arg in args)
    (res.wait() for res in results)
    pool.close()
    pool.join()
    args = ((U[i*rowsPerWorker:(i+1)*rowsPerWorker+1],Ures[i*rowsPerWorker:(i+1)*rowsPerWorker+1],
             V[i*rowsPerWorker:(i+1)*rowsPerWorker+1],Vres[i*rowsPerWorker:(i+1)*rowsPerWorker+1])
            for i in xrange(workers))
    results =(pool.apply_async(mcop,arg) for arg in args)
    (res.wait() for res in results)

    # Neumann conditions: derivatives at the edges are null.
    for Z in (U, V):
        Z[0,:] = Z[1,:]
        Z[-1,:] = Z[-2,:]
        Z[:,0] = Z[:,1]
        Z[:,-1] = Z[:,-2]

def gen_system(size):
    U = sh.zeros((size, size), dtype = np.float)
    V = sh.zeros((size, size), dtype = np.float)
    Ures = sh.zeros((size, size), dtype = np.float)
    Vres = sh.zeros((size, size), dtype = np.float)

    center = size/2
    blob = np.ones((10,10), dtype=np.float)
    blob[1:-1,1:] = 0.0
    bsize, _ = blob.shape

    for i in xrange(1, size, 2*bsize):
        U[ i:i+bsize, i:i+bsize ] = blob

    return (U, V,Ures,Vres)
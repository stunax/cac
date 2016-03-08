#From http://ipython-books.github.io/featured-05/

import numpy as np
import shmarray as sh
#from skimage.util.shape import view_as_windows
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool

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

#def RD_worker(U, V, dx, dt,Ures,Vres):
def RD_worker(args):
    dx, dt,i,rows = args

    #Create views
    Uview = U[i*rows:(i+1)*rows+2]
    Vview = V[i*rows:(i+1)*rows+2]
    Uresview = Ures[i*rows:(i+1)*rows]
    Vresview = Vres[i*rows:(i+1)*rows]

    # We compute the Laplacian of u and v.
    deltaU = laplacian(Uview, dx)
    deltaV = laplacian(Vview, dx)
    # We take the values of u and v inside the grid.
    Uc = Uview[1:-1,1:-1]
    Vc = Vview[1:-1,1:-1]
    # We update the variables.
    Uresview[:] = Uc + dt * (a * deltaU + Uc - Uc**3 - Vc + k)
    Vresview[:] = Vc + dt * (b * deltaV + Uc - Vc) / tau

def mcop(args):
    i,rows = args

    #Create views
    Uview = U[i*rows:(i+1)*rows+2]
    Vview = V[i*rows:(i+1)*rows+2]
    Uresview = Ures[i*rows:(i+1)*rows]
    Vresview = Vres[i*rows:(i+1)*rows]


    Uview[1:-1,1:-1] = Uresview[:]
    Vview[1:-1,1:-1] = Vresview[:]

def fixHor(args):
    i,side = args
    if i < 2:
        if side:
            U[0] = U[1]
        else:
            U[-1] = U[-2]
    else:
        if side:
            V[0] = V[1]
        else:
            V[-1] = V[-2]

def fixVer(args):
    i,side = args
    if i < 2:
        if side:
            U[:,0] = U[:,1]
        else:
            U[:,-1] = U[:,-2]
    else:
        if side:
            V[:,0] = V[:,1]
        else:
            V[:,-1] = V[:,-2]


def RD(dx, dt, pool,workers):
    rowsPerWorker = U.shape[0] / workers+1
    #V = V_
    #U = U_

    #run rd_worker
    args = [(dx,dt,i,rowsPerWorker) for i in xrange(workers)]
    result = pool.map_async(RD_worker,args)
    result.get()

    #Apply result. Doing this barriers are not needed
    args = [(i,rowsPerWorker) for i in xrange(workers)]
    result2 = pool.map_async(mcop,args)
    result2.get()
    # Neumann conditions: derivatives at the edges are null.

    args = [(i,i%2 == 0) for i in xrange(4)]
    resulthor = pool.map_async(fixHor,args)
    resulthor.get()

    args = [(i,i%2 == 0) for i in xrange(4)]
    resultver = pool.map_async(fixVer,args)
    resultver.get()

def gen_system(size):
    global V
    global U
    global Vres
    global Ures
    U = sh.zeros((size, size), dtype = np.float)
    V = sh.zeros((size, size), dtype = np.float)
    Ures = sh.zeros((size-2, size-2), dtype = np.float)
    Vres = sh.zeros((size-2, size-2), dtype = np.float)

    center = size/2
    blob = np.ones((10,10), dtype=np.float)
    blob[1:-1,1:] = 0.0
    bsize, _ = blob.shape

    for i in xrange(1, size, 2*bsize):
        U[ i:i+bsize, i:i+bsize ] = blob

    return (U, V)
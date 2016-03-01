#From http://ipython-books.github.io/featured-05/

import numpy as np
import matplotlib.pyplot as plt
from RD_physics import RD, gen_system
from multiprocessing import Pool

def RD_debug(n, U, V, dx, dt,Ures,Vres):
    plt.ion()
    cpus = 4
    workers = cpus * 2
    my_pool = Pool(cpus)
    for i in range(n):
        RD(U, V, dx, dt, my_pool,workers,Ures,Vres)

        if i%1000 == 0:
            print(U[50,50:55])
            print i
            plt.imshow(U, cmap=plt.cm.copper, extent=[-1,1,-1,1]);
            plt.xticks([]); plt.yticks([]);
            plt.draw()
            plt.pause(0.01)
    plt.ioff()

if __name__ == '__main__':
    size = 100  # size of the 2D grid
    dx = 2./size  # space step
    T = 10.0  # total time
    dt = .9 * dx**2/2  # time step
    n = 50001

    U, V,Ures,Vres = gen_system(size)

    RD_debug(n, U, V, dx, dt,Ures,Vres)

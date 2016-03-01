#From http://ipython-books.github.io/featured-05/
import time

import numpy as np
import matplotlib.pyplot as plt
from RD_physics import RD, gen_system
from multiprocessing import Pool

def RD_bench(n, U, V, dx, dt,Ures,Vres):

    cpus = 32
    workers = cpus * 2
    my_pool = Pool(cpus)
    for i in xrange(n):
        RD(U, V, dx, dt, my_pool,workers,Ures,Vres)

if __name__ == '__main__':

    for size in (500, 1000, 1500):
        dx = 2./size  # space step
        T = 10.0  # total time
        dt = .9 * dx**2/2  # time step
        n = 1000

        U, V,Ures,Vres = gen_system(size)

        start = time.time()
        RD_bench(n, U, V, dx, dt,Ures,Vres)
        stop = time.time()

        print 'Simulated ' + str(size) + ' squared system for ' \
            + str(n) + ' timesteps in ' + str(stop - start) \
            + ' seconds'
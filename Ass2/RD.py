#From http://ipython-books.github.io/featured-05/
import time

import numpy as np
import matplotlib.pyplot as plt
from RD_physics import RD, gen_system
from multiprocessing import Pool

def RD_bench(n, U, V, dx, dt,cpus):

    pool = Pool(processes=cpus)
    for i in xrange(n):
        RD(dx, dt,pool, cpus*2)

    pool.close()
    pool.join()

if __name__ == '__main__':

    cpus = 8
    print str(cpus) + " cpus"
    for size in (500, 1000, 1500):
        dx = 2./size  # space step
        T = 10.0  # total time
        dt = .9 * dx**2/2  # time step
        n = 1000

        U, V= gen_system(size)


        message = str(size)
        for i in range(3):
            start = time.time()
            RD_bench(n, U, V, dx, dt,cpus)
            stop = time.time()
            message += ',' + str(stop - start)
        print  message
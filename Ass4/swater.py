#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Shallow water simulation

Adapted from: http://people.sc.fsu.edu/~jburkardt/m_src/shallow_water_2d/
"""

import numpy
import time
from meshgrid import ndgrid
from swaterphysics import evolve

def droplet(height, width, data_type=numpy.double):
    """Generate grid of droplets"""

    x = numpy.linspace(-1, 1, num=width,
                       endpoint=True).astype(data_type)
    y = numpy.linspace(-1, 1, num=width,
                       endpoint=True).astype(data_type)

    (xx, yy) = ndgrid(x, y)

    droplet = height * numpy.exp(-5 * (xx ** 2 + yy ** 2))

    return droplet

def swater(
    n,
    timesteps,
    ):
    """Simulate shallow water movement following a drop"""

    dt = 0.02  # hard-wired timestep
    dx = 1.0
    dy = 1.0
    D = droplet(2.5, 10)  # simulate a water drop
    droploc = n / 4

    H = numpy.ones((n + 2, n + 2))
    U = numpy.zeros((n + 2, n + 2))
    V = numpy.zeros((n + 2, n + 2))
    Hx = numpy.zeros((n + 1, n + 1))
    Ux = numpy.zeros((n + 1, n + 1))
    Vx = numpy.zeros((n + 1, n + 1))
    Hy = numpy.zeros((n + 1, n + 1))
    Uy = numpy.zeros((n + 1, n + 1))
    Vy = numpy.zeros((n + 1, n + 1))

    (dropx, dropy) = D.shape
    H[droploc:droploc + dropx, droploc:droploc + dropy] += D

    evolve(timesteps, n, H, U, V, Hx, Hy, Ux, Uy, Vx, Vy, dx, dy, dt)


# This visualizer will not show graphics for benchmarking
# but run a larger system for a few timesteps
timesteps=10

#You are allowed to go to even higher n to show that your parallel version
#scales well

for n in (1000, 2000, 4000): 
    start = time.time()
    swater(n, timesteps)
    stop = time.time()

    print 'N=',n,'Time taken', stop - start


#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Shallow water simulation

Adapted from: http://people.sc.fsu.edu/~jburkardt/m_src/shallow_water_2d/

Visualizer for Shallow Water simulation
"""

import numpy
import time
from meshgrid import ndgrid
from swaterphysics import evolve

# Tell pylint that Axes3D import is required although never explicitly used

from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=W0611
import matplotlib.pyplot as plt

from meshgrid import ndgrid

class Visualize:

    """Visualizer"""

    def __init__(
        self,
        n,
        ):

        plt.ion()

        # First parameter is a figure number, second is size in inches

        self.fig = plt.figure(11, (8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_zlim3d(-2, 3)
        self.wframe = None
        (self.X, self.Y) = ndgrid(numpy.linspace(0, n, n),
                                  numpy.linspace(0, n, n))

    def update(self, H, fname=None):
        """Show plot"""

        if self.wframe:

            # Remove previous plot lines

            self.ax.collections.remove(self.wframe)
        self.wframe = self.ax.plot_wireframe(self.X, self.Y, H,
                rstride=1, cstride=1)

        # For some reason the limits need to be set again

        self.ax.set_zlim3d(-2, 4)
        plt.draw()
        plt.pause(0.002)

    def end(self):
        """End plot"""

        # Make sure that the window does not close before we wan't it to

        plt.ioff()
        plt.show()



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
    plotfreq,
    visual,
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

    start = time.time()

    for _ in xrange(timesteps/plotfreq):
        visual.update(H[1:-1, 1:-1])
        evolve(plotfreq, n, H, U, V, Hx, Hy, Ux, Uy, Vx, Vy, dx, dy, dt)
    stop = time.time()
    visual.end()

    return stop - start


# The below lines define a small system with a visualizer that
# will show graphics for debugging

n = 50  # grid size
plotfreq = 8
timesteps = 40 * n
visualizer = Visualize(n)

timing = swater(n, timesteps, plotfreq, visualizer)

print 'Time taken', timing

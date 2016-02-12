#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Interactive visual front end to NBody implementation"""

import time

# Tell pylint that Axes3D import is required although never explicitly used

from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=W0611
import matplotlib.pyplot as plt

from nbodyphysics import move, random_system


def gfx_init(xm, ym, zm):
    """Init plot"""

    plt.ion()
    fig = plt.figure()
    sub = fig.add_subplot(111, projection='3d')
    sub.xm = xm
    sub.ym = ym
    sub.zm = zm
    return sub


def show(sub, solarsystem, bodies):
    """Show plot"""
    #Sun
    sub.clear()

    sub.scatter(
                solarsystem[0,1],
                solarsystem[0,2],
                solarsystem[0,3],
                s=100,
                marker='o',
                c='yellow',
            )
    #Planets
    sub.scatter(solarsystem[1:,1],solarsystem[1:,2],solarsystem[1:,3],s = 5,marker = 'o',c = "blue")
#    sub.scatter(
#                [i['x'] for i in solarsystem[1:]],
#                [i['y'] for i in solarsystem[1:]],
#                [i['z'] for i in solarsystem[1:]],
#                s=5,
#                marker='o',
#                c='blue',
#        )

    
#Asteroids
    sub.scatter(bodies[1:,1],bodies[1:,2],bodies[1:,3],s = 1,marker = '.',c = "green")
#    sub.scatter(
#                [i['x'] for i in bodies],
#                [i['y'] for i in bodies],
#                [i['z'] for i in bodies],
#                s=.1,
#                marker='.',
#                c='green',
#        )


    sub.set_xbound(-sub.xm, sub.xm)
    sub.set_ybound(-sub.ym, sub.ym)
    try:
        sub.set_zbound(-sub.zm, sub.zm)
    except AttributeError:
        print 'Warning: correct 3D plots may require matplotlib-1.1 or later'

    plt.draw()
    plt.pause(0.0000001)


def nbody_debug(n, bodies, time_step):
    """Run simulation with visualization"""

    x_max = 1e18
    y_max = 1e18
    z_max = 1e18
    
    solarsystem, asteroids = random_system(x_max, y_max, z_max, n, bodies)

    P3 = gfx_init(x_max, y_max, z_max)
    dt = 1e12  # One hundred year timesteps ensures that we see movement



    start = time.time()
    for step in range(time_step):
        if step%1 == 0:
            show(P3, solarsystem, asteroids)
        move(solarsystem, asteroids, dt)
        print step
    stop = time.time()
    print 'Simulated ' + str(bodies) + ' bodies for ' + str(time_step) \
        + ' timesteps in ' + str(stop - start) + ' seconds'


if __name__ == '__main__':
    nbody_debug(10, 1000, 1000)


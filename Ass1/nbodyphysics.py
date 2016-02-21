#!/usr/bin/python
# -*- coding: utf-8 -*-

"""NBody in N^2 complexity
Note that we are using only Newtonian forces and do not consider relativity
Neither do we consider collisions between stars
Thus some of our stars will accelerate to speeds beyond c
This is done to keep the simulation simple enough for teaching purposes

All the work is done in the calc_force, move and random_galaxy functions.
To vectorize the code these are the functions to transform.
"""
import numpy
from numpy import exp, arctan, sqrt, pi, cos, sin
from numpy.random import random

G = 6.673e-11
solarmass = 1.98892e30


# def calc_force(a, b, dt):
#    """Calculate forces between bodies
#    F = ((G m_a m_b)/r^2)*((x_b-x_a)/r)
#    """
#    #r = numpy.sum((b[:,-3:]-a[numpy.newaxis,-3:])**2,axis = 1)**0.5
#    r = ((b['x'] - a['x']) ** 2 + (b['y'] - a['y']) ** 2 + (b['z'] - a['z']) ** 2) ** 0.5
#    a['vx'] += G * a['m'] * b['m'] / r ** 2 * ((b['x'] - a['x']) / r) / a['m'] * dt
#    a['vy'] += G * a['m'] * b['m'] / r ** 2 * ((b['y'] - a['y']) / r) / a['m'] * dt
#    a['vz'] += G * a['m'] * b['m'] / r ** 2 * ((b['z'] - a['z']) / r) / a['m'] * dt

def calc_force_vec(a, b, dt):
    """Calculate forces between bodies
    F = ((G m_a m_b)/r^2)*((x_b-x_a)/r)
    """

    temp = b[numpy.newaxis, :, 1:4] - a[:, numpy.newaxis, 1:4]

    r = numpy.sum(temp ** 2, axis=2) ** 0.5

    a[:, 4:7] += numpy.sum(
        (G * a[:, numpy.newaxis, 0] * b[numpy.newaxis, :, 0] / (r ** 2))[:, :, numpy.newaxis] *
        (temp / r[:, :, numpy.newaxis] / a[:, numpy.newaxis, numpy.newaxis, 0] * dt)
        , axis=1
    )
    # r = numpy.sum((b[:,1:4]-a[numpy.newaxis,1:4])**2,axis = 1)**0.5

    # a[4:7] += numpy.sum((G*a[0]*b[:,0] / (r**2))[:,numpy.newaxis] *
    #    ((b[:,1:4] - a[1:4])/r[:,numpy.newaxis]) /a[0] * dt,axis = 0)


#    a[4] += numpy.sum(G*a[0]*b[:,0] / (r**2) * ((b[:,1] - a[1])/r) /a[0] * dt)
#    a[5] += numpy.sum(G*a[0]*b[:,0] / (r**2) * ((b[:,2] - a[2])/r) /a[0] * dt)
#    a[6] += numpy.sum(G*a[0]*b[:,0] / (r**2) * ((b[:,3] - a[3])/r) /a[0] * dt)


def calc_force(a, b, dt, solar):
    """Calculate forces between bodies
    F = ((G m_a m_b)/r^2)*((x_b-x_a)/r)
    """
    diff = b[1:4, numpy.newaxis, :] - a[1:4, :, numpy.newaxis]
    r = numpy.sum(diff ** 2, axis=0) ** 0.5
    if solar:
        r[numpy.diag_indices(r.shape[0])] = 1

    mass = G * b[0, numpy.newaxis, :] * a[0, :, numpy.newaxis] / (r ** 2)

    a[4] += numpy.sum(mass * (diff[0] / r) / a[0, :, numpy.newaxis] * dt, axis=1)
    a[5] += numpy.sum(mass * (diff[1] / r) / a[0, :, numpy.newaxis] * dt, axis=1)
    a[6] += numpy.sum(mass * (diff[2] / r) / a[0, :, numpy.newaxis] * dt, axis=1)



def move(solarsystem, asteroids, dt):
    """Move the bodies
    first find forces and change velocity and then move positions
    """

    calc_force(solarsystem, solarsystem, dt, True)
    calc_force(asteroids, solarsystem, dt, False)

    solarsystem[1:4] += solarsystem[4:7] * dt
    asteroids[1:4] += asteroids[4:7] * dt


#
# for i in solarsystem:
#        for j in solarsystem:
#            if i != j:
#                calc_force(i, j, dt)
#
#    for i in asteroids:
#        for j in solarsystem:
#                calc_force(i, j, dt)


#    
#    for i in solarsystem+asteroids:
#        i['x'] += i['vx'] * dt
#        i['y'] += i['vy'] * dt
#        i['z'] += i['vz'] * dt

def sign(x):  # This function is only used for creating the simulation - dont waste time improving it!
    if x < 0: return -1
    if x > 0: return 1
    return 0


def circlev(rx, ry, rz):  # This function is only used for creating the simulation - dont waste time improving it!
    r2 = sqrt(rx * rx + ry * ry + rz * rz)
    numerator = (6.67e-11) * 1e6 * solarmass
    return sqrt(numerator / r2)


def random_system(
        x_max,
        y_max,
        z_max,
        n,
        b
):
    maxVals = numpy.array([x_max, y_max, z_max])[:, numpy.newaxis]
    """Generate a galaxy of random bodies"""
    # n+1 to include the sun.
    # _planets = numpy.zeros(n+1,7)
    solarsystem = numpy.empty((7, n + 1))
    solarsystem[:, 0] = numpy.array([1e6 * solarmass, 0, 0, 0, 0, 0, 0])
    pos = numpy.random.rand(3, n)
    pos[2] *= .01
    dist = 1.0 / numpy.sqrt(numpy.sum(pos ** 2, axis=0)) - (.8 - numpy.random.rand(n) * .1)
    pos = maxVals * pos * dist * numpy.sign(.5 - numpy.random.rand(3, n))
    magv = circlev(pos[0],pos[1],pos[2])
    absangle = numpy.arctan(numpy.abs(pos[1] / pos[0]))
    thetav = pi / 2 - absangle

    solarsystem[1:4, 1:] = pos
    solarsystem[0, 1:] = numpy.random.random(n) * solarmass * 10 + 1e20
    solarsystem[4, 1:] = -1 * numpy.sign(pos[1]) * numpy.cos(thetav) * magv
    solarsystem[5, 1:] = numpy.sign(pos[0]) * numpy.sin(thetav) * magv
    solarsystem[6, 1:] = 0


    asteroids = numpy.empty((7, b))
    pos = numpy.random.rand(3, b)
    pos[2] *= .01
    dist = 1.0 / numpy.sqrt(numpy.sum(pos ** 2, axis= 0)) - (numpy.random.rand(b) * .2)
    pos = maxVals * pos * dist * numpy.sign(.5 - numpy.random.rand(3, b))
    magv = circlev(pos[0],pos[1],pos[2])
    absangle = numpy.arctan(numpy.abs(pos[1] / pos[0]))
    thetav = pi / 2 - absangle

    asteroids[0] = numpy.random.random(b) * solarmass * 10 + 1e14
    asteroids[1:4] = pos
    asteroids[4] = -1 * numpy.sign(pos[1]) * numpy.cos(thetav) * magv
    asteroids[5] = numpy.sign(pos[0]) * numpy.sin(thetav) * magv
    asteroids[6] = 0

    # asteroids = numpy.empty((b,7))
    # for i in xrange(b):
    #     px, py,pz = random(), random(), random()*.01
    #     dist = (1.0/sqrt(px*px+py*py+pz*pz))-(random()*.2)
    #     px = x_max*px*dist*sign(.5-random())
    #     py = y_max*py*dist*sign(.5-random())
    #     pz = z_max*pz*dist*sign(.5-random())
    #     magv = circlev(px,py, pz)
    #
    #     absangle = arctan(abs(py/px))
    #     thetav= pi/2-absangle
    #     vx   = -1*sign(py)*cos(thetav)*magv
    #     vy   = sign(px)*sin(thetav)*magv
    #     vz   = 0
    #     mass = random()*solarmass*10+1e14;
    #     asteroids[i] = numpy.array([mass,px,py,pz,vx,vy,vz])
    #
    #
    # for i in xrange(n):
    #     px, py,pz = random(), random(), random()*.01
    #     dist = (1.0/sqrt(px*px+py*py+pz*pz))-(.8-random()*.1)
    #     px = x_max*px*dist*sign(.5-random())
    #     py = y_max*py*dist*sign(.5-random())
    #     pz = z_max*pz*dist*sign(.5-random())
    #     magv = circlev(px,py, pz)
    #
    #     absangle = arctan(abs(py/px))
    #     thetav= pi/2-absangle
    #     vx   = -1*sign(py)*cos(thetav)*magv
    #     vy   = sign(px)*sin(thetav)*magv
    #     vz   = 0
    #     mass = random()*solarmass*10+1e20;
    #     solarsystem[i+1] = numpy.array([mass,px,py,pz,vx,vy,vz])

    return solarsystem, asteroids

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
import numpy as np
from numpy import exp, arctan, sqrt, pi, cos, sin
from numpy.random import random


G = 6.673e-11
solarmass=1.98892e30



#def calc_force(a, b, dt):
#    """Calculate forces between bodies
#    F = ((G m_a m_b)/r^2)*((x_b-x_a)/r)
#    """
#    #r = np.sum((b[:,-3:]-a[np.newaxis,-3:])**2,axis = 1)**0.5
#    r = ((b['x'] - a['x']) ** 2 + (b['y'] - a['y']) ** 2 + (b['z'] - a['z']) ** 2) ** 0.5
#    a['vx'] += G * a['m'] * b['m'] / r ** 2 * ((b['x'] - a['x']) / r) / a['m'] * dt
#    a['vy'] += G * a['m'] * b['m'] / r ** 2 * ((b['y'] - a['y']) / r) / a['m'] * dt
#    a['vz'] += G * a['m'] * b['m'] / r ** 2 * ((b['z'] - a['z']) / r) / a['m'] * dt

def calc_force(a, b, dt):
    """Calculate forces between bodies
    F = ((G m_a m_b)/r^2)*((x_b-x_a)/r)
    """
    r = np.sum((b[:,1:4]-a[np.newaxis,1:4])**2,axis = 1)**0.5

    a[4:7] += np.sum((G*a[0]*b[:,0] / (r**2))[:,np.newaxis] * 
        ((b[:,1:4] - a[1:4])/r[:,np.newaxis]) /a[0] * dt,axis = 0)
#    a[4] += np.sum(G*a[0]*b[:,0] / (r**2) * ((b[:,1] - a[1])/r) /a[0] * dt)
#    a[5] += np.sum(G*a[0]*b[:,0] / (r**2) * ((b[:,2] - a[2])/r) /a[0] * dt)
#    a[6] += np.sum(G*a[0]*b[:,0] / (r**2) * ((b[:,3] - a[3])/r) /a[0] * dt)

def move(solarsystem, asteroids, dt):
    """Move the bodies
    first find forces and change velocity and then move positions
    """
    for i,planet in enumerate(solarsystem):
        calc_force(planet,solarsystem[np.arange(len(solarsystem)) != i],dt)
        
    for ast in asteroids:
        calc_force(ast,solarsystem,dt)
    
    asteroids[:,1:4]   += asteroids[:,4:7] * dt   
    solarsystem[:,1:4] += solarsystem[:,4:7] * dt   
    
#    for i in solarsystem:
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

def sign(x):           #This function is only used for creating the simulation - dont waste time improving it!
    if x<0: return -1
    if x>0: return 1
    return 0

def circlev(rx, ry, rz): #This function is only used for creating the simulation - dont waste time improving it!
    r2=sqrt(rx*rx+ry*ry+rz*rz)
    numerator=(6.67e-11)*1e6*solarmass
    return sqrt(numerator/r2)


def random_system(
    x_max,
    y_max,
    z_max,
    n,
    b
    ):
    """Generate a galaxy of random bodies"""
    # n+1 to include the sun.
    #_planets = np.zeros(n+1,7)
    solarsystem = np.zeros((n+1,7))
    solarsystem[0] = np.array([1e6*solarmass,0,0,0,0,0,0])
    for i in xrange(n):
        px, py,pz = random(), random(), random()*.01
        dist = (1.0/sqrt(px*px+py*py+pz*pz))-(.8-random()*.1)
        px = x_max*px*dist*sign(.5-random())
        py = y_max*py*dist*sign(.5-random())
        pz = z_max*pz*dist*sign(.5-random())
        magv = circlev(px,py, pz)
            
        absangle = arctan(abs(py/px))
        thetav= pi/2-absangle
        vx   = -1*sign(py)*cos(thetav)*magv
        vy   = sign(px)*sin(thetav)*magv
        vz   = 0
        mass = random()*solarmass*10+1e20;
        solarsystem[i+1] = np.array([mass,px,py,pz,vx,vy,vz])
        
    asteroids = np.zeros((b,7))
    for i in xrange(b):
        px, py,pz = random(), random(), random()*.01
        dist = (1.0/sqrt(px*px+py*py+pz*pz))-(random()*.2)
        px = x_max*px*dist*sign(.5-random())
        py = y_max*py*dist*sign(.5-random())
        pz = z_max*pz*dist*sign(.5-random())
        magv = circlev(px,py, pz)
        
        absangle = arctan(abs(py/px))
        thetav= pi/2-absangle
        vx   = -1*sign(py)*cos(thetav)*magv
        vy   = sign(px)*sin(thetav)*magv
        vz   = 0
        mass = random()*solarmass*10+1e14;
        asteroids[i] = np.array([mass,px,py,pz,vx,vy,vz])
        
#    
#    solarsystem = [{'m': 1e6*solarmass, 'x': 0, 'y': 0, 'z': 0, 'vx': 0, 'vy': 0, 'vz': 0}]
#    for i in xrange(n):
#        px, py,pz = random(), random(), random()*.01
#        dist = (1.0/sqrt(px*px+py*py+pz*pz))-(.8-random()*.1)
#        px = x_max*px*dist*sign(.5-random())
#        py = y_max*py*dist*sign(.5-random())
#        pz = z_max*pz*dist*sign(.5-random())
#        magv = circlev(px,py, pz)
#            
#        absangle = arctan(abs(py/px))
#        thetav= pi/2-absangle
#        vx   = -1*sign(py)*cos(thetav)*magv
#        vy   = sign(px)*sin(thetav)*magv
#        vz   = 0
#        mass = random()*solarmass*10+1e20;
#        solarsystem.append({'m':mass, 'x':px, 'y':py, 'z':pz, 'vx':vx, 'vy':vy, 'vz':vz})

#    asteroids = []
#    for i in xrange(b):
#        px, py,pz = random(), random(), random()*.01
#        dist = (1.0/sqrt(px*px+py*py+pz*pz))-(random()*.2)
#        px = x_max*px*dist*sign(.5-random())
#        py = y_max*py*dist*sign(.5-random())
#        pz = z_max*pz*dist*sign(.5-random())
#        magv = circlev(px,py, pz)
#        
#        absangle = arctan(abs(py/px))
#        thetav= pi/2-absangle
#        vx   = -1*sign(py)*cos(thetav)*magv
#        vy   = sign(px)*sin(thetav)*magv
#        vz   = 0
#        mass = random()*solarmass*10+1e14;
#        asteroids.append({'m':mass, 'x':px, 'y':py, 'z':pz, 'vx':vx, 'vy':vy, 'vz':vz})

    return solarsystem, asteroids




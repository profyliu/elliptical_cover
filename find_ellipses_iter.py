# -*- coding: utf-8 -*-
"""
Created on Wed May  5 13:21:56 2021

@author: gn0061
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import dot,empty_like
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull
from gams import *
import os
from matplotlib.backends.backend_pdf import PdfPages
import sys
import time
import csv
import utm
import simplekml
from polycircles import polycircles
import pandas as pd
from shapely.geometry import Point, Polygon
from operator import itemgetter
import networkx as nx
from gams import *
from smallestenclosingcircle import make_circle

def find_min_ellipse():
    return '''
* Find the "smallest" ellipse that cover the given points on the plane
* There are three types of points i0, i1, i2
$if not set gdxfile $set gdxfile _gams_py_gdb0.gdx
$if not set slack $set slack 0
set i0, i1, i2;
parameter
    xi0(i0), yi0(i0)
    xi1(i1), yi1(i1)
    xi2(i2), yi2(i2)
    ui1(i1), ui2(i2)
;
$gdxin %gdxfile%
$load i0,i1,i2
$load xi0, yi0, xi1, yi1, xi2, yi2, ui1, ui2
$gdxin
variable x1,y1,x2,y2,objval;
equations c_dist0(i0), c_dist1(i1), c_dist2(i2);
c_dist0(i0)..
    objval =g= sqrt(sqr(xi0(i0)-x1) + sqr(yi0(i0)-y1)) + sqrt(sqr(xi0(i0)-x2) + sqr(yi0(i0)-y2));
c_dist1(i1)..
    objval =g= sqrt(sqr(xi1(i1)-x1) + sqr(yi1(i1)-y1)) + ui1(i1) - %slack%;
c_dist2(i2)..
    objval =g= sqrt(sqr(xi2(i2)-x2) + sqr(yi2(i2)-y2)) + ui2(i2) - %slack%;

model find_ellipse /c_dist0,c_dist1,c_dist2/;
option nlp=baron, optcr=0, optca=0;
solve find_ellipse min objval using nlp;
'''

def distance_two_points(p1, p2):
    distp1p2 = np.linalg.norm(p1 - p2)
    return distp1p2

n = 30
p = 4
update_assignment = 0  # 1 full heuristic; 0 fix initial assignment and only adjust focii location
use_slack = 1 # whether or not to use a decreasing slack in the algorithm
init_assign_by = 'random'  # random, fix or distance
points = np.empty((n, 2))
np.random.seed(2021)
for i in range(n):
    #candidate = np.around(100*np.random.random(2), decimals = 10)
    candidate_x = np.around(0+100*np.random.random(1), decimals = 10)
    candidate_y = np.around(0+30*np.random.random(1), decimals = 10)
    points[i] = [candidate_x, candidate_y]

# Prepare memory
focii = np.empty((p,2))
dij = np.empty((n,p))
zij = np.empty((n,p), dtype=int)
sum_di = np.empty(n)
typei = np.empty(n)
iterlog = np.empty((0,4))


# random sample of initial focii
np.random.seed()  # no seed, totally random
for j in range(p):
    candidate_x = np.around(0+100*np.random.random(1), decimals = 10)
    candidate_y = np.around(0+50*np.random.random(1), decimals = 10)
    focii[j] = [candidate_x, candidate_y]
# initial distance calculation
for i in range(n):
    for j in range(p):
        dij[i,j] = distance_two_points(points[i,:], focii[j,:])


# revise assignment or not
if init_assign_by == 'random':
    np.random.seed(2049)
    for i in range(n):
        rand_pos = np.random.choice(range(p),size=2, replace=False)
        zij[i,:] = 0
        zij[i,rand_pos] = 1
elif init_assign_by == 'distance':
    for i in range(n):
        d2nd = np.sort(dij[i,:])[1]
        for j in range(p):
            if dij[i,j] <= d2nd:
                zij[i,j] = 1
            else:
                zij[i,j] = 0
else:
    sys.exit('specify init_assign_by')

last_maxd = 10000
toler = 1e-8
itercnt = 0
while True:
    # Run iteration
    itercnt = itercnt + 1   
    if use_slack == 1:
        slack = 100*np.exp(-itercnt)
    else:
        slack = 0
        
    for i in range(n):
        sum_di[i] = sum(np.multiply(dij[i,:], zij[i,:]))
    maxi = np.argmax(sum_di)
    maxd = np.max(sum_di)
    
    print("iter {:d}, sum_dist {:.5f}".format(itercnt, maxd))
    iterlog = np.concatenate((iterlog, [[itercnt, maxd, last_maxd - maxd, slack]]), axis=0)
    if np.abs(last_maxd - maxd) < toler:
        break
    else:
        last_maxd = maxd
    
    plt.plot(points[:,0], points[:,1], 'ro')
    plt.plot(focii[:,0], focii[:,1], 'bs')
    i = maxi
    for j in range(p):
        if zij[i,j] == 1:
            plt.plot([points[i,0], focii[j,0]], [points[i,1], focii[j,1]], 'k--')
    plt.axis('equal')
    plt.show()
    
    
    focus1 = -1
    focus2 = -1
    for j in range(p):
        if focus1 == -1:
            if zij[maxi,j] == 1:
                focus1 = j
        else:
            if zij[maxi,j] == 1:
                focus2 = j
                break
        
    for i in range(n):
        if zij[i,focus1] == 1:
            if zij[i,focus2] == 1:
                typei[i] = 0
            else:
                typei[i] = 1           
        else:
            if zij[i,focus2] == 1:
                typei[i] = 2
            else:
                typei[i] = 3  # Not relevant
    
    ws = GamsWorkspace()
    gdb = ws.add_database()
    opt = ws.add_options()
    opt.defines["gdxfile"] = gdb.name
    opt.defines['slack'] = str(slack)
    opt.nlp = 'baron'
    
    i0_g = gdb.add_set("i0", 1, "set of points connected to both focii")
    i1_g = gdb.add_set("i1", 1, "set of points connected to focus 1")
    i2_g = gdb.add_set("i2", 1, "set of points connected to focus 2")
    xi0_g = gdb.add_parameter_dc("xi0", ['i0'], "x coordinate of point i0")
    yi0_g = gdb.add_parameter_dc("yi0", ['i0'], "y coordinate of point i0")
    xi1_g = gdb.add_parameter_dc("xi1", ['i1'], "x coordinate of point i1")
    yi1_g = gdb.add_parameter_dc("yi1", ['i1'], "y coordinate of point i1")
    xi2_g = gdb.add_parameter_dc("xi2", ['i2'], "x coordinate of point i2")
    yi2_g = gdb.add_parameter_dc("yi2", ['i2'], "y coordinate of point i2")
    
    ui1_g = gdb.add_parameter_dc("ui1",['i1'], 'distance to the other focus')
    ui2_g = gdb.add_parameter_dc("ui2",['i2'], 'distance to the other focus')
    for i in range(n):
        if typei[i] == 0:
            i0_g.add_record('i'+str(i))
            xi0_g.add_record('i'+str(i)).value = points[i,0]
            yi0_g.add_record('i'+str(i)).value = points[i,1]
        elif typei[i] == 1:
            i1_g.add_record('i'+str(i))
            xi1_g.add_record('i'+str(i)).value = points[i,0]
            yi1_g.add_record('i'+str(i)).value = points[i,1]
            temp = np.copy(zij[i,])
            temp[focus1] = 0
            other = np.argmax(temp)
            ui1_g.add_record('i'+str(i)).value = distance_two_points(points[i,:], focii[other,:])
        elif typei[i] == 2:
            i2_g.add_record('i'+str(i))
            xi2_g.add_record('i'+str(i)).value = points[i,0]
            yi2_g.add_record('i'+str(i)).value = points[i,1]
            temp = np.copy(zij[i,])
            temp[focus2] = 0
            other = np.argmax(temp)
            ui2_g.add_record('i'+str(i)).value = distance_two_points(points[i,:], focii[other,:])        
              
    t1 = ws.add_job_from_string(find_min_ellipse())
    t1.run(gams_options = opt, databases = gdb)
    
    objval_g = t1.out_db['objval'].find_record().level
    new_focus1_x = t1.out_db['x1'].find_record().level
    new_focus1_y = t1.out_db['y1'].find_record().level
    new_focus2_x = t1.out_db['x2'].find_record().level
    new_focus2_y = t1.out_db['y2'].find_record().level
    focii[focus1,:] = [new_focus1_x, new_focus1_y]
    focii[focus2,:] = [new_focus2_x, new_focus2_y]

    # Update point-focus distance and assignments based on updated focii locations
    for i in range(n):
        for j in range(p):
            dij[i,j] = distance_two_points(points[i,:], focii[j,:])
        if update_assignment == 1:
            d2nd = np.sort(dij[i,:])[1]
            for j in range(p):
                if dij[i,j] <= d2nd:
                    zij[i,j] = 1
                else:
                    zij[i,j] = 0
    
# print log
for i in range(len(iterlog)):
    print("iter: {:.0f} dist: {:.8f} chg: {:.8f} slack: {:.8f}".format(iterlog[i,0],iterlog[i,1],iterlog[i,2],iterlog[i,3]))

# Plot solution
n_used = 0
for j1 in range(p-1):
    for j2 in range(j1+1,p):
        this_used = 0
        for i in range(n):
            if zij[i,j1] == 1 and zij[i,j2] == 1:
                this_used = 1
                break
        if this_used == 0:
            continue
        n_used = n_used + 1
        a1 = focii[j1,0]
        b1 = focii[j1,1]
        a2 = focii[j2,0]
        b2 = focii[j2,1]
        c = objval_g
        print("{:.3f} {:.3f} {:.3f} {:.3f} sum dist: {:.3f}".format(a1,b1,a2,b2,c))
        # Compute ellipse parameters
        a = c / 2                                # Semimajor axis
        x0 = (a1 + a2) / 2                       # Center x-value
        y0 = (b1 + b2) / 2                       # Center y-value
        f = np.sqrt((a1 - x0)**2 + (b1 - y0)**2) # Distance from center to focus
        b = np.sqrt(a**2 - f**2)                 # Semiminor axis
        phi = np.arctan2((b2 - b1), (a2 - a1))   # Angle betw major axis and x-axis
        # Parametric plot in t
        resolution = 1000
        t = np.linspace(0, 2*np.pi, resolution)
        x = x0 + a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)
        y = y0 + a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)
        # Plot ellipse
        plt.plot(x, y)
        # Show focii
        plt.plot(a1, b1, 'bs')
        plt.plot(a2, b2, 'bs')
plt.plot(points[:,0], points[:,1], 'ro')
plt.axis('equal')
plt.title("Iter: {:d} dist: {:.8f} used: {:d}".format(itercnt,objval_g,n_used))
plt.show()
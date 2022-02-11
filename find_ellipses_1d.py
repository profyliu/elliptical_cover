# -*- coding: utf-8 -*-
"""
Created on Mon May 24 09:54:45 2021

@author: gn0061
"""

# Given a set of n demand points on the real line, determine the location of p landmarks such that the maximum sum of distances from a demand point to its two nearest landmarks is minimized.

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

def distance_two_points(p1, p2):
    distp1p2 = np.linalg.norm(p1 - p2)
    return distp1p2

'''
n = 10
points = np.empty(n)
for i in range(n):
    points[i] = np.around(100*np.random.random(), decimals = 2)
'''

#points = np.array([0,19,63,69,80,85,92,100])
#points = np.array([0,19,43,69,80,85,92,100])

points = np.array([0.00000,  90.00000,  98.75000, 99.79167,  99.94792,  99.97396, 100.00000])  # worst case for p = 11

p = 11

global global_count
global_count = 0

def find_ellipses_1d(points, p):
    global global_count
    global_count = global_count + 1
    points = np.sort(points)
    min_point = np.min(points)
    max_point = np.max(points)
    cuts = np.linspace(min_point, max_point, p)
    el = (max_point - min_point)/(p-1)
    ok = 1
    for i in range(p-1):
        if not np.any((points >= cuts[i]) & (points <= cuts[i+1])):
            ok = 0
            break
    if ok == 1:
        el = (max_point - min_point)/(p-1)
        return(el, cuts)
    
    n = len(points)
    bp = np.empty(0, dtype=int)
    for i in range(n-1):
        if points[i+1] - points[i] > el:
            bp = np.concatenate((bp,[i]))
    if len(bp) == 0:
        print("impossible. something is wrong")   

    final_cuts = {}
    global_objval = np.empty(len(bp))
    for k in range(len(bp)):
        clus = {}
        clus[0] = points[0:(bp[k]+1)]
        clus[1] = points[(bp[k]+1):n]
        clus_p = [2,2]
        clus_objval = [points[bp[k]] - points[0], points[n-1] - points[bp[k]+1]]
        clus_cuts = {}
        clus_cuts[0] = [points[0], points[bp[k]]]
        clus_cuts[1] = [points[bp[k]+1], points[n-1]]
        n_free_cuts = p-4
        max_clus_indx = np.argmax(clus_objval)
        global_objval[k] = np.max(clus_objval)
        while n_free_cuts > 0:
            this_val, this_cuts = find_ellipses_1d(clus[max_clus_indx], clus_p[max_clus_indx] + 1)
            n_free_cuts = n_free_cuts - 1
            clus_objval[max_clus_indx] = this_val
            clus_p[max_clus_indx] = clus_p[max_clus_indx] + 1
            clus_cuts[max_clus_indx] = this_cuts
            max_clus_indx = np.argmax(clus_objval)
            global_objval[k] = np.max(clus_objval)
        final_cuts[k] = np.empty(0)
        for j in range(2):
            final_cuts[k] = np.concatenate((final_cuts[k], clus_cuts[j]))
    which_k_best = np.argmin(global_objval)
    return(global_objval[which_k_best], final_cuts[which_k_best])
    
    
def find_ellipses_1d_heur(points, p):
    global global_count
    global_count = global_count + 1
    points = np.sort(points)
    min_point = np.min(points)
    max_point = np.max(points)
    cuts = np.linspace(min_point, max_point, p)
    '''
    plt.plot(points, [0 for i in range(len(points))], 'x')
    plt.plot(cuts, [0 for i in range(len(cuts))], '|', markersize=16)
    plt.show()
    '''
    head_indx = 0
    j = 0
    clus = {}
    for i in range(p-1):
        if cuts[i+1] < points[head_indx]:
            continue
        if not np.any((points >= cuts[i]) & (points <= cuts[i+1])):
            clus[j] = points[(points >= points[head_indx]) & (points <= cuts[i])]
            j = j+1
            head_indx = np.argmin(points <= cuts[i+1])
    clus[j] = points[points >= points[head_indx]]
    j = j+1
    nclus = j
    
    if nclus == 1:
        return((max_point - min_point)/(p-1), cuts)
    
    n_free_cuts = p - 2*nclus
    clus_objval = [10000 for j in range(nclus)]
    clus_p = [2 for j in range(nclus)]
    clus_cuts = {}
    for j in range(nclus):
        clus_objval[j] = np.max(clus[j]) - np.min(clus[j])
        clus_cuts[j] = [np.min(clus[j]), np.max(clus[j])]
    max_clus_indx = np.argmax(clus_objval)
    global_objval = np.max(clus_objval)
    while n_free_cuts > 0:
        this_val, this_cuts = find_ellipses_1d(clus[max_clus_indx], clus_p[max_clus_indx] + 1)
        n_free_cuts = n_free_cuts - 1
        clus_objval[max_clus_indx] = this_val
        clus_p[max_clus_indx] = clus_p[max_clus_indx] + 1
        clus_cuts[max_clus_indx] = this_cuts
        max_clus_indx = np.argmax(clus_objval)
        global_objval = np.max(clus_objval)
    final_cuts = np.empty(0)
    for j in range(nclus):
        final_cuts = np.concatenate((final_cuts, clus_cuts[j]))
    return(global_objval, final_cuts)

        
v, cuts = find_ellipses_1d(points,p)

ax = plt.axes([0,0,1,1], frameon=False)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.autoscale(tight=True)
plt.axis('scaled')
plt.xlim((-10,110))
plt.ylim((-2,2))
plt.plot(points, [0 for i in range(len(points))], 'o', markersize=5)
plt.plot(cuts, [0 for i in range(len(cuts))], '|', markersize=40)
plt.axis('off')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
#plt.show()
plt.savefig('worst_case_demo.pdf')


# Use MINLP 
def find_ellipses_1d_g():
    return '''
$if not set gdxfile $set gdxfile _gams_py_gdb0.gdx
set i;
set j;
parameter xi(i);
$gdxin %gdxfile%
$load i, j, xi
$gdxin
variable xj(j), objval;
binary variable z(i,j);
equations assign(i), c_dist(i);
assign(i)..
    sum(j, z(i,j)) =e= 2;
c_dist(i)..
    objval =g= sum(j,z(i,j)*abs(xi(i)-xj(j)));
model find_ellipse /c_dist, assign/;
option minlp=baron, optcr=0;
solve find_ellipse min objval using minlp;
'''

def find_ellipses_1d_gams(points, p):
    ws = GamsWorkspace()
    gdb = ws.add_database()
    opt = ws.add_options()
    opt.defines["gdxfile"] = gdb.name
    opt.minlp = 'baron'
    n = len(points)
    i_g = gdb.add_set("i", 1, "set of given points")
    j_g = gdb.add_set("j", 1, "set of focus points")
    xi_g = gdb.add_parameter_dc("xi", ['i'], "x coordinate of point i")
    for j in range(p):
        j_g.add_record('j'+str(j))  
    for i in range(n):
        i_g.add_record('i'+str(i))
        xi_g.add_record('i'+str(i)).value = float(points[i])
    t1 = ws.add_job_from_string(find_ellipses_1d_g())
    t1.run(gams_options = opt, databases = gdb)
    v_g = t1.out_db['objval'].find_record().level
    cuts_g = []
    for j in range(p):
        cuts_g = np.append(cuts_g, t1.out_db['xj'].find_record('j'+str(j)).level)
    # Plot gams solution
    plt.plot(points, [0 for i in range(len(points))], 'x')
    plt.plot(cuts_g, [0 for i in range(len(cuts))], '|', markersize=16)
    plt.show()
    return(v_g, cuts_g)
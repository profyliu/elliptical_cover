# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 14:18:09 2021

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

def distance_two_points(p1, p2):
    distp1p2 = np.linalg.norm(p1 - p2)
    return distp1p2

global global_count

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


def find_ellipses_1d_bad(points, p):
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
    #if head_indx != len(points)-1:
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
option minlp=baron, reslim=1800, optcr=0;
solve find_ellipse min objval using minlp;
parameter modstat, solstat;
modstat = find_ellipse.modelstat;
solstat = find_ellipse.solvestat;
display modstat, solstat, objval.l;
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
    modstat = t1.out_db['modstat'].find_record().value
    solstat = t1.out_db['solstat'].find_record().value
    cuts_g = []
    for j in range(p):
        cuts_g = np.append(cuts_g, t1.out_db['xj'].find_record('j'+str(j)).level)
    #print(ws.working_directory)
    return(v_g, cuts_g, modstat, solstat)

def find_ellipses_1d_mip_g():
    return '''
$if not set gdxfile $set gdxfile _gams_py_gdb0.gdx
set i;
set j;
alias(j,j1,j2);
parameter xi(i);
$gdxin %gdxfile%
$load i, j, xi
$gdxin
scalar M;
M = 2*(smax(i,xi(i)) - smin(i,xi(i)));
variable xj(j), objval;
binary variable z(i,j1,j2);
equations assign(i), c_dist1(i,j1,j2),c_dist2(i,j1,j2),c_dist3(i,j1,j2),c_dist4(i,j1,j2);
assign(i)..
    sum((j1,j2)$(ord(j1) < ord(j2)), z(i,j1,j2)) =e= 1;
c_dist1(i,j1,j2)$(ord(j1) < ord(j2))..
    objval =g= (xi(i) - xj(j1)) + (xi(i) - xj(j2)) - (1 - z(i,j1,j2))*M;
c_dist2(i,j1,j2)$(ord(j1) < ord(j2))..
    objval =g= -(xi(i) - xj(j1)) + (xi(i) - xj(j2)) - (1 - z(i,j1,j2))*M;
c_dist3(i,j1,j2)$(ord(j1) < ord(j2))..
    objval =g= (xi(i) - xj(j1)) - (xi(i) - xj(j2)) - (1 - z(i,j1,j2))*M;
c_dist4(i,j1,j2)$(ord(j1) < ord(j2))..
    objval =g= -(xi(i) - xj(j1)) - (xi(i) - xj(j2)) - (1 - z(i,j1,j2))*M;
model find_ellipse /c_dist1, c_dist2,c_dist3,c_dist4, assign/;
option mip=cplex, reslim=1800, optcr=0;
solve find_ellipse min objval using mip;
parameter modstat, solstat;
modstat = find_ellipse.modelstat;
solstat = find_ellipse.solvestat;
display modstat, solstat, objval.l;
'''

def find_ellipses_1d_mip_gams(points, p):
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
    t1 = ws.add_job_from_string(find_ellipses_1d_mip_g())
    t1.run(gams_options = opt, databases = gdb)
    v_g = t1.out_db['objval'].find_record().level
    modstat = t1.out_db['modstat'].find_record().value
    solstat = t1.out_db['solstat'].find_record().value
    cuts_g = []
    for j in range(p):
        cuts_g = np.append(cuts_g, t1.out_db['xj'].find_record('j'+str(j)).level)
    return(v_g, cuts_g, modstat, solstat)


ns = [10,20,50,100]
'''
for n in ns:
    points = np.empty(n)
    for i in range(n):
        points[i] = np.around(100*np.random.random(), decimals = 2)
    np.save('test1dn'+str(n)+'.npy', points)
'''


for n in [50]:
    points = np.load('test1dn'+str(n)+'.npy')
    for p in [4,6,8,10]:
        start = time.time()
        v_mip, cuts_mip, modstat_mip, solstat_mip = find_ellipses_1d_mip_gams(points, p)
        time_mip = time.time() - start
        global_count = 0
        start = time.time()
        v_fast, cuts_fast = find_ellipses_1d(points, p)
        time_fast = time.time() - start
        print("n {:d} p {:d} v_mip {:.3f} t_mip {:.1f} m_mip {:d} s_mip {:d} v_fast {:.3f} t_fast {:.1f} g_fast {:d}".format(n, p, v_mip, time_mip, int(modstat_mip), int(solstat_mip), v_fast, time_fast, global_count))
        file1 = open("log_1d.txt", 'a')
        file1.writelines(" {:d}  {:d}  {:.3f}  {:.1f}  {:d}  {:d} {:.3f} {:.1f} {:d}".format(n, p, v_mip, time_mip, int(modstat_mip), int(solstat_mip), v_fast, time_fast, global_count))
        file1.writelines("\n")
        file1.close()
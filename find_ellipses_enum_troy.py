# -*- coding: utf-8 -*-
"""
Created on Thu May  6 13:57:27 2021

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
import argparse
import timeit

def locate_all_ellipses():
    return '''
* Find ellipses given assignments
$if not set n $set n 3
$if not set gdxfile $set gdxfile _gams_py_gdb0.gdx
set i;
set j;
alias(j,j1,j2);
parameter xi(i), yi(i);
parameter z(i,j1,j2) assignment;
*xi(i) = uniform(0,1);
*yi(i) = uniform(0,1);
$gdxin %gdxfile%
$load i, j, xi, yi, z
$gdxin
variable xj(j), yj(j), objval;
equations c_dist(i,j1,j2);
c_dist(i,j1,j2)$(z(i,j1,j2))..
    objval =g= sqrt(sqr(xi(i)-xj(j1)) + sqr(yi(i)-yj(j1))) + sqrt(sqr(xi(i)-xj(j2)) + sqr(yi(i)-yj(j2)));
model locate_ellipse /c_dist/;
option nlp=baron, optcr=0, optca=0;
solve locate_ellipse min objval using nlp;
'''

def distance_two_points(p1, p2):
    distp1p2 = np.linalg.norm(p1 - p2)
    return distp1p2


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", type=str, default = "troy20", help="path to the input point file")
ap.add_argument("-p", "--ndepots", type=int, default=5, help="number of depots")
ap.add_argument("-v", "--verbose", type=int, default=0, help="verbose level")
ap.add_argument("-m", "--mult", type=int, default=10, help="number of multistart trials")
ap.add_argument("-l", "--log", type=int, default=1, help="1: print and plot iterlog; 0 do not")
args = vars(ap.parse_args())

points_filename = args['file']
p = args['ndepots']
verbose = args['verbose']
mult = args['mult']


points = np.load(points_filename + '.npy')
n = len(points)


sol_focus = np.empty((0,2))  # store found foci locations
sol_val = np.empty(0)
sol_niter = np.empty(0, dtype=int)
sol_best_val = 100000
sol_best_focii = np.empty((p,2))
sol_time = np.zeros(mult)
sol_iterlog = np.empty((mult,2), dtype=int)
when_best = 0 

# Prepare memory

focii = np.empty((p,2))
zij = np.empty((n,p), dtype=int)
dij = np.empty((n,p))
sum_di = np.empty(n)

# Start

np.random.seed()  # no seed, totally random

for iter in range(mult):
    start_time = time.time()
    # randomly scatter focii locations and make point-focus assignments by distance
    for j in range(p):
        candidate = np.around(0+100*np.random.random(2), decimals = 10)
        focii[j] = candidate
    for i in range(n):
        for j in range(p):
            dij[i,j] = distance_two_points(points[i,:], focii[j,:])
        d2nd = np.sort(dij[i,:])[1]
        for j in range(p):
            if dij[i,j] <= d2nd:
                zij[i,j] = 1
            else:
                zij[i,j] = 0
    
    iterlog = np.empty((0,3))
    
    last_objval = 10000
    toler = 1e-8
    itercnt = 0
    while True:
        # Solve for all focii locations in one-shot given the assignment
        ws = GamsWorkspace()
        gdb = ws.add_database()
        opt = ws.add_options()
        opt.defines["gdxfile"] = gdb.name
        opt.nlp = 'baron'
        
        i_g = gdb.add_set("i", 1, "set of given points")
        j_g = gdb.add_set("j", 1, "set of focus points")
        xi_g = gdb.add_parameter_dc("xi", ['i'], "x coordinate of point i")
        yi_g = gdb.add_parameter_dc("yi", ['i'], "y coordinate of point i")
        z_g = gdb.add_parameter_dc("z", ['i','j','j'], "assignment")
        for j in range(p):
            j_g.add_record('j'+str(j))  
        for i in range(n):
            i_g.add_record('i'+str(i))
            xi_g.add_record('i'+str(i)).value = points[i,0]
            yi_g.add_record('i'+str(i)).value = points[i,1]
        for i in range(n):
            focus1 = -1
            focus2 = -1
            for j in range(p):
                if focus1 == -1:
                    if zij[i,j] == 1:
                        focus1 = j
                else:
                    if zij[i,j] == 1:
                        focus2 = j
                        break
            z_g.add_record(['i'+str(i), 'j'+str(focus1), 'j'+str(focus2)]).value = 1        
        
        t1 = ws.add_job_from_string(locate_all_ellipses())
        t1.run(gams_options = opt, databases = gdb)
        
        objval_g = t1.out_db['objval'].find_record().level
        focus_used = np.sum(zij,axis=0)
        for j in range(p):
            if focus_used[j] > 0:
                focii[j,:] = [t1.out_db['xj'].find_record('j'+str(j)).level, t1.out_db['yj'].find_record('j'+str(j)).level]
        
        active_p = np.empty(n)
        for i in range(n):
            j1j2 = np.where(zij[i,:] == 1)
            active_p[i] = (t1.out_db['c_dist'].find_record(['i'+str(i), 'j'+str(j1j2[0][0]), 'j'+str(j1j2[0][1])]).marginal > 0)
        
        
        # Update point-focus distance and assignments based on updated focii locations
        for i in range(n):
            for j in range(p):
                dij[i,j] = distance_two_points(points[i,:], focii[j,:])
            d2nd = np.sort(dij[i,:])[1]
            for j in range(p):
                if dij[i,j] <= d2nd:
                    zij[i,j] = 1
                else:
                    zij[i,j] = 0
        
        # log
        itercnt = itercnt + 1
        iterlog = np.concatenate((iterlog, [[itercnt, objval_g, last_objval - objval_g]]), axis=0)
        if last_objval - objval_g < toler:
            break
        else:
            last_objval = objval_g
    
    '''
    # print log
    for i in range(len(iterlog)):
        print("iter: {:.0f} dist: {:.8f} chg: {:.8f}".format(iterlog[i,0],iterlog[i,1],iterlog[i,2]))
    '''
    
    # append solution
    sol_focus = np.concatenate((sol_focus, focii), axis=0)
    sol_val = np.concatenate((sol_val, [objval_g]))
    sol_niter = np.concatenate((sol_niter, [itercnt]))
    sol_time[iter] = time.time() - start_time
    
    # update the best solution so far
    if objval_g - sol_best_val < -1e-5:
        sol_best_val = objval_g
        when_best = iter+1
        for j in range(p):
            sol_best_focii[j,:] = focii[j,:]
    
    sol_iterlog[iter,:] = [iter+1, len(np.unique(np.round(sol_val, decimals=3)))]

sol_focus_round = np.round(sol_focus, decimals=3)
sol_focus_unique = np.unique(sol_focus_round, axis=0)
sol_val_unique = np.unique(np.round(sol_val, decimals=3))


kml_file = "Troy.kml"
pmar = 30
# Load corners from KML file
border = 0
pd.set_option('display.max_colwidth',1000000)
#kml_file = sys.argv[1]
data = pd.read_table(kml_file,sep='\r\t',header=None,skip_blank_lines=False,engine='python')
foundlable = 0
for i in range(0,len(data)):
    strl = data.iloc[i].to_frame().T
    strl2 = strl.to_string()
    strlist = strl2.split()
    if strlist[2] == '<coordinates>':
        foundlable = 1
        continue
    if foundlable == 1:
        #print(strlist)
        break
minx = 1000000000000
miny = 1000000000000
maxx = 0
maxy = 0
location = list()
utmloc = dict()
for i in range(2,len(strlist)):
    location = strlist[i].split(",")
    templst = utm.from_latlon(float(location[1]),float(location[0]))
    #print(templst)
    if templst[0] <= minx:
        minx = templst[0]
    if templst[0] >= maxx:
        maxx = templst[0]
    if templst[1] <= miny:
        miny = templst[1]
    if templst[1] >= maxy:
        maxy = templst[1]
    temploc = {str(i-1):
             {'x': templst[0],
             'y': templst[1]}}
    utmloc.update(temploc)
utmnumber = templst[2]
utmletter = templst[3]
lenx = maxx - minx
leny = maxy - miny
midx = (maxx + minx)/2
midy = (maxy + miny)/2
sqlen = max(lenx,leny)
origx = midx - sqlen*(0.5 + border)
origy = midy - sqlen*(0.5 + border)

location = list()
sqloc = dict()
for i in range(3,len(strlist)):
    location = strlist[i].split(",")
    templst = utm.from_latlon(float(location[1]),float(location[0]))

    temploc = {str(i-1):
             {'x': (templst[0]-origx)/sqlen/(1+border*2)*100,
             'y': (templst[1]-origy)/sqlen/(1+border*2)*100}}
    sqloc.update(temploc)
corners = np.empty((0,2))
for i in sqloc.keys():
    corners = np.append(corners, np.array(list(sqloc[i].values())))
corners.resize((len(sqloc),2))

# Create outer boundary edges, edge_equations and polygon
edges = np.empty((0,2), dtype = int)
for i in range(len(corners)-1):
    edges = np.append(edges, np.reshape([i, i+1], (1,2)), axis = 0)
edges = np.append(edges, np.reshape([len(corners)-1, 0], (1,2)), axis = 0)
edge_equations = np.empty((len(edges),3))
for i in range(len(edges)):
    x1 = corners[edges[i,0],0]
    y1 = corners[edges[i,0],1]
    x2 = corners[edges[i,1],0]
    y2 = corners[edges[i,1],1]
    edge_equations[i,0] = y2 - y1
    edge_equations[i,1] = x1 - x2
    edge_equations[i,2] = edge_equations[i,0]*x1 + edge_equations[i,1]*y1
polygon = Polygon(corners)


# Plot solution
ax = plt.axes([0,0,1,1], frameon=False)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.autoscale(tight=False)
#plt.axis('scaled')
#plt.axis('off')
plt.xlim((-30,130))
plt.ylim((-30,130))
# Plot the polygon
x,y = polygon.exterior.xy
plt.plot(x,y, linewidth = 1, color='blue')

# Use the best solution 
for j in range(p):
    focii[j,:] = sol_best_focii[j,:]

for i in range(n):
    for j in range(p):
        dij[i,j] = distance_two_points(points[i,:], focii[j,:])
    d2nd = np.sort(dij[i,:])[1]
    for j in range(p):
        if dij[i,j] <= d2nd:
            zij[i,j] = 1
        else:
            zij[i,j] = 0
                    
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
        c = sol_best_val
        print("{:.3f} {:.3f} {:.3f} {:.3f} sum dist: {:.3f}".format(a1,b1,a2,b2,c))
        # if the ellipse is degenerate, the plot a line segment instead
        if np.abs(np.sqrt((a1-a2)**2 + (b1-b2)**2) - c) < toler:
            plt.plot([a1,a2], [b1,b2], 'bs-')
            continue
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
for i in range(n):
    plt.plot(points[i,0], points[i,1], 'ro', markersize=4)
    #plt.text(points[i,0], points[i,1], str(i), fontsize = 'medium')

plt.title("n: {:d}  p: {:d}  dist: {:.4f}  ellipses: {:d}".format(n,p,sol_best_val,n_used))
plt.axis('equal')
#plt.show()
plt.savefig(points_filename + 'p' + str(p) + '.pdf')



def project_point_to_line(point, lineseg):
    result = np.double([0,0])
    vx_lineseg = lineseg[1][0] - lineseg[0][0]
    vy_lineseg = lineseg[1][1] - lineseg[0][1]
    vx_point = point[0] - lineseg[0][0]
    vy_point = point[1] - lineseg[0][1]
    point_dot_lineseg = vx_point*vx_lineseg + vy_point*vy_lineseg
    lineseg_dot_lineseg = vx_lineseg*vx_lineseg + vy_lineseg*vy_lineseg
    l = point_dot_lineseg/lineseg_dot_lineseg
    result[0] = lineseg[0][0] + l*vx_lineseg
    result[1] = lineseg[0][1] + l*vy_lineseg
    return result

# Find the distance from a point to a line (defined by two points)
def dist_point_to_line(point, line):
    x0 = point[0]
    y0 = point[1]
    x1 = line[0,0]
    y1 = line[0,1]
    x2 = line[1,0]
    y2 = line[1,1]
    return(np.abs((x2 - x1)*(y1-y0) - (x1-x0)*(y2-y1))/np.sqrt((x2 - x1)**2 + (y2 - y1)**2))


# Find the lower bound
maxdistij = 0
maxi = maxj = 0
for i in range(n-1):
    for j in range(i+1,n):
        thisdistij = distance_two_points(points[i,:], points[j,:])
        if thisdistij > maxdistij:
            maxdistij = thisdistij
            maxi = i
            maxj = j

# rotate maxj CCW 90 degrees around point maxi
temp1 = points[maxj,:] - points[maxi,:]
temp2 = [-temp1[1], temp1[0]]
temp2 = temp2 + points[maxi,:]
points1d = np.empty(n)
the_line = np.concatenate(([points[maxi,]], [temp2]), axis=0)
for i in range(n):
    points1d[i] = dist_point_to_line(points[i,:], the_line)


def find_ellipses_1d(points, p):
    points = np.sort(points)
    min_point = np.min(points)
    max_point = np.max(points)
    cuts = np.linspace(min_point, max_point, p)
   
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
    if head_indx != len(points)-1:
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

lowerbound, cuts = find_ellipses_1d(points1d,p)






def DistSq2KML(r = 10, rate = sqlen):
    radius = r/100*rate
    return radius


# Write run log
file2 = open("result_" + points_filename + "_ellipses.txt", "a")
file2.writelines("{:d} & {:d} & {:d} & {:d} & {:d} & {:d} & {:.0f} & {:d} & {:.2f} & {:.2f} & {:.2f} \\\\ ".format(n, p, mult, when_best, len(sol_val_unique), min(sol_niter), np.median(sol_niter), max(sol_niter), np.mean(sol_time), DistSq2KML(sol_best_val), DistSq2KML(lowerbound)))
file2.writelines("\n")
file2.close()

if args['log'] == 1:
    file1 = open("unique_log_" + points_filename + "_ellipses.txt", 'a')
    for i in range(mult):
        file1.writelines("{:d} {:d} {:d}".format(p, sol_iterlog[i,0], sol_iterlog[i,1]))
        file1.writelines("\n")
    file1.close()

if verbose == 2:
    file3 = open("sol_val_" + points_filename + "_p" + str(p) + '.txt', 'w')
    for i in range(len(sol_val)):
        file3.writelines("{:.2f}".format(DistSq2KML(sol_val[i])))
        file3.writelines("\n")
    file3.close()


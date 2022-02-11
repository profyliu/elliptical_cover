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

def get_findmaxdist_py_text():
    return '''
* Given focus locations, find the point in the convex polygon that has the largest total distance to the two nearest foci
$if not set gdxfile $set gdxfile _gams_py_gdb0.gdx

set j set of foci
    facet set of facets
;
alias(facet,f);
alias(j,j1,j2);
parameter
    xj(j), yj(j)
    a(f)
    b(f)
    c(f)
    dist
    soltime
    ms, ss;

$gdxin %gdxfile%
$load facet,j
$load a, b, c, xj, yj
$gdxin

variable x, y, maxdist;
equations c1_maxdist(j1,j2), c_facet(f);

c1_maxdist(j1,j2)$(ord(j1) < ord(j2))..
    maxdist =l= sqrt(sqr(x - xj(j1)) + sqr(y - yj(j1))) + sqrt(sqr(x - xj(j2)) + sqr(y - yj(j2)));
c_facet(f)..
    a(f)*x + b(f)*y =l= c(f);
model findmaxdist /c1_maxdist, c_facet/;
option nlp=baron, optcr=0, optca=0;
soltime = timeelapsed;
solve findmaxdist max maxdist using nlp;
soltime = timeelapsed - soltime;
ms = findmaxdist.modelstat;
ss = findmaxdist.solvestat;
display maxdist.l, soltime;
'''


def distance_two_points(p1, p2):
    distp1p2 = np.linalg.norm(p1 - p2)
    return distp1p2


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", type=str, default = "Troy", help="path to the input kml file")
ap.add_argument("-p", "--ndepots", type=int, default=5, help="number of depots")
ap.add_argument("-v", "--verbose", type=int, default=0, help="verbose level")
ap.add_argument("-m", "--mult", type=int, default=10, help="multistart trials")
ap.add_argument("-s", "--stop", type=int, default=8, help="multistart stop if no objval improvement in the last m trials")
ap.add_argument("-l", "--log", type=int, default=1, help="1: print and plot iterlog; 0 do not")
ap.add_argument("-o", "--maxouter", type=int, default=100, help="maximum outer iterations")
ap.add_argument("-t", "--trial", type=int, default=1, help="trial marker")
args = vars(ap.parse_args())

p = args['ndepots']
verbose = args['verbose']
mult = args['mult']
stop_when = args['stop']
maxouter = args['maxouter']
trial_number = args['trial']


#corners = np.array([[20,0],[0,30], [0,60],[10,90],[40,100],[70,100],[80,90], [100, 50], [100,40], [80,0]]) 
    
kml_file = args['file'] +'.kml'
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



hull = ConvexHull(corners)
hull_points = corners[hull.vertices]
hull_polygon = Polygon(hull_points)


points = np.empty((0,2))
points = np.append(points, hull_points, axis = 0)
n_init_sample = 2*p

np.random.seed()
i = 0
while i < n_init_sample:
    candidate = np.around(100*np.random.random(2), decimals = 10)
    if Point(candidate).within(polygon) == True:
        points = np.append(points, [candidate], axis=0)
        i = i+1

# Plot area and initial points
'''
ax = plt.axes([0,0,1,1], frameon=False)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.autoscale(tight=False)
plt.xlim((-30,130))
plt.ylim((-30,130))
# Plot the polygon
x,y = polygon.exterior.xy
plt.plot(x,y, linewidth = 1, color='gray', linestyle = (0,(5,5)))
x,y = hull_polygon.exterior.xy
plt.plot(x,y, linewidth = 1, color = 'blue')
plt.plot(points[:,0], points[:,1], 'bo')
plt.show()
'''


def DistSq2KML(r = 10, rate = sqlen):
    radius = r/100*rate
    return radius

n = len(points)

# Prepare memory
focii = np.empty((p,2))
sol_best_focii = np.empty((p,2))

# Start
# randomly scatter focii locations and make point-focus assignments by distance
np.random.seed()  # no seed, totally random

for j in range(p):
    candidate = np.around(0+100*np.random.random(2), decimals = 10)
    focii[j] = candidate
    
outerlog = np.empty((0,5))
outercnt = 0
verbose = 1
while True:
    outercnt = outercnt + 1
    # prepare memory
    zij = np.empty((n,p), dtype=int)
    dij = np.empty((n,p))
    sol_best_val = 100000
    when_best = 0
    sol_best_val = 100000
    for iter in range(mult):
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
        
        # update the best solution so far
        if objval_g - sol_best_val < -1e-5:
            sol_best_val = objval_g
            when_best = iter
            for j in range(p):
                sol_best_focii[j,:] = focii[j,:]
        # Early termination: if no improvement in the last s iterations, stop
        if iter - when_best > stop_when:
            break
        
        # If not the last iter, randomly scatter focii in preparation for the next iteration
        if iter < mult - 1:
            for j in range(p):
                focii[j] = np.around(0+100*np.random.random(2), decimals = 10)

    # Use the best focii as focii for finding the new point
    for j in range(p):
        focii[j,:] = sol_best_focii[j,:]
    objval_g = sol_best_val
    
    # find uncovered point in the area
    ws = GamsWorkspace()
    gdb = ws.add_database()
    opt = ws.add_options()
    opt.defines["gdxfile"] = gdb.name
    opt.nlp = 'baron'        
    facet_g = gdb.add_set("facet", 1, "set of facets")
    depot_g = gdb.add_set("j", 1, "set of foci")
    a_g = gdb.add_parameter_dc("a", ['facet'], "coefficient of x")
    b_g = gdb.add_parameter_dc("b", ['facet'], "coefficient of y")
    c_g = gdb.add_parameter_dc("c", ['facet'], "rhs")
    xj_g = gdb.add_parameter_dc('xj', ['j'], 'x coordinate of focus')
    yj_g = gdb.add_parameter_dc('yj', ['j'], 'y coordinate of focus')
    
    for i in range(hull.simplices.shape[0]):
        facet_g.add_record('f'+str(i))
        a_g.add_record('f'+str(i)).value = hull.equations[i][0]
        b_g.add_record('f'+str(i)).value = hull.equations[i][1]
        c_g.add_record('f'+str(i)).value = -hull.equations[i][2]            
    
    for j in range(len(focii)):
        depot_g.add_record('j'+str(j))
        xj_g.add_record('j'+str(j)).value = focii[j][0]
        yj_g.add_record('j'+str(j)).value = focii[j][1]
    
    t2 = ws.add_job_from_string(get_findmaxdist_py_text())
    t2.run(gams_options = opt, databases = gdb)
    #t1.out_db.export(os.path.join(ws.working_directory, "tdata.gdx"))
    #ws.working_directory
    max_dist = t2.out_db["maxdist"].find_record().level
    newx = t2.out_db['x'].find_record().level
    newy = t2.out_db['y'].find_record().level

    outerlog = np.append(outerlog, [[outercnt, n, objval_g, max_dist, max_dist - objval_g]], axis=0)
    if verbose == 1:
        print(outerlog[-1,])
            
    # Add in new pivot point
    if DistSq2KML(max_dist - objval_g) > 50 and outercnt < maxouter:
        points = np.append(points, [[newx, newy]], axis=0)
        n = len(points)
    else:
        break




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
plt.savefig(args['file'] + '_p' + str(p) + '_' + str(trial_number) + '.pdf')





if args['log'] == 1:
    file1 = open("area_log_" + args['file'] + "_ellipses.txt", 'a')
    for i in range(len(outerlog)):
        file1.writelines("{:0f} {:.0f} {:.0f} {:.1f} {:.1f} {:.1f}".format(float(trial_number), outerlog[i,0], outerlog[i,1], DistSq2KML(outerlog[i,2]), DistSq2KML(outerlog[i,3]), DistSq2KML(outerlog[i,4])))
        file1.writelines("\n")
    file1.close()
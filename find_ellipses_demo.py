'''
This program demonstrates how the locate-allocate algorithm works
Run:
python find_ellipses_demo.py -n 10 -p 4 --seed 2023

n is the number of demand points, p is the number of depots, seed is for RNG.
A PDF plot will be generated.
'''
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
parser = argparse.ArgumentParser()
parser.add_argument('-n', type = int, default=10)
parser.add_argument('-p', type = int, default=4)
parser.add_argument('--seed', type = int, default = 2021)
args = vars(parser.parse_args())

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


n = args['n']
points = np.empty((n, 2))
np.random.seed(args['seed'])
for i in range(n):
    points[i,0] = np.around(100*np.random.random(1), decimals=10)
    points[i,1] = np.around(100*np.random.random(1), decimals=10)

sol_focus = np.empty((0,2))  # store found foci locations
sol_val = np.empty(0)

# Prepare memory
p = args['p']
focii = np.empty((p,2))
zij = np.empty((n,p), dtype=int)
dij = np.empty((n,p))
sum_di = np.empty(n)


iterlog = np.empty((0,3))

last_objval = 10000
toler = 1e-1
itercnt = 0

with PdfPages('ellipses_demo_n' + str(n) + '_p' + str(p) + '_s' + str(args['seed']) + '.pdf') as pdf:
    # Plot demand points
    ax = plt.axes([0,0,1,1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=False)
    plt.xlim((-30,130))
    plt.ylim((-30,130))  
    for i in range(n):
        plt.plot(points[i,0], points[i,1], 'ro')
    pdf.savefig()
    plt.close()

    # randomly scatter focii locations and make point-focus assignments by distance
    np.random.seed(args['seed']*10)  
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

    # Plot initial guess
    ax = plt.axes([0,0,1,1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=False)
    plt.xlim((-30,130))
    plt.ylim((-30,130))    
    # demand points
    for i in range(n):
        plt.plot(points[i,0], points[i,1], 'ro')
    # focus points
    for j in range(p):
        plt.plot(focii[j,0], focii[j,1], 'bs')
    pdf.savefig()
    plt.close()

    # Plot initial guess + assignment
    ax = plt.axes([0,0,1,1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=False)
    plt.xlim((-30,130))
    plt.ylim((-30,130))    
    # demand points
    for i in range(n):
        plt.plot(points[i,0], points[i,1], 'ro')
    # focus points
    for j in range(p):
        plt.plot(focii[j,0], focii[j,1], 'bs')
    # assigments
    for i in range(n):
        for j in range(p):
            if zij[i,j] == 1:
                plt.plot([points[i,0],focii[j,0]] , [points[i,1], focii[j,1]], 'b--')
    pdf.savefig()
    plt.close()

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
        
        # Plot solution
        ax = plt.axes([0,0,1,1], frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.autoscale(tight=False)
        plt.xlim((-30,130))
        plt.ylim((-30,130))  
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
            if active_p[i]:
                #plt.plot(points[i,0], points[i,1], 'mo')
                plt.plot(points[i,0], points[i,1], 'ro')
            else:
                plt.plot(points[i,0], points[i,1], 'ro')
            #plt.text(points[i,0], points[i,1], str(i), fontsize = 'medium')
        '''
        fig = plt.gcf()
        fig.set_size_inches(8,8)
        for j in range(p):
            ax = fig.gca()
            circle = plt.Circle((focii[j,0], focii[j,1]), c/2, color='b', fill=False)
            ax.add_artist(circle)
        '''
        
        #plt.plot(points[[max_i,max_j],0], points[[max_i,max_j],1], 'k-')
        pdf.savefig()
        plt.close()
        
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

        # Plot updated assignment
        ax = plt.axes([0,0,1,1], frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.autoscale(tight=False)
        plt.xlim((-30,130))
        plt.ylim((-30,130))    
        # demand points
        for i in range(n):
            plt.plot(points[i,0], points[i,1], 'ro')
        # focus points
        for j in range(p):
            plt.plot(focii[j,0], focii[j,1], 'bs')
        # assigments
        for i in range(n):
            for j in range(p):
                if zij[i,j] == 1:
                    plt.plot([points[i,0],focii[j,0]] , [points[i,1], focii[j,1]], 'b--')
        pdf.savefig()
        plt.close()
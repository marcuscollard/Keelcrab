# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:18:02 2022

@author: nbagz
"""
import sys

sys.path.append('C:/Users/nbagz/Documents/MIT/Research/ShipOp_Project')

from HullParameterization import Hull_Parameterization as HP

import numpy as np

import csv

from tqdm import tqdm

#Open the Design Vector csv
path = './Custom_Design_Set/'
filename = 'Input_Vectors.csv'
Vec = []
with open(path + filename) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        Vec.append(row)

#Save as a np.float array
Vec = np.array(Vec)

DesVec = Vec.astype(np.float64())


#loop thru to make point cloud files
for i in tqdm(range(7,8)):     #len(DesVec))):
    
    
    hull = HP(DesVec[i])
            
    PC = hull.gen_pointCloud(NUM_WL = 60, PointsPerWL = 600)
    
    pts = len(PC)
    
    #for k in range(0,pts):
           # if PC[k, 1] != 0.0:
             #   PC = np.append(PC, [[PC[k,0], -PC[k,1], PC[k,2]]], axis = 0)

     
    f = open(path + 'PC/Hull_PC_' + str(i) + '.csv', 'w')

    writer = csv.writer(f)
    
   
    for k in range(0,len(PC)):
        writer.writerow(PC[k])
    
    f.close()

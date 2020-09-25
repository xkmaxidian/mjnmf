# -*- coding: utf-8 -*-
"""
Created on Mon May  4 10:52:11 2020

@author: 12728
"""


from numpy.linalg import inv
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import metrics
import time
import scipy.sparse as sp
import sklearn.preprocessing as nor
from copy import deepcopy
from scipy.io import loadmat
from munkres import  Munkres

networkfile = open("D:\\benchmark\\benchmark\\Debug\\network.dat","rt")

node = 1000
k = 10

def takeSecond(elem):
    return elem[1]


Wold = np.zeros((node,node))
for line in networkfile:
    l = line.split()
    Wold[int(l[0])-1,int(l[1])-1] = 1
    
W = np.zeros((node,node))

communityfile = open("D:\\benchmark\\benchmark\\Debug\\community.dat","rt")

community  = []

for line in communityfile:
    l = line.split()
    community.append(( int(l[0])-1 , int(l[1])))

'''print(community)'''
community.sort(key=takeSecond)

dic = {}
for i in range(1000):
    dic[i] = community[i][0]



for i in range(node):
    for j in range(node):
        W[i,j] = Wold[dic[i],dic[j]]

print(W)

networkfile.close()
communityfile.close()

np.set_printoptions(suppress=True)

np.save("C:\\Users\\12728\Desktop\\multiview\\artificialnetwork\\y5.npy",W)


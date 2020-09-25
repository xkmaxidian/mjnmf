# -*- coding: utf-8 -*-
"""
Created on Fri May  1 08:52:58 2020

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



n = 8

def corr(p,q):
    return np.mean(np.multiply((p-np.mean(p)),(q-np.mean(q))))/(np.std(q)*np.std(p))

b1 = np.load("C:\\Users\\12728\\Desktop\\multiview\\math\\selectk\\h1.npy")
b2 = np.load("C:\\Users\\12728\\Desktop\\multiview\\math\\selectk\\h2.npy")
b3 = np.load("C:\\Users\\12728\\Desktop\\multiview\\math\\selectk\\h3.npy")
b4 = np.load("C:\\Users\\12728\\Desktop\\multiview\\math\\selectk\\h4.npy")
b5 = np.load("C:\\Users\\12728\\Desktop\\multiview\\math\\selectk\\h5.npy")



def geth(p,q):
    r = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            r[i,j]= corr(p[:,i],q[:,j])
    return r        
            
h12 = geth(b1,b2)
h13 = geth(b1,b3)
h14 = geth(b1,b4)
h15 = geth(b1,b5)
h23 = geth(b2,b3)
h24 = geth(b2,b4)
h25 = geth(b2,b5)
h34 = geth(b3,b4)
h35 = geth(b3,b5)
h45 = geth(b4,b5)



def getd(h):
    lmax = np.sum(np.max(h,axis=0))
    hmax = np.sum(np.max(h,axis=1))
    return (2*n - lmax - hmax)/(2*n)

d12 = getd(h12)
d13 = getd(h13)
d14 = getd(h14)
d15 = getd(h15)
d23 = getd(h23)
d24 = getd(h24)
d25 = getd(h25)
d34 = getd(h34)
d35 = getd(h35)
d45 = getd(h45)

d = (d12 + d13 + d14 + d15 + d23 + d24 + d25 + d34 + d35 + d45)/10
print(d)

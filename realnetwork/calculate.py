# -*- coding: utf-8 -*-
"""
Created on Wed May 13 09:13:35 2020

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
from sklearn.cluster import KMeans



node = 21688
k = 6
'''
alpha = 5
beta =  0.5
0.1 0.5 1 5 10 
lamda = 10
decay =   0.2 
0.1 0.2 0.3
order =  3
'''
lower_control = 10 ** -10
m = 3
oldass = -10 ** 10
olddensity = 10 ** 10


h = np.load("C:\\Users\\12728\\Desktop\\multiview\\math\\mvdgnmfhvalue.npy")

W_1 = sp.csr_matrix(np.load("C:\\Users\\12728\\Desktop\\data\\math\\y1.npy"))
W_2 = sp.csr_matrix(np.load("C:\\Users\\12728\\Desktop\\data\\math\\y2.npy"))
W_3 = sp.csr_matrix(np.load("C:\\Users\\12728\\Desktop\\data\\math\\y3.npy"))
''''W_4 = sp.csr_matrix(np.load("C:\\Users\\12728\\Desktop\\data\\amazon\\y4.npy"))'''
Ws = W_1+W_2+W_3


def caldensity(canshu):
    chengji = canshu @ canshu.T
    density1 = np.linalg.norm(W_1.toarray() - chengji ,2)
    density2 = np.linalg.norm(W_2.toarray() - chengji ,2)
    density3 = np.linalg.norm(W_3.toarray() - chengji ,2)
    ''''density4 = np.linalg.norm(W_4.toarray() - chengji ,2)'''
    return (density1 + density2 + density3) / m

number = 0
print("running")
''''for h in hvalue:''' 

ans = np.argmax(h,1)
x=np.zeros_like(h)
x[np.arange(len(x)),ans]=1
ind= nor.normalize(x,axis=0,norm="l2")

density = caldensity(ind)
ind = sp.csr_matrix(ind)
ass = ((ind.T @ Ws @ ind).toarray().trace())/m
    
print(number,density, ass)
    

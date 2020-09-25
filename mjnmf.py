# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 14:01:50 2020

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
from sklearn.cluster import KMeans

d = loadmat("C:\\Users\\12728\\Desktop\\read\\WebKB.mat")

''''truelabel  hangxiangliang '''



'''
node = (d["truelabel"][0][0].shape)[1]
true = ((d["truelabel"][0][0]).astype(np.int64)-1)[0]
'''

node = (d["truelabel"][0][0].shape)[1]
true = (d["truelabel"][0][0].ravel()-1).astype(np.int64)




''''truelabel liexiangliang '''
'''true = ((d["truelabel"][0][0]).astype(np.int64)-1)[:,0]
node = (d["truelabel"][0][0].shape)[0]
'''



k = (np.unique(true).shape)[0]

layer = (d["data"].shape)[1]


l = 
theta = 


alpha = 1
beta = 0.1
''' 0.01 0.05 0.1 0.5 1 5 10 50 100'''
lamda = 10
'''
decay =  0.15
 0.1 0.15 0.2 0.25 0.3
order =  3
'''


'''1 2 3 4 5'''
lower_control = 10 ** -10
iternum = 500

errorradio = 10 ** -6

''' ===============adcajency matrix============================='''


adj1 = d["data"][0,0].astype(np.float64)
adj2 = d["data"][0,1].astype(np.float64)
adj3 = d["data"][0,2].astype(np.float64)
''''adj4 = d["data"][0,3].astype(np.float64)'''



'''
adj1 = d["data"][0,0].toarray()
adj2 = d["data"][0,1].toarray()
adj3 = d["data"][0,2].toarray()
adj4 = d["data"][0,3].toarray()
'''


def caldistance(adj):
    '''numberoffeature = (adj.shape)[0]'''
    pingfanghe = np.sum(adj ** 2,0)
    jiaocha = adj.T @ adj
    distance = np.sqrt(pingfanghe.reshape(node,1) +pingfanghe.reshape(1,node)- 2 * jiaocha)
    return distance
   

def constructkernal(adjcency):
    dis = caldistance(adjcency)
    sig = np.median(dis)
    '''sig = 1'''
    fenzi = dis ** 2
    fenmu = max(2 *(sig ** 2), lower_control)
    wkernal = np.exp(-fenzi / fenmu)
    np.fill_diagonal(wkernal,0)
    return wkernal
    
W_1 = constructkernal(adj1)
W_2 = constructkernal(adj2)
W_3 = constructkernal(adj3)
'''W_4 = constructkernal(adj4)
'''





'''=================high-order ====================='''

'''
def high_order(W):
    P = np.zeros((node,node))
    for i in range(order):
        if i == 0:
            A = deepcopy(W)
        else:
            A = A @ W
        P += (decay ** i) * A
    return P
'''

def high_order(W):
    P = np.zeros((node,node))
    temp = 1
    for i in range(l):
        if i == 0:
            A = deepcopy(W)
        else:
            A = A @ W
        temp *= (i+1) 
        P += ((A * (theta ** i)) / temp)
    return P
    






w1 = high_order(W_1)
w2 = high_order(W_2)
w3 = high_order(W_3)
''''w4 = high_order(W_4)'''




'''
w1 = w1 / np.sum(w1)
w2 = w2 / np.sum(w2)
w3 = w3 / np.sum(w3)
'''


''''w4 = w4 / np.sum(w4)'''


'''============== similarity =============================='''

def similarity(W):
    cs = nor.normalize(W,axis=0,norm="l2")
    re = cs.T @cs
    return re

s1 = similarity(adj1)
np.fill_diagonal(s1,1)
s2 = similarity(adj2)
np.fill_diagonal(s2,1)
s3 = similarity(adj3)
np.fill_diagonal(s3,1)
'''s4 = similarity(adj4)
np.fill_diagonal(s4,1)'''




    
''' ================== normalize similarity ======================='''

def cal_sim(s):
    re = nor.normalize(s,axis=1,norm="l1")
    return re

u1 = cal_sim(s1)    
u2 = cal_sim(s2)    
u3 = cal_sim(s3)
''''u4 = cal_sim(s4)'''




w1 = cal_sim(w1)
w2 = cal_sim(w2)
w3 = cal_sim(w3)
''''w4 = cal_sim(w4)'''




    
''' ====================initial ====================='''
   
f1 = np.random.rand(node,k)
f2 = np.random.rand(node,k)
f3 = np.random.rand(node,k)
''''f4 = np.random.rand(node,k)'''



h = np.random.rand(node,k)
b = np.random.rand(node,k)


def b_update(B):
    fenzi = w1 @ f1 + w2 @ f2 + w3 @ f3 
    fenmu = B @ (f1.T @ f1 + f2.T @ f2+ f3.T @ f3   )
    '''B = B * np.sqrt(fenzi / np.maximum(fenmu, lower_control))'''
    B = B * (fenzi / np.maximum(fenmu, lower_control))
    return B

    
def f_update(w,u,f):
    fenzi = w.T @ b + beta * (u @ h)    
    fenmu = f @ (b.T @ b) + beta * f
    '''f = f * np.sqrt(fenzi / np.maximum(fenmu, lower_control))'''
    f = f * (fenzi / np.maximum(fenmu, lower_control))
    return f


def h_update(H):
    uuh = 2*beta*(u1.T@(u1 @ H)+u2.T@(u2 @ H) +u3.T@(u3 @ H)  )
    sh = 4 * alpha * (s1.T @ H + s2.T @ H + s3.T @ H  )
    uf = 2 * beta * (u1.T @ f1 + u2.T @ f2 + u3.T @ f3 ) 
    hhh = 8 * (layer * alpha + lamda) * (H @ (H.T @ H))
    
    fenzi = -uuh + np.sqrt((uuh * uuh) + (2*hhh)* (sh + uf + 4 * lamda * H))    
    H = H * np.sqrt(fenzi / np.maximum(hhh, lower_control))
    return H


'''def h_update(H):
    sh = (s1 + s2 + s3 +s4)@H
    utf = u1.T @ f1 + u2.T @ f2 + u3.T @ f3 + u4.T @ f4
    utuh = u1.T @(u1 @H)+ u2.T @(u2 @H) + u3.T @(u3 @H) + u4.T @(u4 @H) 
    hhth = H@(H.T @H)
    fenzi = 2*alpha *sh +beta* utf + 2*lamda * H
    fenmu = 2*(layer*alpha + lamda)* hhth + beta * utuh
    H = H * (fenzi / np.maximum(fenmu, lower_control))
    return H
'''

def u_update(u,f):
    fenzi = f @ (h.T)
    fenmu = u @ h @ (h.T)
    u = u * (fenzi / np.maximum(fenmu, lower_control))
    return u

def error(w,f,s,u):
    e1 =  np.sum((w - b @ f.T)**2) 
    e2 =  alpha * np.sum((s - h @ h.T) ** 2) 
    e3 =  beta * np.sum((f - u @ h)**2)
    '''e4 =  lamda * np.linalg.norm(h.T @ h - np.eye(k),2)'''
    err = e1 + e2 + e3 
    return err

''' ============================bestmap ====================='''

def calresult(cal):
    dic = {}
    G = np.zeros((k,k))
    for i in range(k):
        ind_cla1 = cal == i
        ind_cla1 = ind_cla1.astype(float)
        for j in range(k):
            ind_cla2 = true == j
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2*ind_cla1)

    m = Munkres()
    index = m.compute(-G)
    for i in range(k):
        dic[index[i][0]] = index[i][1]

    for i in range(node):
        cal[i] = dic[cal[i]]
    return cal    


e = [10 **7]    

acc = nmi = pre = recall = fscore = ari = - 10


answer = []
print(''''running''')
number = 0      
for i in range(iternum):
    
    number += 1
    
    b = b_update(b)
    
    f1 = f_update(w1,u1,f1) 
    f2 = f_update(w2,u2,f2)
    f3 = f_update(w3,u3,f3) 
    '''f4 = f_update(w4,u4,f4)'''
    

    
    u1 = u_update(u1,f1)
    u2 = u_update(u2,f2)
    u3 = u_update(u3,f3)
    ''''u4 = u_update(u4,f4)'''


    h = h_update(h)
    
    err1 = error(w1,f1,s1,u1)
    err2 = error(w2,f2,s2,u2)
    err3 = error(w3,f3,s3,u3)
    ''''err4 = error(w4,f4,s4,u4)'''
    e.append(err1 + err2)
    
    print(number,e[-1])
    if abs(e[-1] - e[-2])/e[-1] <= errorradio:
        break
    
ans = np.argmax(h,1)
predict = calresult(ans)

'''ACC'''
newacc = metrics.accuracy_score(true,predict)
    
'''NMI'''
newnmi = metrics.normalized_mutual_info_score(true, predict)

'''precision'''    
newpre = metrics.precision_score(true, predict,average="weighted")
    
'''recall'''
newrecall = metrics.recall_score(true, predict,average="weighted")

'''fscore'''    
newfscore = metrics.f1_score(true,predict,average="weighted")
    
'''ari'''    
newari = metrics.adjusted_rand_score(true,predict)

print(newacc,newnmi,newari,newfscore,newpre,newrecall)

   
 
''''print(e)'''
'''np.save("C:\\Users\\12728\\Desktop\\read\\err\\Hdigit.npy",e)'''
np.save("C:\\Users\\12728\\Desktop\\multiview\\image\\error\\WebKB\\mjnmf.npy",e)

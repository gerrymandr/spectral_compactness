# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 00:49:44 2018

@author: MGGG
"""
import numpy as np
import networkx as nx
import scipy.linalg
from scipy.sparse import csc_matrix
import scipy


def log_weighted_number_trees(G):
    m = nx.laplacian_matrix(G, weight = "weight")[1:,1:]
    m = csc_matrix(m)
    splumatrix = scipy.sparse.linalg.splu(m)
    diag_L = np.diag(splumatrix.L.A)
    diag_U = np.diag(splumatrix.U.A)
    S_log_L = [np.log(s) for s in diag_L]
    S_log_U = [np.log(s) for s in diag_U]
    LU_prod = np.sum(S_log_U) + np.sum(S_log_L)
    return LU_prod


m = 4
n = 4
d = 2
G = nx.grid_graph([m,n])
H = nx.grid_graph([m,n])
for x in G.edges():
    a = x[0]
    b = x[1]
    G.edges[x]["weight"] = (np.abs(a[0] - b[0]) + np.abs(a[1] - b[1]))
for x in H.edges():
    a = x[0]
    b = x[1]
    H.edges[x]["weight"] = (d*np.abs(a[0] - b[0]) + (1/d)* np.abs(a[1] - b[1]))

W_G = log_weighted_number_trees(G)
W_H = log_weighted_number_trees(H)
print("WG",W_G)
print("WH", W_H)
print("W_G - W_H", W_G - W_H)

#W_G 98.44804291763761
#W_H 209.95528522499345
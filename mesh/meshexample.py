# pip install scipy, pyglet, matplotlib, trimesh

import numpy as np
from scipy.sparse import *
import trimesh
from scipy.sparse.linalg import eigsh
from matplotlib.pyplot import plot
import matplotlib.cm as cm
from mayavi.mlab import *

def tri_indices(simplices):
    return ([triplet[c] for triplet in simplices] for c in range(3))
    
def row_norms(mtx):
    return np.sum(np.abs(mtx)**2,axis=-1)**.5

def barycentric_areas(X,T): # I'll go ahead and do part (d) here
    vv = []
    I = T[:,0]; J = T[:,1]; K = T[:,2];
    vv.append( X[I,:] ); vv.append( X[J,:] ); vv.append( X[K,:] )

    # Triangle areas
    nn = np.cross(vv[1]-vv[0],vv[2]-vv[0])
    triangleAreas = .5*row_norms(nn)

    # Angle deficits (integrated curvature) and barycentric areas
    barycentricAreas = np.zeros(nv)
    for i in range(0, 3):
        barycentricAreas = barycentricAreas + np.bincount(T[:,i],triangleAreas/3)
    
    return barycentricAreas

def mass_matrix(X,T): # lumped diagonal mass matrix
    nv = X.shape[0]
    return spdiags(barycentric_areas(X,T),0,nv,nv)

def cot_laplacian(X,T): # the famous cotangent Laplacian matrix; convention here is it's positive semidefinite
    nv = X.shape[0]
    nt = T.shape[0]
    
    vv = []
    I = T[:,0]; J = T[:,1]; K = T[:,2];
    vv.append( X[I,:] ); vv.append( X[J,:] ); vv.append( X[K,:] )

    # Triangle areas
    nn = np.cross(vv[1]-vv[0],vv[2]-vv[0])
    triangleAreas = .5*row_norms(nn)

    # Angle deficits (integrated curvature) and barycentric areas
    innerCotangents = np.zeros((nt,3))
    for i in range(0, 3):
        e1 = vv[(i+1)%3]-vv[i]
        e2 = vv[(i+2)%3]-vv[i]
        innerCotangents[:,i] = np.sum(np.multiply(e1,e2),axis=-1)/(2*triangleAreas) # dot product over cross product
    
    L = (csr_matrix((innerCotangents[:,2],(I,J)),shape=(nv,nv)) +
         csr_matrix((innerCotangents[:,0],(J,K)),shape=(nv,nv)) +
         csr_matrix((innerCotangents[:,1],(K,I)),shape=(nv,nv))
        )
    
    L = L+L.transpose()
    rowSums = np.sum(L,axis=-1).transpose()
    L = L - spdiags(rowSums,0,nv,nv)
    return -.5*L

def laplacian_spectrum(X,T,k,boundary='neumann'):
    L = cot_laplacian(X,T)
    M = mass_matrix(X,T)

    if boundary=='neumann': # natural boundary conditions, ignore boundary
        vals,vecs = eigsh(L,k=k,M=M,which='SM')
    if boundary=='dirichlet': # zero out the boundary
        boundary_verts,interior_verts = boundary_vertices(T)
        L0 = L[interior_verts,:][:,interior_verts] # is there a faster way to get a submatrix?
        M0 = M.tocsr()[interior_verts,:][:,interior_verts]

        vals,vecs0 = eigsh(L0,k=k,M=M0,which='SM')
        vecs = zeros((X.shape[0],k))
        vecs[interior_verts,:] = vecs0
    
    return vals,vecs

def boundary_vertices(T): # some magic to find a list of boundary vertices, probably slow
    E1 = column_stack((T[:,0],T[:,1]))
    E2 = column_stack((T[:,1],T[:,2]))
    E3 = column_stack((T[:,2],T[:,0]))
    E = row_stack((E1,E2,E3))
    E.sort(axis=1)
    ne = E.shape[0]
    o = ones(ne)
    
    adj = csr_matrix((o,(E[:,0],E[:,1])))
    
    bdrypart = (adj==1) #edges repeated 2x are in the interior
    idx=find(bdrypart)
    idx1=idx[0]
    idx2=idx[1]
    boundary_verts = unique(row_stack((idx1,idx2)))

    all_verts = range(0,T.max()+1)
    interior_verts = list(set(all_verts)-set(boundary_verts))
        
    return boundary_verts,interior_verts


mesh = trimesh.load('moomoo_chopped.off')
X = mesh.vertices; # each row is the position of a vertex
I,J,K=tri_indices(mesh.faces)
T = np.column_stack((I,J,K)) # rows are (i,j,k) indices of triangle vertices
nv = X.shape[0] # number of vertices
nt = T.shape[0] #number of triangles

k = 100


vals,vecs = laplacian_spectrum(X,T,k,'dirichlet')
triangular_mesh(X[:,0],X[:,1],X[:,2], T, scalars=vecs[:,0])
colorbar()
title('Dirichlet base eigenfunction')


vals,vecs = laplacian_spectrum(X,T,k,'neumann')
figure()
triangular_mesh(X[:,0],X[:,1],X[:,2], T, scalars=vecs[:,1]) # notice the 1!
colorbar()
title('Neumann base eigenfunction')
# pip install scipy, pyglet, matplotlib, trimesh

import numpy as np
from math import *
from scipy.sparse import *
import trimesh
from scipy.sparse.linalg import eigsh, inv
from scipy.special import jn_zeros
from matplotlib.pyplot import plot
import matplotlib.cm as cm
#import mayavi.mlab as mlab
from matplotlib import pyplot as plt


def tri_indices(simplices):
    return ([triplet[c] for triplet in simplices] for c in range(3))
    
def row_norms(mtx):
    return np.sum(np.abs(mtx)**2,axis=-1)**.5

def barycentric_areas(X,T): 
    # I'll go ahead and do part (d) here
    vv = []
    nv = X.shape[0]
    I = T[:,0]; J = T[:,1]; K = T[:,2];
    vv.append( X[I,:] ); vv.append( X[J,:] ); vv.append( X[K,:] )

    # Triangle areas
    nn = np.cross(vv[1]-vv[0],vv[2]-vv[0])
    triangleAreas = .5*row_norms(nn)

    # Angle deficits (integrated curvature) and barycentric areas
    barycentricAreas = np.zeros(nv)
    for i in range(0, 3):
        barycentricAreas = barycentricAreas + np.bincount(T[:,i],triangleAreas/3,nv) #added for flat states
    
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
    area = np.sum(M)
    M = M / area * pi

    if boundary=='neumann': # natural boundary conditions, ignore boundary
        vals,vecs = eigsh(csc_matrix(L),k=k,M=csc_matrix(M),sigma=-1)#which='SM') #needed to converge on flat states
    if boundary=='dirichlet': # zero out the boundary
        boundary_verts,interior_verts = boundary_vertices(T)
        L0 = L[interior_verts,:][:,interior_verts] # is there a faster way to get a submatrix?
        M0 = M.tocsr()[interior_verts,:][:,interior_verts]

        vals,vecs0 = eigsh(csc_matrix(L0),k=k,M=csc_matrix(M0),sigma=-1)#which='SM')#needed to converge on flat states
        vecs = np.zeros((X.shape[0],k))
        vecs[interior_verts,:] = vecs0
    
    return vals,vecs

def boundary_vertices(T): # some magic to find a list of boundary vertices, probably slow
    E1 = np.column_stack((T[:,0],T[:,1]))
    E2 = np.column_stack((T[:,1],T[:,2]))
    E3 = np.column_stack((T[:,2],T[:,0]))
    E = np.row_stack((E1,E2,E3))
    E.sort(axis=1)
    ne = E.shape[0]
    o = np.ones(ne)
    
    adj = csr_matrix((o,(E[:,0],E[:,1])))
    
    bdrypart = (adj==1) #edges repeated 2x are in the interior
    idx=find(bdrypart)
    idx1=idx[0]
    idx2=idx[1]
    boundary_verts = np.unique(np.row_stack((idx1,idx2)))

    all_verts = range(0,T.max()+1)
    interior_verts = list(set(all_verts)-set(boundary_verts))
        
    return boundary_verts,interior_verts

def heat_kernel(x, evals):
    return [sum([np.exp(-t*l) for l in evals]) for t in x]

def heat_kernel_taylor(x, L):
    L2 = L*L
    L3 = L2*L
    L4 = L3*L
    L5 = L4*L
    estimate = [(identity(L.shape[0], format='csr') - t*L + t**2*L2/2 - \
                    (t**3)*L3/6 + (t**4)*L4/24 - (t**5)*L5/120) for t in x]
    return [np.trace(i.A) for i in estimate]

def small_heat_kernel(x, p, a):
    return [a/(2*pi*t) - p/(4*sqrt(pi*t)) for t in x]

def find_area(X,T):
    return np.sum(mass_matrix(X,T))

def find_perimeter(X,T):
    E1 = np.column_stack((T[:,0],T[:,1]))
    E2 = np.column_stack((T[:,1],T[:,2]))
    E3 = np.column_stack((T[:,2],T[:,0]))
    E = np.row_stack((E1,E2,E3))
    E.sort(axis=1)
    ne = E.shape[0]
    o = np.ones(ne)
    
    adj = csr_matrix((o,(E[:,0],E[:,1])))
    
    bdrypart = (adj==1) #edges repeated 2x are in the interior
    first_vertex, second_vertex, potatoe = find(bdrypart)
    perimeter = 0
    for i in range(len(first_vertex)):
        perimeter += sqrt((X[first_vertex[i]][0] - X[second_vertex[i]][0])**2 \
                          + (X[first_vertex[i]][1] - X[second_vertex[i]][1])**2)
    return perimeter
    

def big_t(files, circle, x, k):
    '''
    This function calculates the values for our estimation of the heat kernel for large t. Takes .off files,
    x (list of t values) and k(number of eigenvalues to find). 
    '''
    #loading circle stuffs
    mesh = trimesh.load(circle)
    X = mesh.vertices; # each row is the position of a vertex
    I,J,K=tri_indices(mesh.faces)
    T = np.column_stack((I,J,K)) # rows are (i,j,k) indices of triangle vertices
    nv = X.shape[0] # number of vertices
    nt = T.shape[0] #number of triangles
    circle_vals,circle_vecs = laplacian_spectrum(X,T,k,'dirichlet')
    
    big_t_values = []
    big_t_values.append(heat_kernel(x,circle_vals))
    
    #for historical value
    y = [[0]*100, [0]*100]
    
    # Calculating values for all other files
    for file in files:
        mesh = trimesh.load(file)
        X = mesh.vertices; # each row is the position of a vertex
        I,J,K=tri_indices(mesh.faces)
        T = np.column_stack((I,J,K)) # rows are (i,j,k) indices of triangle vertices
        nv = X.shape[0] # number of vertices
        nt = T.shape[0] #number of triangles
        vals,vecs = laplacian_spectrum(X,T,k,'dirichlet')
        big_t_values.append(heat_kernel(x, vals))
        
    return big_t_values
    
def small_t(files, circle, x):
    ''' This calculates the values for our estimation for small values of t, where we use that 
    Z(t) = (A/2*pi*t)-(P/4)*(1/sqrt(2*pi*t)). Takes .off files and x (list of t values).'''
    
    mesh = trimesh.load(circle)
    X = mesh.vertices; # each row is the position of a vertex
    I,J,K=tri_indices(mesh.faces)
    T = np.column_stack((I,J,K)) # rows are (i,j,k) indices of triangle vertices
    perimeter = find_perimeter(X,T)/sqrt(find_area(X,T)/pi)
    print('circle', perimeter)
    
    small_t_values = []
    small_t_values.append(small_heat_kernel(x,perimeter,pi))
    for file in files:
        mesh = trimesh.load(file)
        X = mesh.vertices;
        I,J,K = tri_indices(mesh.faces)
        T = np.column_stack((I,J,K))
        
        perimeter = find_perimeter(X,T)/sqrt(find_area(X,T)/pi)
        small_t_values.append(small_heat_kernel(x,perimeter,pi))
        print(file, perimeter)
        
    return small_t_values
        
    
def interpolate_t(files, circle, k):
    ''' This function takes .off files and k (number of eigenvalues). It deals with reweighting the two 
    estimates and plotting the results.'''
    
    x = np.linspace(0,.5, 200).tolist()
    x = x[1:]
    #weights = np.linspace(0,1,200).tolist() + [1]*200
    weights = [np.exp(-50*t) for t in x]
    #weights = [0]*200
    #x = np.linspace(0.001,0.05, 200)
    #x = x[1:]
    
    big_t_values = big_t(files, circle, x, k)
    small_t_values = small_t(files, circle, x)
    
    # interpolate and plot
    interp = [[(weights[j])*max(small_t_values[i][j],0) + (1-weights[j])*big_t_values[i][j] for j in range(len(x))] for i in range(len(files)+1)]
    log_interp = [[np.log(j) for j in i] for i in interp]
    
    names = [f[:-4] for f in files]
    names.insert(0, 'circle')
    
    new_interp = [[(interp[0][j] - interp[i][j]) for j in range(len(x))] for i in range(len(files)+1)]
    
    plt.xscale('log')
    plt.yscale('log')
    for i in range(len(files)+1):
        plt.plot(x, new_interp[i])
    plt.legend(names)
    plt.show()


interpolate_t(['snail.off', 'star.off'], 'circle.off', 50)
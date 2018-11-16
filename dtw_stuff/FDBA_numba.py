from fastdtw import fastdtw
import numpy as np
from tslearn.preprocessing import TimeSeriesResampler
from sklearn.cluster.k_means_ import _k_init
from sklearn.utils import check_random_state
from numba import jit,float64,int64,none
from numba.types import List,UniTuple,Tuple,Array

@jit(Tuple((float64, List(Tuple((int64,int64)),reflected=True)))
(Array(int64, 2, 'C'), Array(int64, 2, 'C'), int64, none))
def fastdtwj(x,y,radius=1,dist=None):
    return fastdtw(x,y,radius,dist)

@jit#(nopython=True)
def medoid(D):#Find mediod of database
    #compute distance matrix
    n=D.shape[0]
    dist_matrix=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            dist_matrix[i,j]=fastdtwj(D[i],D[j])[0]
    dist_matrix=np.maximum(dist_matrix,dist_matrix.T)

    #check for mediod
    med=D[np.argmin(np.sum(dist_matrix,axis=1))]
    return med

@jit#(nopython=True)
def s_contrib(center,s):#Contribution of sequence s to barycenter center

    alignment=np.array(fastdtwj(center,s)[1])
    contrib=np.zeros((center.shape[0],2))

    for i in range(center.shape[0]):
        i_contrib=alignment[alignment[:,0]==i][:,1]
        contrib[i,0]=np.sum(s[i_contrib])
        contrib[i,1]=i_contrib.shape[0]

    return contrib

@jit(nopython=True)
def FDBA_update(center_matrix):#Update barycenter according to the center_matrix
    return np.divide(center_matrix[:,0],center_matrix[:,1])

@jit#(nopython=True)
def FDBA(D,n_iterations=10,verbose=False):#Fast DTW barycenter averaging full algorithm
    center=medoid(D)#initializing center as mediod

    for i in range(n_iterations):
        if verbose:
            print ("FDBA iteration:")
            print (i+1)
        new_center=np.zeros((center.shape[0],2))
        for j in range(D.shape[0]):
            new_center=np.add(new_center,s_contrib(center,D[j]))
        center=FDBA_update(new_center)

    return center

@jit#(nopython=True)
def assign(centers,D):#Assign all the time series's to the closest center in centers
    dists=np.zeros((D.shape[0],centers.shape[0]))

    for i in range(D.shape[0]):
        for j in range(centers.shape[0]):
            dists[i,j]=fastdtwj(D[i],centers[j])[0]
    matched_labels=dists.argmin(axis=1)

    return matched_labels

#@jit#(nopython=True)
def FDBA_clustering(D,n_iterations=10,k=2,verbose=False):
    sz=np.max(np.array(list(map(lambda s: len(s),D))))
    D=TimeSeriesResampler(sz=sz).fit_transform(D)
    D=np.squeeze(D)
    D_squared_norms=(D * D).sum(axis=1)
    centers=_k_init(D,k,D_squared_norms,check_random_state(None))

    for i in range(n_iterations):
        if verbose:
            print ("Clustering iteration: ")
            print (i+1)
        matched_labels=assign(centers,D)
        for j in range(centers.shape[0]):
            centers[j]=FDBA(D[matched_labels==j],verbose=verbose)

    return centers

# cython: infer_types=True

from fastdtw import fastdtw
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.preprocessing import TimeSeriesResampler
from sklearn.cluster.k_means_ import _k_init
from sklearn.utils import check_random_state
import numpy as np
import time
cimport numpy as np
cimport cython

DTYPE_FLOAT=np.float
DTYPE_INT=np.int

ctypedef np.float_t DTYPE_FLOAT_t
ctypedef np.int_t DTYPE_INT_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def medoid(np.ndarray[DTYPE_FLOAT_t, ndim=2] D):#Find mediod of database
    t1=time.clock()
    #compute distance matrix
    n=D.shape[0]
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] dist_matrix=np.zeros((n,n),dtype=DTYPE_FLOAT)

    for i in range(n):
        for j in range(i+1):
            dist_matrix[i,j]=fastdtw(D[i],D[j])[0]
    dist_matrix=np.maximum(dist_matrix,dist_matrix.T,dtype=DTYPE_FLOAT)

    #check for mediod
    med=D[np.argmin(np.sum(dist_matrix,axis=1))]
    print ("mediod: "+str(t1-time.clock()))
    return med

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def s_contrib(np.ndarray[DTYPE_FLOAT_t, ndim=1] center,np.ndarray[DTYPE_FLOAT_t, ndim=1] s):#Contribution of sequence s to barycenter center
    t1=time.clock()
    cdef np.ndarray[DTYPE_INT_t, ndim=2] alignment=np.array(fastdtw(center,s)[1],dtype=DTYPE_INT)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] contrib=np.zeros((len(center),2),dtype=DTYPE_FLOAT)
    cdef np.ndarray[DTYPE_INT_t, ndim=1] i_contrib

    for i in range(len(center)):
        i_contrib=alignment[alignment[:,0]==i][:,1]
        contrib[i,0]=np.sum(s[i_contrib])
        contrib[i,1]=i_contrib.shape[0]
    print ("s_contrib: "+str(t1-time.clock()))
    return contrib

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def FDBA_update(np.ndarray[DTYPE_FLOAT_t, ndim=2] center_matrix):#Update barycenter according to the center_matrix
    return np.divide(center_matrix[:,0],center_matrix[:,1],dtype=DTYPE_FLOAT)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def FDBA(np.ndarray[DTYPE_FLOAT_t, ndim=2] D,int n_iterations=10,bint verbose=False):#Fast DTW barycenter averaging full algorithm
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] center=medoid(D)#initializing center as mediod
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] new_center
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] s

    for i in range(n_iterations):#actual algorithm
        if verbose:
            print ("FDBA iteration: "+str(i+1))
        new_center=np.zeros((len(center),2),dtype=DTYPE_FLOAT)#creating new center

        for s in D:#calculating contribution of series
            new_center=np.add(new_center,s_contrib(center,s),dtype=DTYPE_FLOAT)
        center=FDBA_update(new_center)#updating center

    return center

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def assign(np.ndarray[DTYPE_FLOAT_t, ndim=2] centers,np.ndarray[DTYPE_FLOAT_t, ndim=2] D):#assign time series to centers
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] dists=np.zeros((D.shape[0],centers.shape[0]),dtype=DTYPE_FLOAT)

    for i in range(D.shape[0]):#the loop
        for j in range(centers.shape[0]):
            dists[i,j]=fastdtw(D[i],centers[j])[0]

    cdef np.ndarray[DTYPE_INT_t, ndim=1] matched_labels=dists.argmin(axis=1)#writing labels
    return matched_labels

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def FDBA_clustering(np.ndarray[DTYPE_FLOAT_t, ndim=2] D,int n_iterations=10,int k=2,bint verbose=False):#kmeans using FDBA
    sz=np.max(np.array(list(map(lambda s: len(s),D))))
    D=np.squeeze(TimeSeriesResampler(sz=sz).fit_transform(D))#transforming D for ease of usage

    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] D_squared_norms=(D * D).sum(axis=1)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] centers=_k_init(D,k,D_squared_norms,check_random_state(None))
    cdef np.ndarray[DTYPE_INT_t, ndim=1] matched_labels

    for i in range(n_iterations):
        if verbose:
            print ("Clustering iteration: "+str(i+1))
        matched_labels=assign(centers,D)
        for j in range(len(centers)):
            centers[j]=FDBA(D[matched_labels==j],verbose=verbose)

    return centers

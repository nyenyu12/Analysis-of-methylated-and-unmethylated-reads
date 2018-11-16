from numba import jit,autojit
import numba
import numpy as np
from fastdtw_numba import fastdtw
import fastdtw
from test_data_sine import series
import FDBA
import time
from my_timeit import my_timeit

@jit
def fastdtw_n_c(x,y,radius=1,dist=None):
    a=fastdtw.fastdtw(x,y)
    print (numba.typeof(x),numba.typeof(y),numba.typeof(radius),numba.typeof(dist))

@jit(nopython=True)
def foo(D):
    n=D.shape[0]
    dist_matrix=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            tmp=fastdtw(D[i],D[j])
            dist_matrix[i,j]=fastdtw(D[i],D[j])[0]
    return np.sum(D)

@jit(nopython=True)
def min_ind(a):
    return a[0]

@jit(nopython=True)
def foob(a,b,c):
    c=np.stack((a,b,c))
    print (c)
    return c[np.argmin(c[:,0])]

#@jit(nopython=True)
def medoid(D):#Find mediod of database
    #compute distance matrix
    n=D.shape[0]
    dist_matrix=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            t1=time.perf_counter()
            dist_matrix[i,j]=fastdtw(D[i],D[j])[0]
            print (time.perf_counter()-t1)
    dist_matrix=np.maximum(dist_matrix,dist_matrix.T)

    #check for mediod
    med=D[np.argmin(np.sum(dist_matrix,axis=1))]
    return med

#if __name__=='_-__main__':
a=np.array([[1,2,3],[0,0,4],[3,4,5]])
#medoid(a)
#medoid.inspect_types()
#foo(a)
#foo.inspect_types()
window=[(1,2),(2,2),(3,3)]

def main():
    #print (medoid(series))
    #fastdtw_n_c(a,a)
    np.random.random((10000,10000))

#foob.inspect_types()
if __name__=="__main__":
    print (my_timeit(main,1))

from fastdtw import fastdtw
import numpy as np
import matplotlib.pyplot as plt
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.preprocessing import TimeSeriesResampler
from sklearn.cluster.k_means_ import _k_init
from sklearn.utils import check_random_state
import timeit
import time

def medoid(D):#Find medoid of database
    #compute distance matrix
    n=D.shape[0]
    dist_matrix=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            #t1=time.perf_counter()
            dist_matrix[i,j]=fastdtw(D[i],D[j])[0]
            #print (time.perf_counter()-t1)
    dist_matrix=np.maximum(dist_matrix,dist_matrix.T)

    #check for mediod
    med=D[np.argmin(np.sum(dist_matrix,axis=1))]
    return med

def s_contrib(center,s):#Contribution of sequence s to barycenter center

    alignment=np.array(fastdtw(center,s)[1])
    contrib=np.zeros((center.shape[0],2))

    for i in range(center.shape[0]):
        i_contrib=alignment[alignment[:,0]==i][:,1]
        contrib[i,0]=np.sum(s[i_contrib])
        contrib[i,1]=i_contrib.shape[0]

    return contrib

def FDBA_update(center_matrix):#Update barycenter according to the center_matrix
    return np.divide(center_matrix[:,0],center_matrix[:,1])

def FDBA(D,n_iterations=10,verbose=False):#Fast DTW barycenter averaging full algorithm
    center=medoid(D)#initializing center as mediod

    for i in range(n_iterations):
        if verbose:
            print ("FDBA iteration: "+str(i+1))
        new_center=np.zeros((center.shape[0],2))
        for s in D:
            new_center=np.add(new_center,s_contrib(center,s))
        center=FDBA_update(new_center)

    return center

def assign(centers,D):
    dists=np.zeros((D.shape[0],centers.shape[0]))

    for i in range(D.shape[0]):
        for j in range(centers.shape[0]):
            dists[i,j]=fastdtw(D[i],centers[j])[0]
    matched_labels=dists.argmin(axis=1)

    return matched_labels

def FDBA_clustering(D,n_iterations=10,k=2,verbose=False):
    sz=np.max(np.array(list(map(lambda s: len(s),D))))
    D=TimeSeriesResampler(sz=sz).fit_transform(D)
    D=np.squeeze(D)
    D_squared_norms=(D * D).sum(axis=1)
    centers=_k_init(D,k,D_squared_norms,check_random_state(None))

    for i in range(n_iterations):
        if verbose:
            print ("Clustering iteration: "+str(i+1))
        matched_labels=assign(centers,D)
        for j in range(len(centers)):
            centers[j]=FDBA(D[matched_labels==j],verbose=verbose)

    return centers

def main():
    #generating synthetic data
    n_series = 20
    length = 2000

    series = list()
    padding_length=30
    indices = range(0, length-padding_length)
    main_profile_gen = np.array(list(map(lambda j: np.sin(2*np.pi*j/len(indices)),indices)))
    for i in range(0,n_series):
        n_pad_left = np.random.randint(0,padding_length)
        #adding zero at the start or at the end to shif the profile
        series_i = np.pad(main_profile_gen,(n_pad_left,padding_length-n_pad_left),mode='constant',constant_values=0)
        #randomize a bit
        series_i = list(map(lambda j:np.random.normal(0,0.02)+j,series_i))
        assert(len(series_i)==length)
        series.append(series_i)
    series = np.array(series)

    #plotting the synthetic data
    for s in series:
        plt.plot(range(0,length), s,color='gray')

    #calculating average series with DBA
    t1=time.perf_counter()
    fastdtw(series[0],series[1])
    print (time.perf_counter()-t1)
    average_series=FDBA(series)
'''
    #plotting the average series
    plt.plot(range(0,length), average_series,color='r')
    plt.show()'''

if __name__== "__main__":
    #timeit.timeit("FDBA",number=3)
    main()

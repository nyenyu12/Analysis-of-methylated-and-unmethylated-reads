import matplotlib
matplotlib.use('TkAgg')
import numpy
import matplotlib.pyplot as plt
import timeit
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler

poly1=numpy.polynomial.polynomial.Polynomial((3,4,1,-1,0.5,1,-1/3,1))
poly2=numpy.polynomial.polynomial.Polynomial((-1,-3,2,0.1,0.7,-2))
X_train=[]

for i in range(0,40):
    X_train.append(poly1.linspace(3000,(i,i+10))[1])
    X_train.append(poly2.linspace(3000,(i,i+20))[1])

X_train=numpy.array(X_train)
print (X_train.shape)
seed = 0
numpy.random.seed(seed)

'''X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
X_train = X_train[y_train < 4]  # Keep first 3 classes'''
numpy.random.shuffle(X_train)
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:50])  # Keep only 50 time series
X_train = TimeSeriesResampler(sz=40).fit_transform(X_train)  # Make time series shorter
sz = X_train.shape[1]


def main():
    # DBA-k-means
    #print("DBA k-means")
    dba_km = TimeSeriesKMeans(n_clusters=2, n_init=2, metric="dtw", verbose=False, max_iter_barycenter=10, random_state=seed)
    y_pred = dba_km.fit_predict(X_train)

    for yi in range(2):
        #plt.subplot(3, 3, 4 + yi)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
        plt.xlim(0, sz)
        plt.ylim(-4, 4)
        if yi == 1:
            plt.title("DBA $k$-means")

    #plt.tight_layout()
    plt.show()

if __name__== "__main__":
    #timeit.timeit("main()",number=3)
    main()

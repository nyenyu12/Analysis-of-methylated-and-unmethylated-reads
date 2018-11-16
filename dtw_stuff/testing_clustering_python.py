import FDBA
import numpy
import matplotlib.pyplot as plt
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
import timeit
'''
poly1=numpy.polynomial.polynomial.Polynomial((3,4,1,-1,0.5,1,-1/3,1))
poly2=numpy.polynomial.polynomial.Polynomial((-1,-3,2,0.1,0.7,-2))
X_train=[]

for i in range(0,40):
    X_train.append(poly1.linspace(3000,(i,i+20))[1])
    X_train.append(poly2.linspace(3000,(i,i+20))[1])

X_train=numpy.array(X_train)
print (X_train.shape)
seed = 0
numpy.random.seed(seed)
'''
seed = 0
numpy.random.seed(seed)
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
X_train = X_train[y_train < 4]  # Keep first 3 classes
numpy.random.shuffle(X_train)
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:50])  # Keep only 50 time series
X_train = TimeSeriesResampler(sz=40).fit_transform(X_train)  # Make time series shorter
X_train=numpy.squeeze(X_train)

sz = X_train.shape[1]

def main():
    centers = FDBA.FDBA_clustering(X_train,n_iterations=6,k=2,verbose=False)
    y_pred=FDBA.assign(centers,X_train)
    print (y_pred)
    '''for yi in range(2):
        #plt.subplot(3, 3, 4 + yi)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(centers[yi].ravel(), "r-")
        plt.xlim(0, sz)
        plt.ylim(-4, 4)
        if yi == 1:
            plt.title("FDBA $k$-means")

    #plt.tight_layout()
    plt.show()'''

if __name__== "__main__":
    #timeit.timeit("main()",number=3)
    main()

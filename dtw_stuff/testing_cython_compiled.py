import matplotlib
matplotlib.use('Agg')
import FDBA_cython
import numpy as np
import matplotlib.pyplot as plt

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
average_series=FDBA_cython.FDBA(series)

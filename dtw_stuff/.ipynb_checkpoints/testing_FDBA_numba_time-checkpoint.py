from test_data_sine import series
import FDBA_numba
import numpy as np
import matplotlib.pyplot as plt
from my_timeit import my_timeit
import timeit

def main():
    #calculating average series with DBA
    average_series=FDBA_numba.FDBA(series)

    #print (my_timeit(FDBA.FDBA_clustering,1,series))
    #currently around 3.07 sec per loop for FDBA
    #currently around 29.45 sec per loop for FDBA_clustering

if __name__=="__main__":
    #main()
    #timeit.timeit("main()",number=3)
    print ('hi')
    print (my_timeit(FDBA_numba.FDBA,10,series))

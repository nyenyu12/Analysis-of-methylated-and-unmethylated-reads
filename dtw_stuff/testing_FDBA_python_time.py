from test_data_sine import series
import FDBA
import numpy as np
import matplotlib.pyplot as plt
from my_timeit import my_timeit
import timeit

def main():
    #calculating average series with DBA
    average_series=FDBA.FDBA(series)

    #print (my_timeit(FDBA.FDBA_clustering,1,series))
    #currently 8.98 sec per loop

if __name__=="__main__":
    main()
    #timeit.timeit("main()",number=3)
    print (my_timeit(FDBA.FDBA_clustering,10,series))

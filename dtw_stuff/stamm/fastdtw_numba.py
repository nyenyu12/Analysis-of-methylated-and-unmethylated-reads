from __future__ import absolute_import, division
import numbers
import numpy as np
from numba import jit
import timeit
from test_arrays import X_train,expand_window_input

try:
    range = xrange
except NameError:
    pass

@jit(nopython=True)
def __difference_numba(a, b):
    return abs(a-b)

@jit(nopython=True)
def __reduce_by_half_numba(x):
    return [(x[i] + x[1+i]) / 2 for i in range(0, len(x) - len(x) % 2, 2)]

@jit(nopython=True)
def __expand_window_numba(path, len_x, len_y, radius):
    path_ = set(path)
    for i, j in path:
        for a, b in [(i + a, j + b) for a in range(-radius, radius+1) for b in range(-radius, radius+1)]:
            path_.add((a, b))
            #pass

    window_ = set()
    for i, j in path_:
        for a, b in ((i * 2, j * 2), (i * 2, j * 2 + 1),
                     (i * 2 + 1, j * 2), (i * 2 + 1, j * 2 + 1)):
            window_.add((a, b))
            #pass

    window = []
    start_j = 0
    for i in range(0, len_x):
        new_start_j = None
        for j in range(start_j, len_y):
            if (i, j) in window_:
                window.append((i, j))
                if new_start_j is None:
                    new_start_j = j
            elif new_start_j is not None:
                break
        start_j = new_start_j
    #pass
    return window

@jit(nopython=True)
def __dtw_numba(x, y, window):
    len_x, len_y = len(x), len(y)
    if window is None:
        window = [(i, j) for i in range(len_x) for j in range(len_y)]
    window = [(i + 1, j + 1) for i, j in window]

    #my code
    D=np.full((window[-1][0]+1,window[-1][1]+1,3),np.inf)#,dtype=np.float_)
    D[0, 0] = np.array([0, 0, 0])

    for i,j in window:
        dt = __difference_numba(x[i-1], y[j-1])
        priors=np.array([(D[i-1, j][0], i-1, j), (D[i, j-1][0], i, j-1),
                      (D[i-1, j-1][0], i-1, j-1)])
        D[i,j] = priors[np.argmin(priors[:,0])]
        D[i,j][0]+=dt

    path = []
    i, j = len_x, len_y
    while not (i == j == 0):
        path.append((i-1, j-1))
        i, j = int(D[i, j][1]), int(D[i, j][2])
    path.reverse()
    return (D[len_x, len_y][0], path)

@jit(nopython=True)
def __fastdtw_numba(x, y, radius):
    min_time_size = radius + 2

    if len(x) < min_time_size or len(y) < min_time_size:
        return dtw(x, y)

    x_shrinked = __reduce_by_half_numba(x)
    y_shrinked = __reduce_by_half_numba(y)
    distance, path = \
        __fastdtw_numba(x_shrinked, y_shrinked, radius=radius)
    window = __expand_window_numba(path, len(x), len(y), radius)
    return __dtw_numba(x, y, window)

@jit(nopython=True)
def dtw(x, y):
    ''' return the distance between 2 time series without approximation
        Parameters
        ----------
        x : array_like
            input array 1
        y : array_like
            input array 2
        dist : function or int
            The method for calculating the distance between x[i] and y[j]. If
            dist is an int of value p > 0, then the p-norm will be used. If
            dist is a function then dist(x[i], y[j]) will be used. If dist is
            None then abs(x[i] - y[j]) will be used.
        Returns
        -------
        distance : float
            the approximate distance between the 2 time series
        path : list
            list of indexes for the inputs x and y
        Examples
        --------
        >>> import numpy as np
        >>> import fastdtw
        >>> x = np.array([1, 2, 3, 4, 5], dtype='float')
        >>> y = np.array([2, 3, 4], dtype='float')
        >>> fastdtw.dtw(x, y)
        (2.0, [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2)])
    '''
    return __dtw_numba(x, y, None)

@jit(nopython=True)
def fastdtw(x, y, radius=1):
    ''' return the approximate distance between 2 time series with O(N)
        time and memory complexity
        Parameters
        ----------
        x : array_like
            input array 1
        y : array_like
            input array 2
        radius : int
            size of neighborhood when expanding the path. A higher value will
            increase the accuracy of the calculation but also increase time
            and memory consumption. A radius equal to the size of x and y will
            yield an exact dynamic time warping calculation.
        dist : function or int
            The method for calculating the distance between x[i] and y[j]. If
            dist is an int of value p > 0, then the p-norm will be used. If
            dist is a function then dist(x[i], y[j]) will be used. If dist is
            None then abs(x[i] - y[j]) will be used.
            ##### Currently not functional, always will use abs.
        Returns
        -------
        distance : float
            the approximate distance between the 2 time series
        path : list
            list of indexes for the inputs x and y
        Examples
        --------
        >>> import numpy as np
        >>> import fastdtw
        >>> x = np.array([1, 2, 3, 4, 5], dtype='float')
        >>> y = np.array([2, 3, 4], dtype='float')
        >>> fastdtw.fastdtw(x, y)
        (2.0, [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2)])
    '''
    return __fastdtw_numba(x, y, radius)

a=np.array([1.0,2.0,3.0],dtype=np.float)
b=np.array([2.0,3.0,4.0],dtype=np.float)
d=2.0
c=4.7
e=2
f=[1.0,2.0,3.0]
g=[1.0,2.0,3.0]

def main():
    '''a=np.array([1.0,2.0,3.0],dtype=np.float)
    b=np.array([2.0,3.0,4.0],dtype=np.float)
    d=2.0
    c=4.7
    e=2'''
    fastdtw(X_train[0],X_train[-1])
    #__expand_window(expand_window_input[0],expand_window_input[1],expand_window_input[2],expand_window_input[3])
    #cProfile.run('fastdtw(f,g)')
    #__dtw(a,b,None,__difference)
    #fastdtw.inspect_types()


if __name__=="__main__":
    main()
    #print (timeit.timeit("main()","gc.enable; from __main__ import main",number=100))

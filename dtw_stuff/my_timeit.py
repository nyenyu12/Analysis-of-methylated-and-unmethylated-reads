from time import clock

def my_timeit(f,n,*args,**kwargs):
    '''
    My timing function for when timeit is fucking stupid.
    f - Function to time.
    n - Number of runs to avergage the time from.
    *args and **kwargs - The input parameters in the order they should be inputted to f.
    '''
    avg_time=0
    for i in range(n):
        t0=clock()
        f(*args,**kwargs)
        avg_time+=clock()-t0
    return avg_time/n

def foo(a,b,c=0):
    print ((a+b)*c)
    pass

def main():
    my_timeit(foo,10,1,2)

if __name__=="__main__":
    main()

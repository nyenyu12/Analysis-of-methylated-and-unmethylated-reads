import fastdtw_cython
import fastdtw
import t_fastdtw
import time
import numpy as np
import timeit

poly1=np.polynomial.polynomial.Polynomial((3,4,1,-1,0.5,1,-1/3,1))
poly2=np.polynomial.polynomial.Polynomial((-1,-3,2,0.1,0.7,-2))
X_train=[]

for i in range(0,40):
    X_train.append(poly1.linspace(3000,(i,i+20))[1])
    X_train.append(poly2.linspace(3000,(i,i+20))[1])

X_train=np.array(X_train)

def test_speed_cython(n):
    s=0
    for i in range(n):
        t1=time.clock()
        fastdtw_cython.fastdtw(X_train[0],X_train[-1])
        s+=time.clock()-t1
    return s/n

def test_speed(n):
    s=0
    for i in range(n):
        t1=time.clock()
        fastdtw.fastdtw(X_train[0],X_train[-1])
        s+=time.clock()-t1
    return s/n

def test_speed2(n):
    s=0
    for i in range(n):
        t1=time.clock()
        t_fastdtw.fastdtw(X_train[0],X_train[-1])
        s+=time.clock()-t1
    return s/n

'''#print (test_speed2(100))
s1=test_speed(1000)
s2=test_speed_cython(1000)
print (s1)
print (s1/s2)'''
long_array1=np.sin(np.linspace(0,100,100))
long_array2=np.cos(np.linspace(0,100,100))
def main():
    fastdtw.fastdtw(long_array1,long_array2)#X_train[0],X_train[-1])

if __name__=="__main__":
    print (timeit.timeit("main()","from __main__ import main",number=10))

import numpy as np
import matplotlib.pyplot as plt

#polynomial
poly1=np.polynomial.polynomial.Polynomial((3,4,1,-1,0.5,1,-1/3,1))
poly2=np.polynomial.polynomial.Polynomial((-1,-3,2,0.1,0.7,-2))
X_train=[]

for i in range(0,40):
    X_train.append(poly1.linspace(3000,(i,i+20))[1])
    X_train.append(poly2.linspace(3000,(i,i+20))[1])

X_train=np.array(X_train)

#sinus's
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

expand_window_input=([(0, 0), (1, 0), (2, 1), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2), (10, 3), (11, 4), (12, 5), (13, 6), (14, 7), (15, 8), (16, 9), (17, 10), (18, 11), (19, 12), (20, 13), (21, 14), (22, 15), (23, 16), (24, 17), (25, 18), (26, 19), (27, 20), (28, 21), (29, 22), (30, 23), (31, 24), (32, 25), (33, 26), (34, 27), (35, 28), (36, 29), (37, 30), (38, 31), (39, 32), (40, 33), (41, 34), (42, 35), (43, 36), (44, 37), (45, 38), (46, 39), (47, 40), (48, 41), (48, 42), (48, 43), (48, 44), (48, 45), (48, 46), (48, 47), (49, 48), (49, 49)],100,100,1)

def plot_X_train():
    for x in X_train:
        plt.plot(range(x.shape[0]),x)
    plt.show()

def plot_series():
    for x in series:
        plt.plot(range(x.shape[0]),x)
    plt.show()

if __name__=="__name__":
    plot_X_train()

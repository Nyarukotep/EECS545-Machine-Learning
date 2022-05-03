import numpy as np
from scipy.io.wavfile import write
Fs = 11025

def normalize(dat):
    return 0.99 * dat / np.max(np.abs(dat))

def load_data():
    mix = np.loadtxt('q5data/q5.dat')
    return mix

def sigmoid(x):
    """
    A numerically stable sigmoid function for the input x.
    
    It calculates positive and negative elements with different equations to 
    prevent overflow by avoid exponentiation with large positive exponent, 
    thus achieving numerical stability.
    
    For negative elements in x, sigmoid uses this equation
    
    $$sigmoid(x_i) = \frac{e^{x_i}}{1 + e^{x_i}}$$
    
    For positive elements, it uses another equation:
    
    $$sigmoid(x_i) = \frac{1}{e^{-x_i} + 1}$$
    
    The two equations are equivalent mathematically. 
    
    x is of shape: B x H
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)

    # specify dtype! otherwise, it may all becomes zero, this could have different
    # behaviors depending on numpy version
    z = np.zeros_like(x, dtype=float)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])

    top = np.ones_like(x, dtype=float)
    top[neg_mask] = z[neg_mask]
    s = top / (1 + z)
    ### END YOUR CODE
    return s

def unmixer(X):
    # M: length
    # N: number of microphones
    M, N = X.shape
    W = np.eye(N)
    losses = []

    anneal = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01,
              0.005, 0.005, 0.002, 0.002, 0.001, 0.001]
    print('Separating tracks ...')
    ######## Your code here ##########
    for alpha in anneal:
        print('working on alpha = {0}'.format(alpha))
        for xi in X:
            p1 = np.outer(1 - 2 * sigmoid(np.dot(W, xi.T)), xi)
            p2 = np.linalg.inv(W.T)
            W += alpha * (p1 + p2)
    ###################################
    return W

def unmix(X, W):
    ######### Your code here ##########
    S = np.dot(X, W.T)
    ##################################
    return S


X = normalize(load_data())
print('Saving mixed track 1')
write('q5_mixed_track_1.wav', Fs, X[:, 0])

import time
t0 = time.time()
W = unmixer(X) # This will take around 2min
print('time=', time.time()-t0)
S = normalize(unmix(X, W))

for track in range(5):
    print(f'Saving unmixed track {track}')
    write(f'q5_unmixed_track_{track}.wav', Fs, S[:, track])

print('W solution:')
print(W)

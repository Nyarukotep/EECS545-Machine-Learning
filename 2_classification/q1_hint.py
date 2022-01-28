import numpy as np
import math
import matplotlib.pyplot as plt


def main():
    # We format the data matrix so that each row is the feature for one sample.
    # The number of rows is the number of data samples.
    # The number of columns is the dimension of one data sample.
    X = np.load('q1x.npy')
    N = X.shape[0]
    Y = np.load('q1y.npy')
    # To consider intercept term, we append a column vector with all entries=1.
    # Then the coefficient correpsonding to this column is an intercept term.
    X = np.concatenate((np.ones((N, 1)), X), axis=1)
    

if __name__ == "__main__":
    main()
        

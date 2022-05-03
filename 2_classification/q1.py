import numpy as np
import math
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

## Iterative
def log_regression(X, Y):
    #rows of X are training samples
    #rows of Y are corresponding 0/1 lables
    #newton method: w = w - inv(H)*grad
    #import ipdb; ipdb.set_trace()
    N, d = X.shape
    w = np.zeros(d)
    max_iters = 100
    for iter in range(max_iters):
        grad = np.zeros(d)
        H = np.zeros((d, d))
        for i in range(N):
            hxi = sigmoid(np.dot(X[i,:], w))
            grad = grad + X[i,:]*(Y[i]-hxi)
            H = H - hxi*(1-hxi)*np.matmul(np.transpose(X[i,:].reshape((1, d))), X[i,:].reshape((1,d)))
        w = w - np.matmul(np.linalg.inv(H), grad)
    return w

## Vectorized
def log_regression(X, Y):
    #rows of X are training samples
    #rows of Y are corresponding 0/1 lables
    #newton method: w = w - inv(H)*grad
    #import ipdb; ipdb.set_trace()
    N, d = X.shape
    w = np.zeros(d)
    max_iters = 100
    for iter in range(max_iters):
        preds = sigmoid(X@w) 
        grad = X.T@(Y - preds)
        H = - X.T@ (np.expand_dims((preds * (1 - preds)), -1) * X)
        w = w - np.matmul(np.linalg.inv(H), grad)
    return w

def main():
    X = np.load('q1x.npy')
    N = X.shape[0]
    Y = np.load('q1y.npy')
    X = np.concatenate((np.ones((N, 1)), X), axis=1)
    w = log_regression(X, Y)
    print(w)
    plt.figure()
    for i in range(N):
        if Y[i] == 0:
            plt.scatter(x = X[i,1], y = X[i,2], c='r', marker='x')
        else:
            plt.scatter(x = X[i,1], y = X[i,2], c='g', marker='o')
    #border_x = [x for x in range(min(X[:,1]), max(X[:,1]), 0.01)]
    border_x = np.linspace(start=min(X[:,1]), stop=max(X[:,1]), num=50)
    border_y = - w[0]/w[2] - w[1]/w[2]*border_x
    plt.plot(border_x, border_y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig('q1.png')

if __name__ == "__main__":
    main()
        
'''
def sigmoid(x, w):
    return 1/(1 + np.exp(- x @ w))

def loglikehood(x, y, w):
    return np.sum(y.T @ np.log(sigmoid(x,w)) + (1 - y.T) @ np.log(1 - np.log(sigmoid(x,w))))

def gradient(x, y, w):
    grad = -x.T @ (y - sigmoid(x, w))
    return grad

def hessian(x, y, w):
    m = sigmoid(x,w)*(1-sigmoid(x,w)).T
    h = -x.T @ np.diag(np.diag(m)) @ x
    return h
def newton(x, y, w, num_iter = 100):
    loss = np.zeros(100)
    loss[0]= 1
    i = 1
    while i < num_iter-1:
        i += 1
        w += np.linalg.inv(hessian(x, y, w)) @ gradient(x, y, w)
        loss[i] = - 1/99 * loglikehood(x, y, w)
    print("{}".format(w))
    plt.figure(figsize = (5,5), facecolor='white')
    plt.plot(range(100), loss,  c = "black")
    plt.show()
    return w
x = np.load('q1x.npy')
n = x.shape[0]
y = np.reshape(np.load('q1y.npy'), (x.shape[0], 1))
x = np.concatenate((np.ones((n, 1)), x), axis=1)
w = np.zeros((x.shape[1], 1))
w = newton(x, y, w)
plt.figure(figsize = (5,5), facecolor='white')
plt.scatter(x[np.argwhere(y==1),1],x[np.argwhere(y==1),2], c='r',alpha=0.5)
plt.scatter(x[np.argwhere(y==0),1],x[np.argwhere(y==0),2], c = 'b',alpha=0.5)
r = np.arange(min(x[:,1]),max(x[:,1]))
plt.plot(r, -(w[0,] + w[1, ]*r)/w[2,], c = "black")
plt.title("Logistic regression")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()'''
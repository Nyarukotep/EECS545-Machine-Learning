import numpy as np
import matplotlib.pyplot as plt

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
plt.show()
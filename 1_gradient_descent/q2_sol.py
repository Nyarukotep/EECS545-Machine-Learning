import numpy as np
import math
import matplotlib.pyplot as plt

def main():
    x = np.load('q2x.npy')
    N = x.shape[0]
    x = np.reshape(x, (N, 1))
    y = np.load('q2y.npy')
    X = np.concatenate((np.ones((N, 1)), x), axis=1)
    
    # scatter plot
    plt.scatter(x, y, marker='.', c='blue')    
 
    # linear regression
    tmp = np.matmul(np.transpose(X), X)
    tmp = np.linalg.inv(tmp)
    tmp = np.matmul(tmp, np.transpose(X))
    theta = np.matmul(tmp, y)
    line_x = np.linspace(start=min(X[:,1]), stop=max(X[:,1]), num=50)
    line_y = theta[1]*line_x + theta[0]
    plt.plot(line_x, line_y, c='blue', label = 'linear')
      
    # locally weighted linear regression
    taus = [0.1, 0.3, 0.8, 2, 10]
    colors = ['r', 'g', 'm', 'y', 'k']
    N = X.shape[0]
    for j, tau in enumerate(taus):
        for k in range(len(line_x)):
            W = np.zeros((N, N)) 
            for i in range(N):
                W[i,i] = np.exp(-(line_x[k]-x[i])**2/(2*(tau**2)))
            tmp = np.matmul(np.transpose(X), W)
            tmp = np.matmul(tmp, X)
            tmp = np.linalg.inv(tmp)
            tmp = np.matmul(tmp, np.transpose(X))
            tmp = np.matmul(tmp, W)
            theta = np.matmul(tmp, y)
            line_y[k] = theta[1]*line_x[k] + theta[0]
        plt.plot(line_x, line_y, c=colors[j], label='tau=%g'%tau)
    plt.legend()
    plt.savefig('q2.png')
    plt.close()

if __name__ == "__main__":
    main()
        

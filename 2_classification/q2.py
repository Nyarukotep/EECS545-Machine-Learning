import numpy as np
import matplotlib.pyplot as plt

q2_data = np.load('q2_data.npz')
trainx = q2_data['q2x_train']
trainy = q2_data['q2y_train']
testx = q2_data['q2x_test']
testy = q2_data['q2y_test']

N = trainx.shape[0]
M = trainx.shape[1]
K = len(np.unique(testy))
W = np.zeros((K, M), dtype=np.float32)

def compute_softmax_probs(W, x):
    probs = np.exp(np.matmul(W, x.T))
    probs = probs/sum(probs)
    return np.reshape(probs, (K, 1))

alpha = 0.0005
for i in range(300):
    delta_W = np.zeros((K, M), dtype=np.float32)
    for j in range(N):
        indicator = np.zeros((K,1))
        indicator[int(trainy[j,0])-1] = 1
        probs = compute_softmax_probs(W, trainx[j, :])
        delta_W = delta_W + np.multiply(np.tile(trainx[j, :], (K, 1)), np.tile((indicator - probs), (1, M)))
    
    W_new = W + alpha * delta_W
    W[:K-1, :] = W_new[:K-1, :]

corr = 0
N_test = testx.shape[0]
for i in range(N_test):
    probs = compute_softmax_probs(W, testx[i, :])
    idx = np.argmax(probs)
    if (idx+1 == testy[i]):
        corr += 1

accuracy = corr / N_test
print('The accuracy is: ', 100*accuracy, '%')
from sklearn.linear_model import LogisticRegression
mod = LogisticRegression(multi_class='multinomial',solver ='newton-cg')
mod.fit(trainx,trainy.ravel())

pre = mod.predict(testx)
err = sum(abs(pre - testy.ravel()))
accuracy = 100*(1 - err/testy.shape[0])

print('The LogisticRegression accuracy of sklearn is', accuracy, '%')
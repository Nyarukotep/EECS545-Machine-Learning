import numpy as np
import matplotlib.pyplot as plt

# Load data
q2_data = np.load('q2_data.npz')
q2x_train = q2_data['q2x_train']
q2x_test = q2_data['q2x_test']
q2y_train = q2_data['q2y_train']
q2y_test = q2_data['q2y_test']

# TRAINING 
N = q2x_train.shape[0]        # number of samples in training dataset
M = q2x_train.shape[1]        # dimension of feature
K = len(np.unique(q2y_test))  # number of class labels
W = np.zeros((K, M), dtype=np.float32)

# Computes probabilities for x being each class. 
def compute_softmax_probs(W, x):
    # W : K * M matrix (the last row is a zero vector)
    # x : 1 * M
    probs = np.exp(np.matmul(W, x.T))
    probs = probs/sum(probs)
    
    return np.reshape(probs, (K, 1))

# TRAINING 
alpha = 0.0005
count_c = 0
count_iteration = 0
while True:
    # A single iteration over all training examples
    delta_W = np.zeros((K, M), dtype=np.float32)
    for i in range(N):
        indicator = np.zeros((K,1), dtype=np.int32)
        indicator[int(q2y_train[i,0])-1] = 1
        probs = compute_softmax_probs(W, q2x_train[i, :])
        
        delta_W = delta_W + np.multiply(np.tile(q2x_train[i, :], (K, 1)), np.tile((indicator - probs), (1, M)))
    
    W_new = W + alpha * delta_W
    W[:K-1, :] = W_new[:K-1, :]
    
    # Stopping criteria
    count_c += 1 if count_iteration > 300 and np.sum(abs(alpha * delta_W)) < 0.05 else 0
    if count_c > 5:
        break
    count_iteration += 1

# Compute accuracy
count_correct = 0
N_test = q2x_test.shape[0]
for i in range(N_test):
    probs = compute_softmax_probs(W, q2x_test[i, :])
    idx = np.argmax(probs)
    if (idx+1 == q2y_test[i]):
        count_correct += 1

accuracy = count_correct / N_test
print('The accuracy of Softmax Regression - our implementation: ', 100*accuracy, '%')

from sklearn.linear_model import LogisticRegression

# Load data
q2_data = np.load('q2_data.npz')
q2x_train = q2_data['q2x_train']
q2x_test = q2_data['q2x_test']
q2y_train = q2_data['q2y_train']
q2y_test = q2_data['q2y_test']

# [Accuracy depends on the solver] newton-cg and lbgs: 92%, sag: 96%, saga: 94%
MLR = LogisticRegression(multi_class='multinomial',solver ='newton-cg')
MLR.fit(q2x_train,q2y_train.reshape(q2x_train.shape[0],))

# Generate predictions and compute accuracy
preds = MLR.predict(q2x_test)
error = sum(abs(preds - q2y_test.reshape(q2y_test.shape[0],)))
accuracy = 100*(1 - error/q2y_test.shape[0])

print('The accuracy of Sklearn Logistic Regression is: ', accuracy, '%')

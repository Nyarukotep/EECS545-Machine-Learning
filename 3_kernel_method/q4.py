# EECS 545 HW3 Q4
# Your name: Aabhaas Vaish (aabhaas@umich.edu)

import numpy as np
import matplotlib.pyplot as plt

# Instruction: use these hyperparameters for both (b) and (d)
eta = 0.5
C = 5
iterNums = [5, 50, 100, 1000, 5000, 6000]


def get_gradient(state:dict, matrix: np.ndarray, label: np.ndarray, C: int):
    N, D = matrix.shape
    indicator_w = np.multiply(np.multiply(np.multiply(np.transpose(state['W'] @ matrix.T) + state['b'], label) < 1, label), matrix)
    indicator_b = np.multiply(np.multiply(np.transpose(state['W'] @ matrix.T) + state['b'], label) < 1, label)
    delta_w = state['W'] -1*C*(np.sum(indicator_w, axis=0))
    delta_b = -1*C*(np.sum(indicator_b, axis=0))
    
    return delta_w, delta_b


def svm_train_bgd(matrix: np.ndarray, label: np.ndarray, nIter: int):
    # Implement your algorithm and return state (e.g., learned model)
    state = {}
    N, D = matrix.shape

    ##################################
    # TODO: Implement your code here #
    ##################################
    state['W'] = np.zeros((1, D)) 
    state['b'] = np.float64(0)
    
    for j in range(1, nIter+1):
        delta_w, delta_b = get_gradient(state, matrix, label, C)
        state['W'] -= (eta/(1 + j*eta))*delta_w
        state['b'] -= 0.01*(eta/(1 + j*eta))*delta_b
        
    return state


def svm_train_sgd(matrix: np.ndarray, label: np.ndarray, nIter: int):
    # Implement your algorithm and return state (e.g., learned model)
    state = {}
    N, D = matrix.shape

    ##################################
    # TODO: Implement your code here #
    ##################################
    state['W'] = np.zeros((1, D)) 
    state['b'] = np.float64(0)
    
    for j in range(1, nIter+1):
        perm = np.random.permutation(N)
        for i in perm:
            delta_w = state['W']/N - C*(label[i,0]*(state['W'] @ np.transpose(matrix[i, :]) + state['b']) < 1)*label[i,0]*matrix[i, :]
            delta_b = -1*C*(label[i,0]*(state['W'] @ np.transpose(matrix[i, :]) + state['b']) < 1)*label[i,0]
            state['W'] -= (eta/(1 + j*eta))*delta_w
            state['b'] -= 0.01*(eta/(1 + j*eta))*delta_b
        
    return state


def svm_test(matrix: np.ndarray, state):
    # Classify each test data as +1 or -1
    output = np.ones( (matrix.shape[0], 1) )

    ##################################
    # TODO: Implement your code here #
    ##################################
    output = (np.transpose(state['W'] @ matrix.T) + state['b'] > 0)*2 - 1
    return output


def evaluate(output: np.ndarray, label: np.ndarray, nIter: int) -> float:
    # Use the code below to obtain the accuracy of your algorithm
    accuracy = (label * output > 0).sum() * 1. / len(output)
    print('Iter {:4d}: test accuracy = {:2.4f}%'.format(nIter, 100 * accuracy))

    return accuracy


def load_data():
    # Note1: label is {-1, +1}
    # Note2: data matrix shape  = [Ndata, 4]
    # Note3: label matrix shape = [Ndata, 1]

    # Load data
    q4_data = np.load('q4_data/q4_data.npy', allow_pickle=True).item()

    train_x = q4_data['q4x_train']
    train_y = q4_data['q4y_train']
    test_x = q4_data['q4x_test']
    test_y = q4_data['q4y_test']
    return train_x, train_y, test_x, test_y


def run_bgd(train_x, train_y, test_x, test_y):
    '''(c) Implement SVM using **batch gradient descent**.
    For each of the nIter's, print out the following:

    *   Parameter w
    *   Parameter b
    *   Test accuracy (%)
    '''
    for nIter in iterNums:
        # Train
        state = svm_train_bgd(train_x, train_y, nIter)
        
        # TODO: Test and evluate
        prediction = svm_test(test_x, state)
        evaluate(prediction, test_y, nIter)
        print("Trained W values are", np.squeeze(state['W']))
        print("Trained b value is", state['b'])
        print("")

def run_sgd(train_x, train_y, test_x, test_y):
    '''(c) Implement SVM using **stocahstic gradient descent**.
    For each of the nIter's, print out the following:

    *   Parameter w
    *   Parameter b
    *   Test accuracy (%)

    [Note: Use the same hyperparameters as (b)]
    [Note: If you implement it correctly, the running time will be ~15 sec]
    '''
    for nIter in iterNums:
        # Train
        state = svm_train_sgd(train_x, train_y, nIter)
        
        # TODO: Test and evluate
        prediction = svm_test(test_x, state)
        evaluate(prediction, test_y, nIter)
        print("Trained W values are", np.squeeze(state['W']))
        print("Trained b value is", state['b'])
        print("")


def main():
    train_x, train_y, test_x, test_y = load_data()
    print("---------------------------------------")
    print("Batch Gradient Descent")
    print("---------------------------------------")
    
    run_bgd(train_x, train_y, test_x, test_y)
    
    print("---------------------------------------")
    print("Stochastic Gradient Descent")
    print("---------------------------------------")
    
    run_sgd(train_x, train_y, test_x, test_y)


if __name__ == '__main__':
    main()